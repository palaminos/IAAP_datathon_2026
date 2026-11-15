"""Herramientas de diagnóstico del equipo. Solo leen: se pueden lanzar con el recomendador corriendo.

    python monitor.py check          # ¿funciona la conexión? filas por tabla y ejemplos de datos
    python monitor.py conv [horas]   # conversión por versión del modelo vs sistema antiguo (def. 1 h)
    python monitor.py latency [min]  # fiabilidad: a tiempo vs fallback (missing/late/invalid) y latencias
    python monitor.py lag [seg]      # cuánto tarda una request en aparecer en shop.requests (def. 20 s)
    python monitor.py lb [min]       # LA MEDIDA BUENA: conversión esperada por versión según el leaderboard (def. 60 min)
    python monitor.py lbs [HH:MM]    # lo mismo desde una hora (aunque sean más de 3 h), con margen de error y z
    python monitor.py ab             # LA QUE DECIDE: cada versión frente a la campeona, por tramos entre despliegues
    python monitor.py all            # check + conv + latency

Lee TEAM_ID y RAWTREE_API_KEY de .env.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import requests

for line in Path(".env").read_text().splitlines() if Path(".env").exists() else []:
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())
API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ["TEAM_ID"]
http = requests.Session()
http.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"


def query(sql: str, timeout: float = 120) -> list[dict]:
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
    return r.json()["data"]


def show(title: str, sql: str) -> None:
    print(f"\n=== {title} ===")
    try:
        for row in query(sql):
            print(json.dumps(row, ensure_ascii=False, default=str)[:300])
    except Exception as e:
        print(f"  (falla: {e})")


def table(rows: list[dict], cols: list[tuple[str, str, str]]) -> None:
    """cols = [(clave, cabecera, formato)]"""
    print("  " + "".join(f"{h:>12}" if i else f"{h:<22}" for i, (_, h, _) in enumerate(cols)))
    for r in rows:
        print("  " + "".join(f"{format(r[k] if not i else float(r[k]), f):>12}" if i else f"{str(r[k]):<22}"
                             for i, (k, _, f) in enumerate(cols)))


# --------------------------------------------------------------------------- check
def check() -> None:
    """Conexión, tamaño de cada tabla y ejemplos de los datos."""
    print(f"API={API}  TEAM={TEAM}")
    for t in ["shop.catalog", "shop.history_sessions", "shop.history_events", "shop.requests", "shop.events",
              "shop.catalog_updates", f"{TEAM}.feedback", f"{TEAM}.submissions", f"{TEAM}.scratch"]:
        show(f"filas en {t}", f"SELECT count() AS n FROM {t}")
    show("productos por categoría", "SELECT category, count() AS n FROM shop.catalog GROUP BY category ORDER BY n DESC")
    show("ejemplo de producto", "SELECT * FROM shop.catalog LIMIT 2")
    show("rango temporal del histórico",
         "SELECT min(ts::DateTime64(3)) AS desde, max(ts::DateTime64(3)) AS hasta FROM shop.history_events")
    show("últimas 3 requests en vivo", "SELECT * FROM shop.requests ORDER BY ts DESC LIMIT 3")
    show("últimos cambios de catálogo", "SELECT ts, change, product_id FROM shop.catalog_updates ORDER BY ts DESC LIMIT 5")


# --------------------------------------------------------------------------- conv
OLD_SYSTEM = """
SELECT 'sistema antiguo' AS version, count() AS requests,
       100 * countIf(clicked != '') / count()   AS ctr_pct,
       100 * countIf(purchased != '') / count() AS conversion_pct,
       sum(revenue_eur::Float64) / count()      AS eur_per_request
FROM shop.history_sessions"""

# model_version está en submissions y el resultado en feedback: se unen por request_id.
# Lo que el comprador no vio de vuestro modelo (tarde, inválido, sin respuesta) sale como 'fallback'.
OURS = """
SELECT if(f.served_by::String = 'team', s.model_version, 'fallback') AS version,
       count()                                    AS requests,
       100 * countIf(f.clicked != '') / count()   AS ctr_pct,
       100 * countIf(f.purchased != '') / count() AS conversion_pct,
       sum(f.revenue_eur::Float64) / count()      AS eur_per_request
FROM {team}.feedback AS f
LEFT JOIN (SELECT request_id::String AS request_id, toJSONString(items) AS shown,
                  any(model_version::String) AS model_version
           FROM {team}.submissions GROUP BY request_id, shown) AS s
    -- unir también por la lista mostrada: si una request tuvo dos respuestas, cuenta la que se vio
    ON s.request_id = f.request_id::String AND s.shown = toJSONString(f.shown)
WHERE f.request_id != '__init__' AND f.ts::DateTime64(3) > now() - INTERVAL {hours} HOUR
GROUP BY version
ORDER BY requests DESC"""

CONV_COLS = [("version", "versión", ""), ("requests", "requests", ".0f"), ("ctr_pct", "CTR %", ".2f"),
             ("conversion_pct", "conv %", ".2f"), ("eur_per_request", "€/req", ".3f")]


def conv(hours: int = 1) -> None:
    """Conversión por versión del modelo (incluye las variantes del test A/B)."""
    print("\nReferencia: sistema antiguo (histórico de 60 días)")
    table(query(OLD_SYSTEM), CONV_COLS)
    print(f"\nVosotros ({TEAM}): últimas {hours} h")
    rows = query(OURS.format(team=TEAM, hours=hours))
    table(rows, CONV_COLS) if rows else print("  (aún no hay feedback)")
    print("\nCon menos de ~5.000 requests por versión las diferencias pueden ser suerte.")


# --------------------------------------------------------------------------- latency
RELIABILITY = """
SELECT if(served_by::String = 'team', 'a tiempo', concat('fallback: ', fallback_reason::String)) AS version,
       count()                              AS requests,
       100 * count() / sum(count()) OVER () AS pct,
       quantile(0.5)(latency_ms::Float64)   AS p50_ms,
       quantile(0.95)(latency_ms::Float64)  AS p95_ms,
       quantile(0.99)(latency_ms::Float64)  AS p99_ms
FROM {team}.feedback
WHERE request_id != '__init__' AND ts::DateTime64(3) > now() - INTERVAL {minutes} MINUTE
GROUP BY version
ORDER BY requests DESC"""


def latency(minutes: int = 15) -> None:
    """Fiabilidad (10 puntos): qué % llega a tiempo y por qué falla el resto. Objetivo: >99 % a tiempo."""
    print(f"\nFiabilidad de {TEAM}: últimos {minutes} min (límite 3000 ms)")
    rows = query(RELIABILITY.format(team=TEAM, minutes=minutes))
    table(rows, [("version", "estado", ""), ("requests", "requests", ".0f"), ("pct", "%", ".1f"),
                 ("p50_ms", "p50 ms", ".0f"), ("p95_ms", "p95 ms", ".0f"), ("p99_ms", "p99 ms", ".0f")]) \
        if rows else print("  (aún no hay feedback)")
    print("  missing = no respondisteis · late = llegó tarde · invalid = producto agotado/repetido/no son 10")


# --------------------------------------------------------------------------- lag
def lag(seconds: int = 20) -> None:
    """Retraso entre el ts de una request y el momento en que la vemos. Lo que queda de 3 s es vuestro."""
    seen: set[str] = set()
    lags: list[float] = []
    first = True
    end = time.time() + seconds
    while time.time() < end:
        rows = query("SELECT request_id, ts FROM shop.requests ORDER BY ts DESC LIMIT 20")
        now = dt.datetime.now(dt.timezone.utc)
        for r in rows:
            if r["request_id"] not in seen:
                seen.add(r["request_id"])
                if not first:  # la primera tanda ya existía antes de empezar a medir
                    ts = dt.datetime.fromisoformat(str(r["ts"])[:23]).replace(tzinfo=dt.timezone.utc)
                    lags.append((now - ts).total_seconds() * 1000)
        first = False
        time.sleep(0.1)
    if not lags:
        print("no han llegado requests nuevas")
        return
    lags.sort()
    print(f"retraso hasta ver una request (n={len(lags)}): p50={lags[len(lags) // 2]:.0f} ms  "
          f"p90={lags[int(len(lags) * .9)]:.0f} ms  max={lags[-1]:.0f} ms")
    print("(depende también de que el reloj del portátil esté en hora)")


# --------------------------------------------------------------------------- lb
LEADERBOARD = "https://datathon.rawtree.dev/api/state?by=model"


def leaderboard(minutes: int = 60) -> tuple[dict, dict | None]:
    """Datos públicos del leaderboard por versión del modelo, para una ventana de 15 a 180 minutos.

    Es la medida que puntúa: compras ESPERADAS por 100 requests, calculadas con la probabilidad de compra
    del simulador para lo que enseñó cada versión, no con las compras reales. Por eso tiene muchísimo menos
    ruido que `conv` (que cuenta compras reales): con ~1.500 requests por versión ya distingue un 10 %.
    Solo se pueden comparar versiones dentro de la MISMA ventana (el tráfico cambia a lo largo del día)."""
    import urllib.request
    span = min(max(int(minutes) * 60, 900), 10800)
    to = int(time.time()) // 300 * 300
    req = urllib.request.Request(f"{LEADERBOARD}&from={to - span}&to={to}", headers={"User-Agent": "datathon-client"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    return d, next((t for t in d["teams"] if t["team_id"] == TEAM), None)


def lb(minutes: int = 60) -> None:
    d, me = leaderboard(minutes)
    hhmm = lambda t: dt.datetime.fromtimestamp(t).strftime("%H:%M")
    ranked = sorted(d["teams"], key=lambda t: -t["conversion"])
    print(f"\nLeaderboard, ventana {hhmm(d['window'][0])}–{hhmm(d['window'][1])} (conversión ESPERADA, la que puntúa)")
    print(f"  baseline {d['baseline']['conversion']:.3f} % | referencia de los organizadores {d['reference']['conversion']:.3f} % | "
          f"mejor equipo {ranked[0]['conversion']:.3f} % ({ranked[0]['team_id']})")
    if me is None:
        print(f"  {TEAM} no aparece en esta ventana")
        return
    pos = 1 + sum(t["conversion"] > me["conversion"] for t in d["teams"])
    print(f"  {TEAM}: {me['conversion']:.3f} % (puesto {pos} de {len(d['teams'])} por conversión) | cold {me['cold_conversion']:.3f} % | a tiempo {me['on_time']:.1f} %")
    print(f"  {'versión':<22}{'requests':>10}{'conv %':>9}{'cold %':>9}{'€/req':>8}")
    for v in sorted(me.get("versions", []), key=lambda v: -v["requests"]):
        print(f"  {v['model_version']:<22}{v['requests']:>10}{v['conversion']:>9.3f}{v['cold_conversion']:>9.3f}{v['revenue_per_request']:>8.3f}")
    print("  Con menos de ~1.000 requests por versión, diferencias de 0,15 puntos aún pueden ser ruido.")


NOISE_SD = 5.9   # desviación típica de la conversión esperada de UNA request, en puntos. Medida la noche del
# sábado con las diferencias entre versiones que corrieron a la vez, en ventanas de 15 minutos (analysis/ab_noise.py,
# 210 comparaciones): total 5,9 · compradores conocidos 4,8 · cold 7,5. Hasta las 05:15 del domingo se usó 4,0 (medido
# con solo dos versiones gemelas): todas las z anteriores están infladas un 50 %.
NOISE_KNOWN, NOISE_COLD = 4.8, 7.5


def _fetch(a: int, b: int) -> dict:
    import urllib.request
    req = urllib.request.Request(f"{LEADERBOARD}&from={a}&to={b}", headers={"User-Agent": "datathon-client"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def since(start: float, end: float | None = None) -> tuple[dict, dict, tuple[int, int]]:
    """Suma de ventanas del leaderboard entre `start` y `end` (epoch), para periodos de más de 3 horas.
    Devuelve ({versión: {n, conv, cold_n, cold, known}}, {"reference": %, "baseline": %}, (desde, hasta)).
    `known` = conversión con compradores conocidos (los que no son cold start)."""
    to = int(end or time.time()) // 300 * 300
    frm = min(-(-int(start) // 300) * 300, to - 900)
    k = -(-(to - frm) // 10800)
    cuts = [frm + round((to - frm) * i / k / 300) * 300 for i in range(k)] + [to]
    acc: dict[str, dict] = {}
    ref = {"reference": [0.0, 0], "baseline": [0.0, 0]}
    for a, b in zip(cuts, cuts[1:]):
        d = _fetch(a, b)
        for key in ref:
            ref[key][0] += d[key]["conversion"] * d[key]["requests"]
            ref[key][1] += d[key]["requests"]
        me = next((t for t in d["teams"] if t["team_id"] == TEAM), None)
        for v in (me or {}).get("versions", []):
            x = acc.setdefault(v["model_version"], {"n": 0, "s": 0.0, "cold_n": 0, "cold_s": 0.0})
            x["n"] += v["requests"]
            x["s"] += v["conversion"] * v["requests"]
            x["cold_n"] += v.get("cold_requests", 0)
            x["cold_s"] += v.get("cold_conversion", 0.0) * v.get("cold_requests", 0)
    out = {}
    for name, x in acc.items():
        kn = x["n"] - x["cold_n"]
        out[name] = {"n": x["n"], "conv": x["s"] / x["n"], "cold_n": x["cold_n"],
                     "cold": x["cold_s"] / x["cold_n"] if x["cold_n"] else float("nan"),
                     "known": (x["s"] - x["cold_s"]) / kn if kn else float("nan")}
    return out, {k: v[0] / v[1] if v[1] else float("nan") for k, v in ref.items()}, (frm, to)


def zscore(a: dict, b: dict) -> float:
    """Cuántos márgenes de error separan la conversión de `a` de la de `b` (|z| < 2: puede ser suerte)."""
    return (a["conv"] - b["conv"]) / (NOISE_SD * (1 / a["n"] + 1 / b["n"]) ** 0.5)


def compare(start: float, end: float | None = None, champion: str | None = None) -> list[str]:
    """Líneas de texto con la comparación de versiones entre `start` y `end`, cada una frente a la campeona."""
    vs, ref, (a, b) = since(start, end)
    hhmm = lambda t: dt.datetime.fromtimestamp(t).strftime("%H:%M")
    lines = [f"leaderboard {hhmm(a)}–{hhmm(b)} (conversión esperada) | referencia {ref['reference']:.3f} % | baseline {ref['baseline']:.3f} %"]
    if not vs:
        return lines + ["  (sin versiones con suficientes requests)"]
    champ = champion if champion in vs else max(vs, key=lambda k: vs[k]["n"])
    lines.append(f"  {'versión':<22}{'requests':>9}{'conv %':>8}{'± error':>9}{'conocidos':>11}{'cold':>8}{'vs campeona':>13}{'z':>7}")
    for name, x in sorted(vs.items(), key=lambda kv: -kv[1]["conv"]):
        z = "" if name == champ else f"{zscore(x, vs[champ]):+.1f}"
        diff = "(campeona)" if name == champ else f"{x['conv'] - vs[champ]['conv']:+.3f}"
        lines.append(f"  {name:<22}{x['n']:>9}{x['conv']:>8.3f}{NOISE_SD / x['n'] ** 0.5:>9.3f}{x['known']:>11.3f}{x['cold']:>8.3f}{diff:>13}{z:>7}")
    return lines


def stratified(periods: list[str], champion: str, end: float | None = None, control: list[str] | None = None) -> list[str]:
    """Comparación de cada versión con la campeona POR TRAMOS (entre dos despliegues el reparto no cambia).

    Si el reparto cambia a mitad de la noche, sumar todo mezclaría horas distintas con pesos distintos para
    cada versión (y la conversión cambia con la hora). Aquí se calcula la diferencia con la campeona dentro
    de cada tramo y se combinan los tramos pesando cada uno por su precisión. `periods` = horas de inicio de
    cada tramo ('YYYY-MM-DD HH:MM', hora local); el último llega hasta ahora."""
    ts = [dt.datetime.strptime(x, "%Y-%m-%d %H:%M").timestamp() for x in periods] + [end or time.time()]
    acc: dict[str, dict] = {}
    for a, b in zip(ts, ts[1:]):
        if b - a < 900:      # el leaderboard no da ventanas de menos de 15 minutos
            continue
        vs, _, _ = since(a, b)
        # control: la campeona sola o, si se indica, varias versiones equivalentes sumadas (menos ruido)
        group = [vs[k] for k in (control or [champion]) if k in vs]
        if not group:
            continue
        tot = sum(g["n"] for g in group)
        cold_n = sum(g["cold_n"] for g in group)
        conv = sum(g["conv"] * g["n"] for g in group) / tot
        cold = sum(g["cold"] * g["cold_n"] for g in group if g["cold_n"]) / max(cold_n, 1)
        c = {"n": tot, "conv": conv, "cold": cold, "cold_n": cold_n, "known": (conv * tot - cold * cold_n) / max(tot - cold_n, 1)}
        for name, x in vs.items():
            if name in (control or [champion]):
                continue
            var = NOISE_SD ** 2 * (1 / x["n"] + 1 / c["n"])
            r = acc.setdefault(name, {"w": 0.0, "wd": 0.0, "n": 0, "nc": 0, "s": 0.0, "sc": 0.0, "kd": 0.0, "cd": 0.0, "tramos": 0,
                                      "kw": 0.0, "kwd": 0.0, "cw": 0.0, "cwd": 0.0})
            r["w"] += 1 / var
            r["wd"] += (x["conv"] - c["conv"]) / var
            r["kd"] += (x["known"] - c["known"]) / var
            r["cd"] += (x["cold"] - c["cold"]) / var
            # cada segmento con su propio ruido y sus propias requests (para la z de conocidos y la de cold)
            xk, ck = x["n"] - x["cold_n"], c["n"] - c["cold_n"]
            if xk > 0 and ck > 0 and x["cold_n"] > 0 and c["cold_n"] > 0:
                vk, vc = NOISE_KNOWN ** 2 * (1 / xk + 1 / ck), NOISE_COLD ** 2 * (1 / x["cold_n"] + 1 / c["cold_n"])
                r["kw"] += 1 / vk; r["kwd"] += (x["known"] - c["known"]) / vk
                r["cw"] += 1 / vc; r["cwd"] += (x["cold"] - c["cold"]) / vc
            r["n"] += x["n"]; r["nc"] += c["n"]; r["s"] += x["conv"] * x["n"]; r["sc"] += c["conv"] * c["n"]; r["tramos"] += 1
    hhmm = lambda t: dt.datetime.fromtimestamp(t).strftime("%H:%M")
    lines = [f"cada versión frente a {' + '.join(control) if control else champion}, por tramos desde {hhmm(ts[0])} (solo los tramos en que corrieron las dos)",
             f"  {'versión':<20}{'tramos':>7}{'requests':>10}{'conv %':>8}{'campeona':>10}{'diferencia':>12}{'± error':>9}{'z':>7}{'conocidos (z)':>16}{'cold (z)':>15}"]
    for name, r in sorted(acc.items(), key=lambda kv: -kv[1]["wd"] / kv[1]["w"]):
        d, se = r["wd"] / r["w"], r["w"] ** -0.5
        if r["kw"] > 0:
            kd, cd = r["kwd"] / r["kw"], r["cwd"] / r["cw"]
            seg = f"{kd:>+9.3f} ({kd * r['kw'] ** 0.5:+.1f})".rjust(16) + f"{cd:>+8.3f} ({cd * r['cw'] ** 0.5:+.1f})".rjust(15)
        else:
            seg = f"{r['kd'] / r['w']:>+16.3f}{r['cd'] / r['w']:>+15.3f}"
        lines.append(f"  {name:<20}{r['tramos']:>7}{r['n']:>10}{r['s'] / r['n']:>8.3f}{r['sc'] / r['nc']:>10.3f}{d:>+12.3f}{se:>9.3f}{d / se:>+7.1f}{seg}")
    return lines


def ab() -> None:
    """Comparación por tramos con la campeona, según logs/night_state.json (campos champion y periods)."""
    st = json.loads(Path("logs/night_state.json").read_text())
    print("\n" + "\n".join(stratified(st["periods"], st["champion"])))
    if st.get("control"):   # versiones que los datos dan por equivalentes a la campeona: sumadas hacen de control con menos ruido
        print("\n" + "\n".join(stratified(st["periods"], st["champion"], control=st["control"])))
    print("  diferencia = puntos de conversión sobre la campeona; z = diferencia / error (|z| < 2: puede ser suerte)")
    print("  conocidos / cold = la misma diferencia solo con requests de compradores conocidos / solo con requests cold, cada una con su z")


def lbs(hhmm: str = "") -> None:
    """Comparación desde una hora dada (HH:MM de hoy, o de ayer si aún no ha llegado). Sin hora: 3 h."""
    if hhmm:
        t = dt.datetime.combine(dt.date.today(), dt.time.fromisoformat(hhmm))
        if t > dt.datetime.now():
            t -= dt.timedelta(days=1)
        start = t.timestamp()
    else:
        start = time.time() - 10800
    print("\n" + "\n".join(compare(start)))
    print("  z = diferencia medida en márgenes de error. Con |z| < 2 la diferencia puede ser suerte.")


COMMANDS = {"check": check, "conv": conv, "latency": latency, "lag": lag, "lb": lb}

if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["all"]
    if cmd == "all":
        check(); conv(); latency()
    elif cmd == "lbs":
        lbs(*args)
    elif cmd == "ab":
        ab()
    elif cmd in COMMANDS:
        COMMANDS[cmd](*map(int, args))
    else:
        print(__doc__)
