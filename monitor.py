"""Herramientas de diagnóstico del equipo. Solo leen: se pueden lanzar con el recomendador corriendo.

    python monitor.py check          # ¿funciona la conexión? filas por tabla y ejemplos de datos
    python monitor.py conv [horas]   # conversión por versión del modelo vs sistema antiguo (def. 1 h)
    python monitor.py latency [min]  # fiabilidad: a tiempo vs fallback (missing/late/invalid) y latencias
    python monitor.py lag [seg]      # cuánto tarda una request en aparecer en shop.requests (def. 20 s)
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
LEFT JOIN (SELECT request_id::String AS request_id, any(model_version::String) AS model_version
           FROM {team}.submissions GROUP BY request_id) AS s
    ON s.request_id = f.request_id::String
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


COMMANDS = {"check": check, "conv": conv, "latency": latency, "lag": lag}

if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["all"]
    if cmd == "all":
        check(); conv(); latency()
    elif cmd in COMMANDS:
        COMMANDS[cmd](*map(int, args))
    else:
        print(__doc__)
