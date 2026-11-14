"""Chequeo de producción y lectura del A/B, pensado para ejecutarse periódicamente. Solo lee de RawTree.

    python night_check.py            # imprime el informe y lo añade a logs/night_report.md
    python night_check.py --quiet    # solo lo imprime

Salud: procesos, % a tiempo, duplicados y avisos del log. A/B: todas las variantes activas, comparadas
SOLO en el periodo en que han corrido todas a la vez, con su margen de error.
"""
import datetime as dt
import math
import subprocess
import sys
from pathlib import Path

from monitor import TEAM, query

PROD_LOG = Path(__file__).resolve().parent.parent / "rawtree-prod" / "logs" / "production.log"
REPORT = Path(__file__).resolve().parent / "logs" / "night_report.md"
out: list[str] = []
say = lambda s="": (out.append(s), print(s))

now = dt.datetime.now()
say(f"## {now:%Y-%m-%d %H:%M} (hora de Barcelona)")

# ---------------------------------------------------------------- salud
procs = subprocess.run(["pgrep", "-fl", "run_forever.sh|python3 -u .*recommender"], capture_output=True, text=True).stdout.strip().splitlines()
n_py = sum("python3 -u" in p and "caffeinate" not in p for p in procs)
n_launcher = sum("run_forever.sh" in p for p in procs)
r = query(f"""SELECT count() n, countIf(served_by::String = 'team') ok, countIf(fallback_reason::String = 'missing') missing,
       countIf(fallback_reason::String = 'late') late, countIf(fallback_reason::String = 'invalid') invalid,
       round(quantileIf(0.99)(latency_ms::Float64, served_by::String = 'team')) p99
FROM {TEAM}.feedback WHERE ts::DateTime64(3) > now() - INTERVAL 15 MINUTE AND request_id != '__init__'""")[0]
n = max(int(r["n"]), 1)
on_time = 100 * int(r["ok"]) / n
d = query(f"""SELECT countIf(c > 1) dup, count() total FROM (SELECT request_id::String rid, count() c FROM {TEAM}.submissions
WHERE request_id::String IN (SELECT request_id::String FROM shop.requests WHERE ts::DateTime64(3) > now() - INTERVAL 5 MINUTE) GROUP BY rid)""")[0]
warn = []
if PROD_LOG.exists():
    tail = PROD_LOG.read_text(errors="ignore").splitlines()[-400:]
    starts = [i for i, line in enumerate(tail) if "arrancando" in line]
    tail = tail[starts[-1] + 1:] if starts else tail   # solo lo ocurrido desde el último arranque
    warn = [line[:150] for line in tail if any(k in line for k in ("AVISO", "fallid", "error", "perdidas", "se ha parado", "Traceback"))][-4:]
healthy = n_py == 1 and n_launcher == 1 and on_time >= 97 and int(d["dup"]) <= 5
say(f"**SALUD: {'OK' if healthy else 'PROBLEMA'}** | recomendadores: {n_py}, lanzadores: {n_launcher} | últimos 15 min: {on_time:.1f} % a tiempo "
    f"({r['n']} requests; missing {r['missing']}, late {r['late']}, invalid {r['invalid']}; p99 {r['p99']} ms) | duplicadas 5 min: {d['dup']} de {d['total']}")
for w in warn:
    say(f"- log: `{w}`")
try:  # memoria: el Mac tiene 8 GB; si el recomendador crece mucho o el swap se llena, la latencia sufre
    pid = next(p.split()[0] for p in procs if "python3 -u" in p and "caffeinate" not in p)
    rss = int(subprocess.run(["ps", "-o", "rss=", "-p", pid], capture_output=True, text=True).stdout.strip() or 0) / 1024
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    say(f"- memoria del recomendador: {rss:.0f} MB en RAM | swap del Mac: {swap[:60]}")
except Exception:
    pass

# ---------------------------------------------------------------- A/B
active = query(f"""SELECT model_version::String v, count() n FROM {TEAM}.submissions
WHERE request_id::String IN (SELECT request_id::String FROM shop.requests WHERE ts::DateTime64(3) > now() - INTERVAL 10 MINUTE)
GROUP BY v HAVING n > 20 ORDER BY v""")
versions = [a["v"] for a in active if a["v"] != "fallback-propio"]
say(f"variantes activas: {', '.join(versions) or 'ninguna'}")
if len(versions) >= 2:
    vs = ", ".join(f"'{v}'" for v in versions)
    # inicio común: el último arranque entre las variantes activas (primera respuesta de cada una en 30 h)
    start = query(f"""SELECT toString(max(t)) s FROM (SELECT s.model_version::String v, min(r.ts::DateTime64(3)) t
FROM {TEAM}.submissions s INNER JOIN shop.requests r ON r.request_id::String = s.request_id::String
WHERE s.model_version::String IN ({vs}) AND r.ts::DateTime64(3) > now() - INTERVAL 30 HOUR GROUP BY v)""")[0]["s"]
    rows = query(f"""
SELECT s.v v, count() n, countIf(f.clicked != '') clics, countIf(f.purchased != '') compras, sum(f.revenue_eur::Float64) eur
FROM {TEAM}.feedback f
INNER JOIN (SELECT request_id::String rid, toJSONString(items) sh, any(model_version::String) v FROM {TEAM}.submissions GROUP BY rid, sh) s
  ON s.rid = f.request_id::String AND s.sh = toJSONString(f.shown)
INNER JOIN (SELECT request_id::String rid, ts::DateTime64(3) ts FROM shop.requests) r ON r.rid = f.request_id::String
WHERE f.served_by::String = 'team' AND s.v IN ({vs}) AND r.ts > toDateTime64('{start}', 3)
GROUP BY v ORDER BY v""", timeout=300)
    say(f"A/B desde {start} UTC:")
    say("| versión | compradores | CTR % | conversión % | compras | €/req |")
    say("|---|---|---|---|---|---|")
    for x in rows:
        nn, b, c = int(x["n"]), int(x["compras"]), int(x["clics"])
        p = b / nn
        say(f"| {x['v']} | {nn} | {100 * c / nn:.2f} | {100 * p:.2f} ± {196 * math.sqrt(p * (1 - p) / nn):.2f} | {b} | {float(x['eur']) / nn:.3f} |")
    base = next((x for x in rows if x["v"].startswith("v4")), rows[0])
    for x in rows:
        if x is base:
            continue
        for key, label in (("compras", "conversión"), ("clics", "CTR")):
            n1, b1, n2, b2 = int(base["n"]), int(base[key]), int(x["n"]), int(x[key])
            if b1 and b2:
                p = (b1 + b2) / (n1 + n2)
                z = (b2 / n2 - b1 / n1) / math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
                say(f"- {x['v']} vs {base['v']}, {label}: {100 * (b2 / n2) / (b1 / n1) - 100:+.0f} % (z = {z:+.2f})")
# ---------------------------------------------------------------- leaderboard (la medida que puntúa)
try:
    import json
    import time
    from monitor import compare, stratified
    state = json.loads((REPORT.parent / "night_state.json").read_text())
    lines = compare(time.time() - 3600, champion=state.get("champion"))
    say(f"- última hora: {lines[0]}")
    say("```")
    for line in lines[1:] + stratified(state["periods"], state["champion"]):
        say(line)
    say("```")
except Exception as e:
    say(f"- leaderboard no disponible: {e!r}")
say()
if "--quiet" not in sys.argv:
    REPORT.parent.mkdir(exist_ok=True)
    with REPORT.open("a") as f:
        f.write("\n".join(out) + "\n")
sys.exit(0 if healthy else 2)
