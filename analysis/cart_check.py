"""Dos preguntas sobre el feedback real de hoy (solo lee):
  1. Carritos abandonados: ¿se compran más? ¿en qué puesto los ponemos? ¿cuántos dejamos fuera?
  2. Productos muy mirados sin carrito ni compra: ¿se compran?
Las señales de cada respuesta se calculan como eran en ese momento (se reproduce el día en orden).

    python analysis/cart_check.py     # desde la raíz del repo
"""
import datetime as dt
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model  # noqa: E402
import recommender as rec  # noqa: E402
from train_live import EVENTS, LAG_MS, LIVE  # noqa: E402

state = rec.build_state(rec.hist_popularity())
items = state.items
users = model.UserStore()
users.load_file(rec.CACHE / "users_history_v2.pkl")
reqs = sorted((r for i in range(4) for r in rec.query(LIVE.format(team=rec.TEAM, i=i), timeout=300)), key=lambda r: int(r["ms"]))
reqs = [r for r in reqs if len(r["shown"]) == 10 and all(p in model.PRODUCTS.idx for p in r["shown"])]
recent_from = int(reqs[-1]["ms"]) - 2 * 3600 * 1000   # últimas 2 h: v3/v4

hour = dt.datetime.fromisoformat(rec.query("SELECT toString(toStartOfHour(min(ts::DateTime64(3)))) AS d FROM shop.events")[0]["d"])
events, e = [], 0
tab = defaultdict(lambda: np.zeros(3))          # (grupo, valor) -> impresiones, clics, compras
cart_slots = np.zeros(10)
opp = defaultdict(int)
M = model.MASK
for r in reqs:
    limit = int(r["ms"]) - LAG_MS
    while True:
        while e < len(events) and int(events[e]["ms"]) < limit:
            ev = events[e]; users.add_event(ev["u"], ev["p"], ev["t"], items); e += 1
        if e < len(events) or hour.replace(tzinfo=dt.timezone.utc).timestamp() * 1000 > limit:
            break
        events, e = rec.query(EVENTS.format(a=hour, b=hour + dt.timedelta(hours=1)), timeout=300), 0
        hour += dt.timedelta(hours=1)
    row = users.row.get(r["u"])
    mine = users.prod.get(row, {}) if row is not None else {}
    recent = int(r["ms"]) >= recent_from
    for k, p in enumerate(r["shown"]):
        packed = mine.get(model.PRODUCTS.idx[p], 0)
        v, a, b = packed & M, (packed >> 20) & M, packed >> 40
        kind = "1 ya comprado" if b else "2 carrito sin comprar" if a else "3 visto 3+ veces, sin carrito" if v >= 3 else "4 visto 2 veces, sin carrito" if v == 2 else "5 visto 1 vez (hoy)" if v == 1 else "6 nunca tocado"
        out = (1, p == r["clicked"], p == r["purchased"])
        tab[("TODO EL DÍA", kind)] += out
        tab[(f"POR PUESTO: {'1' if k == 0 else '2-3' if k < 3 else '4-10'}", kind)] += out
        if recent and kind.startswith("2"):
            cart_slots[k] += 1
    if recent:
        carts = {i for i, pk in mine.items() if (pk >> 20) & M and not pk >> 40 and i < items.n and items.avail[i]}
        opp["requests"] += 1
        if carts:
            shown = {model.PRODUCTS.idx[p] for p in r["shown"]}
            opp["con algún carrito abandonado"] += 1
            opp["con alguno mostrado"] += bool(carts & shown)
            opp["con alguno en el puesto 1"] += model.PRODUCTS.idx[r["shown"][0]] in carts
            opp["carritos abandonados (total)"] += len(carts)
            opp["carritos abandonados mostrados"] += len(carts & shown)
    if r["purchased"]:
        users.add_event(r["u"], r["purchased"], "purchase", items)

last = None
for (group, kind), (n, c, b) in sorted(tab.items()):
    if group != last:
        print(f"\n## {group}"); last = group
    print(f"  {kind:<32} impresiones={int(n):>7}  clic={100 * c / n:6.3f}%  compra={100 * b / n:6.3f}%  (compras={int(b)})")
print("\n## ÚLTIMAS 2 H: ¿en qué puesto sale un carrito abandonado cuando lo mostramos?")
print("  " + "  ".join(f"{k + 1}:{int(x)}" for k, x in enumerate(cart_slots)))
print("\n## ÚLTIMAS 2 H: oportunidad")
for k, v in opp.items():
    print(f"  {k:<36}{v:>7}")
