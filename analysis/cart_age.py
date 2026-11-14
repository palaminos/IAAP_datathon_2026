"""Compradores conocidos: ¿qué distingue a un carrito que acaba en compra? Histórico del carrusel. Solo lee.

    python analysis/cart_age.py

El 72 % de las compras de los compradores conocidos sale de productos que tenían en el carrito sin comprar.
Aquí se mira, con los carruseles del 1-14 nov y los carritos anteriores al 1-nov, cómo cambia la compra de
esos productos según: cuántos días hace del último carrito, cuántas veces lo metió y cuántas veces lo vio.
Se da la tasa de compra y el multiplicador a igual puesto (regresión con el puesto como control).
"""
import datetime as dt
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.simplefilter("ignore")
from experiments import *  # noqa: F401,F403,E402

CARTS = """
SELECT user_id::String AS u, product_id::String AS p, count() AS n,
       dateDiff('hour', max(ts::DateTime64(3)), toDateTime64('{until}', 3)) / 24 AS days,
       dateDiff('hour', min(ts::DateTime64(3)), toDateTime64('{until}', 3)) / 24 AS first_days
FROM shop.history_events WHERE event_type::String = 'add_to_cart' AND ts::DateTime64(3) < '{until}'
  AND cityHash64(user_id::String) % 4 = {i} GROUP BY u, p"""
VIEWS_AFTER = """
SELECT user_id::String AS u, product_id::String AS p,
       dateDiff('hour', max(ts::DateTime64(3)), toDateTime64('{until}', 3)) / 24 AS days
FROM shop.history_events WHERE event_type::String = 'view' AND ts::DateTime64(3) < '{until}'
  AND cityHash64(user_id::String) % 4 = {i} GROUP BY u, p HAVING count() >= 1"""

t0 = time.time()
ctx = Ctx()
it, users = ctx.items, ctx.users
P, M = model.PRODUCTS.idx, model.MASK
carts = cached("cart_age", lambda: [r for i in range(4) for r in rec.query(CARTS.format(until=UNTIL, i=i), timeout=600)])
info = {(r["u"], r["p"]): (float(r["days"]), int(r["n"]), float(r["first_days"])) for r in carts}
print(f"{len(info)} pares comprador-producto con carrito antes del corte ({time.time() - t0:.0f}s)")
recs = []
for s in ctx.sessions:
    row = users.row.get(s["u"])
    if row is None:
        continue
    mine = users.prod.get(row, {})
    n_cart = sum(1 for pk in mine.values() if (pk >> 20) & M and not pk >> 40)
    w = 1.0 if s["clicked"] else 100 / NEG_PCT
    for k, p in enumerate(s["shown"]):
        pk = mine.get(P[p], 0)
        if (pk >> 20) & M and not pk >> 40:           # en el carrito y sin comprar
            d = info.get((s["u"], p))
            if d:
                recs.append((d[0] + 7 * s["week"] + 3.5, d[1], pk & M, p == s["purchased"], k, w, n_cart, it.static[P[p], 2], it.static[P[p], 1]))
R = np.array(recs, float)
days, ncart, nview, buy, slot, w, ncarts_user, lprice, pop = R.T
buy = buy > 0; slot = slot.astype(int)
print(f"{len(R)} impresiones de productos en el carrito sin comprar, {int(buy.sum())} compras; tasa {100 * (buy * w).sum() / w.sum():.3f} %\n")


def table(title, x, edges, labels):
    b = np.searchsorted(edges, x, side="right")
    Z = np.hstack([np.eye(10)[slot][:, 1:], np.eye(len(labels))[b][:, 1:]])
    lr = LogisticRegression(C=100, max_iter=3000).fit(Z, buy, sample_weight=w)
    odds = np.exp(np.concatenate([[0.0], lr.coef_[0][9:]]))
    print(f"## {title}")
    for g, lab in enumerate(labels):
        m = b == g
        if m.sum():
            print(f"   {lab:<22} impresiones={w[m].sum():>9.0f}  compra={100 * (buy[m] * w[m]).sum() / w[m].sum():6.3f} %  x{odds[g]:5.2f} a igual puesto  compras={int(buy[m].sum())}")
    print()


table("Días desde el último carrito", days, [7, 14, 21, 30, 45], ["< 7 días", "7-14", "14-21", "21-30", "30-45", "> 45"])
table("Veces que lo metió en el carrito", ncart, [2, 3, 4], ["1 vez", "2 veces", "3 veces", "4 o más"])
table("Veces que lo vio", nview, [1, 2, 4, 8], ["0", "1", "2-3", "4-7", "8 o más"])
table("Cuántos productos tiene en el carrito sin comprar", ncarts_user, [3, 6, 10, 20], ["1-2", "3-5", "6-9", "10-19", "20 o más"])
table("Precio (log)", lprice, list(np.quantile(lprice, [0.25, 0.5, 0.75])), ["barato", "medio-bajo", "medio-alto", "caro"])
table("Popularidad global (1 = el más vendido)", pop, [0.9, 0.97, 0.99], ["< 0,90", "0,90-0,97", "0,97-0,99", "top 1 %"])
print(f"total {time.time() - t0:.0f}s")
