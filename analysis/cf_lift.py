"""¿Predice la compra un filtrado colaborativo de producto a producto ("quien compró X también compró Y")?

    python analysis/cf_lift.py     # desde la raíz del repo. Solo lee.

Hipótesis: nuestro punto débil son los usuarios con historial (en el leaderboard, el recomendador de
referencia de los organizadores saca ~2,1 % con ellos y nosotros ~1,3 %). Tenemos sus gustos por
subcategoría, categoría y marca, pero nada que diga QUÉ PRODUCTOS concretos compra la gente con un
historial parecido. Aquí se mide, en el histórico del carrusel y sin hacer trampa con el tiempo:
  1. Similitud entre productos con los carritos y compras ANTERIORES al 1-nov (coseno de co-ocurrencia).
  2. Para cada producto mostrado en un carrusel del 1 al 14-nov: suma de su similitud con lo que ese
     usuario había metido en el carrito o comprado antes.
  3. Tasa de compra según esa puntuación, también dentro de productos que el usuario nunca había tocado.
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import *  # noqa: F401,F403,E402

PAIRS = """
SELECT a.p AS p, b.p AS q, count() AS n
FROM (SELECT DISTINCT user_id::String AS u, product_id::String AS p FROM shop.history_events
      WHERE event_type::String IN ('purchase', 'add_to_cart') AND ts::DateTime64(3) < '{until}') AS a
INNER JOIN (SELECT DISTINCT user_id::String AS u, product_id::String AS p FROM shop.history_events
            WHERE event_type::String IN ('purchase', 'add_to_cart') AND ts::DateTime64(3) < '{until}') AS b ON a.u = b.u
WHERE a.p != b.p GROUP BY p, q HAVING n >= 2 ORDER BY n DESC LIMIT 100 BY p"""
COUNTS = """SELECT product_id::String AS p, uniqExact(user_id::String) AS n FROM shop.history_events
WHERE event_type::String IN ('purchase', 'add_to_cart') AND ts::DateTime64(3) < '{until}' GROUP BY p"""

t0 = time.time()
ctx = Ctx()
it, users = ctx.items, ctx.users
pairs = cached("cf_pairs", lambda: rec.query(PAIRS.format(until=UNTIL), timeout=900))
cnt = {r["p"]: float(r["n"]) for r in cached("cf_counts", lambda: rec.query(COUNTS.format(until=UNTIL), timeout=600))}
P = model.PRODUCTS.idx
rows, cols, vals = [], [], []
for r in pairs:
    i, j = P.get(r["p"]), P.get(r["q"])
    if i is not None and j is not None:
        rows.append(i), cols.append(j), vals.append(float(r["n"]) / np.sqrt(cnt[r["p"]] * cnt[r["q"]]))
S = csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(it.n, it.n))
print(f"{len(vals)} pares de productos con similitud ({time.time() - t0:.0f}s)", flush=True)

M = model.MASK
tab = defaultdict(lambda: np.zeros(2))
recs = []   # (cf, comprado, tocado antes, cuota de su subcategoría, puesto)
for s in ctx.sessions:
    row = users.row.get(s["u"])
    mine = users.prod.get(row, {}) if row is not None else {}
    liked = [i for i, pk in mine.items() if (pk >> 20) & M or pk >> 40]     # carrito o compra antes del corte
    idx = [P[p] for p in s["shown"]]
    cf = np.asarray(S[liked].sum(0)).ravel()[idx] if liked else np.zeros(10)
    sc = users.sub[row].astype(np.float32) if row is not None else None
    tot = float(sc.sum()) if sc is not None else 0.0
    w = 1.0 if s["clicked"] else 100 / NEG_PCT       # los carruseles sin clic están muestreados
    for k, p in enumerate(s["shown"]):
        share = float(sc[it.sub[idx[k]]]) / tot if tot else 0.0
        recs.append((cf[k], p == s["purchased"], idx[k] in mine, share, k, w, bool(liked)))
R = np.array(recs, dtype=float)
cf, buy, touched, share, slot, w, has = R.T
print(f"{len(R)} impresiones (ponderadas: {w.sum() / 1e6:.2f} M), {int(buy.sum())} compras ({time.time() - t0:.0f}s)\n")
pos = cf[(cf > 0)]
qs = np.quantile(pos, [0.5, 0.8, 0.95])


def bucket(x):
    return np.where(x <= 0, 0, np.where(x < qs[0], 1, np.where(x < qs[1], 2, np.where(x < qs[2], 3, 4))))


names = ["0 sin similitud", "1 baja", "2 media", "3 alta (top 20 %)", "4 muy alta (top 5 %)"]
b = bucket(cf)


def table(title, mask):
    print(f"## {title}")
    base = (buy[mask] * w[mask]).sum() / w[mask].sum()
    for g in range(5):
        m = mask & (b == g)
        if w[m].sum() > 0:
            rate = (buy[m] * w[m]).sum() / w[m].sum()
            print(f"   {names[g]:<24} impresiones={w[m].sum():>10.0f}  compra={100 * rate:6.3f} %  ({rate / base:5.1f}x la media)  compras={int(buy[m].sum())}")
    print()


known = has > 0
table("usuarios con algún carrito/compra previo: todas las impresiones", known)
table("...solo productos que el usuario NUNCA había tocado", known & (touched == 0))
table("...nunca tocados y de una subcategoría que SÍ le gusta (cuota >= 10 %)", known & (touched == 0) & (share >= 0.10))
table("...nunca tocados y de una subcategoría que le gusta poco (cuota < 3 %)", known & (touched == 0) & (share < 0.03))
table("...nunca tocados, puesto 1", known & (touched == 0) & (slot == 0))
table("...nunca tocados, puestos 4-10", known & (touched == 0) & (slot >= 3))
