"""¿Sirve el mismo "parecido entre productos" (train_cf.py) aplicado a lo que el comprador ha VISTO?

    python analysis/cf_views_lift.py     # desde la raíz del repo. Solo lee (usa la caché de cf_lift.py).

Por qué: cf solo mira carritos y compras, así que no dice nada de un comprador nuevo. De él solo sabemos
lo que acaba de ver (recent_views). Se mide en el histórico del carrusel (1-14 nov, similitud calculada
con datos anteriores al 1-nov), para cada producto mostrado:
  cf_sess = suma de su parecido con los productos de recent_views (la sesión)
  cf_view = suma de su parecido con lo que el comprador había visto antes sin meterlo en el carrito
y la tasa de compra según esas puntuaciones, también cuando las señales que ya tenemos valen cero.
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import *  # noqa: F401,F403,E402

t0 = time.time()
ctx = Ctx()
it, users = ctx.items, ctx.users
pairs = cached("cf_pairs", lambda: None)
cnt = {r["p"]: float(r["n"]) for r in cached("cf_counts", lambda: None)}
P = model.PRODUCTS.idx
rows, cols, vals = [], [], []
for r in pairs:
    i, j = P.get(r["p"]), P.get(r["q"])
    if i is not None and j is not None:
        rows.append(i), cols.append(j), vals.append(float(r["n"]) / np.sqrt(cnt[r["p"]] * cnt[r["q"]]))
S = csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(it.n, it.n))
M = model.MASK
recs = []
for s in ctx.sessions:
    row = users.row.get(s["u"])
    mine = users.prod.get(row, {}) if row is not None else {}
    liked = [i for i, pk in mine.items() if (pk >> 20) & M or pk >> 40]
    viewed = [i for i, pk in mine.items() if not ((pk >> 20) & M or pk >> 40)]
    rv = [P[p] for p in s["rv"] if p in P]
    idx = [P[p] for p in s["shown"]]
    z = np.zeros(10)
    cl = np.asarray(S[liked].sum(0)).ravel()[idx] if liked else z
    cv = np.asarray(S[viewed].sum(0)).ravel()[idx] if viewed else z
    cs = np.asarray(S[rv].sum(0)).ravel()[idx] if rv else z
    last_sub = it.sub[rv[-1]] if rv else -1
    w = 1.0 if s["clicked"] else 100 / NEG_PCT
    for k, p in enumerate(s["shown"]):
        recs.append((cl[k], cv[k], cs[k], p == s["purchased"], idx[k] in mine, idx[k] in rv, it.sub[idx[k]] == last_sub, k, w,
                     bool(liked), row is None or not mine, bool(rv)))
R = np.array(recs, dtype=float)
cl, cv, cs, buy, touched, insess, samesub, slot, w, has_liked, is_new, has_rv = R.T
print(f"{len(R)} impresiones, {int(buy.sum())} compras ({time.time() - t0:.0f}s)\n")


def table(title, x, mask):
    pos = x[mask & (x > 0)]
    if len(pos) < 100:
        print(f"## {title}: sin datos\n"); return
    qs = np.quantile(pos, [0.5, 0.8, 0.95])
    b = np.where(x <= 0, 0, np.where(x < qs[0], 1, np.where(x < qs[1], 2, np.where(x < qs[2], 3, 4))))
    base = (buy[mask] * w[mask]).sum() / w[mask].sum()
    print(f"## {title}  (media {100 * base:.3f} %)")
    for g, name in enumerate(["0 sin parecido", "1 bajo", "2 medio", "3 alto (top 20 %)", "4 muy alto (top 5 %)"]):
        m = mask & (b == g)
        if w[m].sum() > 0:
            rate = (buy[m] * w[m]).sum() / w[m].sum()
            print(f"   {name:<22} impresiones={w[m].sum():>10.0f}  compra={100 * rate:6.3f} %  ({rate / base:5.1f}x)  compras={int(buy[m].sum())}")
    print()


fresh = (touched == 0) & (insess == 0)      # productos que no ha tocado nunca ni están en la sesión
table("SESIÓN: todas las impresiones con recent_views", cs, has_rv > 0)
table("SESIÓN: comprador sin historial (nuevo), producto no visto en la sesión", cs, (is_new > 0) & (has_rv > 0) & (insess == 0))
table("SESIÓN: nuevo, no visto, y de OTRA subcategoría que la última vista", cs, (is_new > 0) & (has_rv > 0) & (insess == 0) & (samesub == 0))
table("SESIÓN: nuevo, no visto, MISMA subcategoría que la última vista", cs, (is_new > 0) & (has_rv > 0) & (insess == 0) & (samesub == 1))
table("SESIÓN: con historial, producto nunca tocado y SIN parecido con sus carritos/compras", cs, (has_liked > 0) & fresh & (cl <= 0) & (has_rv > 0))
table("SESIÓN: con historial, nunca tocado, CON parecido con carritos/compras (¿añade algo?)", cs, (has_liked > 0) & fresh & (cl > 0) & (has_rv > 0))
table("VISTAS ANTIGUAS: comprador con vistas pero sin carritos/compras, producto nunca tocado", cv, (has_liked == 0) & (is_new == 0) & fresh)
table("VISTAS ANTIGUAS: con carritos/compras, nunca tocado y SIN parecido con ellos", cv, (has_liked > 0) & fresh & (cl <= 0))
print(f"cuota de impresiones: comprador nuevo {100 * w[is_new > 0].sum() / w.sum():.1f} % | con recent_views {100 * w[has_rv > 0].sum() / w.sum():.1f} % | total {time.time() - t0:.0f}s")
