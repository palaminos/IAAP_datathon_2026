"""¿Qué enseña cada versión? Compara las listas de varias versiones sobre las mismas requests recientes.

    python analysis/compare_lists.py v8 v12 v13 v14      # la primera es la referencia. Solo lee, no envía nada.

Para cada versión: productos en común con la referencia, % de huecos con producto nuevo, con producto ya
visto/tocado por el comprador, con producto de la sesión, popularidad mediana (0 = el más vendido, 1 = el
que menos) y milisegundos por request. Sirve para detectar una versión que enseñe cosas raras ANTES de
desplegarla (p. ej. llenar la lista de productos nuevos o de productos que nadie compra).
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model  # noqa: E402
import recommender as rec  # noqa: E402

names = [a for a in sys.argv[1:] if a.startswith("v")] or ["v8", "v12", "v13"]
bg = rec.Background(rec.hist_popularity())
m = rec.load_models()
bg._load_users()
s, it = bg.state, bg.state.items
blocked = {p for p, last in rec.stock_changes().items() if last == "out_of_stock"}
reqs = rec.query("""SELECT request_id::String AS request_id, user_id::String AS user_id, recent_views, context, is_new_user FROM shop.requests
WHERE ts::DateTime64(3) > now() - INTERVAL 6 MINUTE LIMIT 1000""")
idx, FI = model.PRODUCTS.idx, model.FI
isnew = np.array([str(r.get("is_new_user")).lower() in ("true", "1") for r in reqs])
print(f"{len(reqs)} requests recientes ({100 * isnew.mean():.0f} % de compradores nuevos)\n")
feats = [model.featurize(r["user_id"], r.get("recent_views") or [], it, bg.users, m.coviews_idx) for r in reqs[:400]]
out = {}
for v in names:
    t = time.time()
    out[v] = [rec.respond(r, v + "-x", s, blocked, bg.users, m) for r in reqs]
    ms = (time.time() - t) * 1000 / len(reqs)
    fell = sum(a["model_version"] != v + "-x" for a in out[v])
    L = [[idx[p] for p in a["items"]] for a in out[v]]
    ref = [[idx[p] for p in a["items"]] for a in out[names[0]]]
    common = np.mean([len(set(a) & set(b)) for a, b in zip(L, ref)])
    first = 100 * np.mean([a[0] == b[0] for a, b in zip(L, ref)])
    new = np.array([[it.is_new[i] for i in a] for a in L])
    rank = np.median([[1 - it.static[i, 1] for i in a] for a in L])
    tch = 100 * np.mean([[(F[i, FI["p_view"]] > 0) or (F[i, FI["p_cart"]] > 0) for i in a] for a, F in zip(L, feats)])
    cart = 100 * np.mean([[F[i, FI["p_cart"]] > 0 for i in a] for a, F in zip(L, feats)])
    ses = 100 * np.mean([[F[i, FI["in_session"]] > 0 for i in a] for a, F in zip(L, feats)])
    cf0 = 100 * np.mean([[F[i, FI["cf"]] > 0 for i in a] for a, F in zip(L, feats)])
    print(f"{v:<6} en común con {names[0]}: {common:4.1f}/10 | mismo 1.º: {first:3.0f} % | nuevos: {100 * new.mean():4.1f} % (compr. nuevos {100 * new[isnew].mean():4.1f}, conocidos {100 * new[~isnew].mean():4.1f}) | "
          f"ya vistos: {tch:4.1f} % | en carrito: {cart:4.1f} % | de la sesión: {ses:4.1f} % | con cf: {cf0:3.0f} % | popularidad mediana: {rank:.3f} | {ms:5.2f} ms" + (f" | NO usaron la versión: {fell}" if fell else ""))
