"""¿Aporta separar la afinidad por subcategoría / marca según el tipo de evento (compras y carritos, no solo vistas)?

    python analysis/sub_buy.py      # usa cache/live_matrix.npz (con urow) y cache/users_history_v2.pkl

sub_share cuenta todos los eventos por igual y las vistas son el 86 %. Aquí, por comprador con historial: nº de
productos distintos comprados / metidos en el carrito en la subcategoría, la categoría y la marca del producto candidato.
"""
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.simplefilter("ignore")
import recommender as rec  # noqa: E402

t0 = time.time()
d = np.load(rec.CACHE / "live_matrix.npz")
X, slot, buy, new, when, urow, prod, names = d["X"].astype(np.float64), d["slot"], d["buy"], d["new"].astype(bool), d["when"], d["urow"], d["prod"], d["names"]
FEATS = list(d["features"]); FI = {n: i for i, n in enumerate(FEATS)}
cat = {r["p"]: (r["sub"], r["cat"], r["brand"]) for r in rec.query("""SELECT p, argMax(sub, v) AS sub, argMax(cat, v) AS cat, argMax(brand, v) AS brand FROM (
    SELECT product_id::String AS p, subcategory::String AS sub, category::String AS cat, brand::String AS brand, '' AS v FROM shop.catalog
    UNION ALL SELECT product_id::String, subcategory::String, category::String, brand::String, toString(ts::DateTime64(3)) FROM shop.catalog_updates) GROUP BY p""", timeout=120)}
with (rec.CACHE / "users_history_v2.pkl").open("rb") as f:
    h = pickle.load(f)
ids = [{}, {}, {}]


def code(p):
    c = cat.get(p, ("", "", ""))
    return [ids[k].setdefault(c[k], len(ids[k])) for k in range(3)]


hist_codes = np.array([code(p) for p in h["products"]])
now_codes = np.array([code(p) for p in names.tolist()])
n_users = len(h["row"])
MASK = (1 << 20) - 1
tabs = {(k, t): {} for k in range(3) for t in ("buy", "cart")}       # (nivel, tipo) -> {(fila, código): nº de productos}
for r, mine in h["prod"].items():
    for i, packed in mine.items():
        a, b = (packed >> 20) & MASK, packed >> 40
        t = "buy" if b else "cart" if a else None
        if t:
            for k in range(3):
                key = (r, int(hist_codes[i, k]))
                tabs[(k, t)][key] = tabs[(k, t)].get(key, 0) + 1
ok = (urow >= 0) & (urow < n_users)
P = {}
for (k, t), tab in tabs.items():
    keys = np.fromiter((a * 100000 + b for a, b in tab.keys()), np.int64, len(tab))
    vals = np.fromiter(tab.values(), np.float64, len(tab))
    o = np.argsort(keys); keys, vals = keys[o], vals[o]
    q = urow.astype(np.int64) * 100000 + now_codes[prod, k]
    pos = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
    P[f"{('sub', 'cat', 'marca')[k]}_{t}"] = np.where(ok & (keys[pos] == q), np.log1p(vals[pos]), 0.0)
print(f"tablas listas ({time.time() - t0:.0f}s)")
top, kn = slot < 3, ~new
untouched = (X[:, FI["p_cart"]] == 0) & (X[:, FI["p_view"]] == 0)
print("tasa de compra (conocidos, puestos 1-3, productos que NO había tocado) según productos COMPRADOS antes en esa subcategoría:")
c = np.rint(np.expm1(P["sub_buy"]))
print("   " + " | ".join(f"{name}: {100 * buy[kn & top & untouched & m].mean():.3f} % ({int(buy[kn & top & untouched & m].sum())}/{int((kn & top & untouched & m).sum())})"
                         for name, m in (("0", c == 0), ("1", c == 1), ("2-3", (c >= 2) & (c <= 3)), ("4+", c >= 4))))
c = np.rint(np.expm1(P["sub_cart"]))
print("lo mismo según productos en CARRITO (sin comprar) en esa subcategoría:")
print("   " + " | ".join(f"{name}: {100 * buy[kn & top & untouched & m].mean():.3f} % ({int(buy[kn & top & untouched & m].sum())}/{int((kn & top & untouched & m).sum())})"
                         for name, m in (("0", c == 0), ("1", c == 1), ("2-3", (c >= 2) & (c <= 3)), ("4+", c >= 4))))
DROP = ("cart_no_buy", "viewed_no_cart", "organic", "on_sale", "cart_live", "view_live", "pop_log", "rate_buy")
BASE = [n for n in FEATS if n not in DROP]
onehot = np.eye(10)[slot][:, 1:]
block = np.minimum((np.argsort(np.argsort(when, kind="stable"), kind="stable") * 5 // len(when)), 4)


def cv(mask, extra, C=0.3):
    Z = np.stack([X[:, FI[n]] for n in BASE] + [P[e] for e in extra], 1)
    pr, yy = [], []
    for f in range(5):
        tr, te = mask & (block != f), mask & (block == f)
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        lr = LogisticRegression(C=C, max_iter=3000).fit(np.hstack([(Z[tr] - mu) / sd, onehot[tr]]), buy[tr])
        pr.append(lr.predict_proba(np.hstack([(Z[te] - mu) / sd, onehot[te]]))[:, 1]); yy.append(buy[te])
    p, y = np.concatenate(pr), np.concatenate(yy)
    mu, sd = Z[mask].mean(0), Z[mask].std(0) + 1e-9
    lr = LogisticRegression(C=C, max_iter=3000).fit(np.hstack([(Z[mask] - mu) / sd, onehot[mask]]), buy[mask])
    return log_loss(y, p), roc_auc_score(y, p), lr.coef_[0][len(BASE):len(BASE) + len(extra)] / sd[len(BASE):], [log_loss(b, a) for a, b in zip(pr, yy)]


seg = ~new
print(f"\n## COMPRADORES CONOCIDOS: {int(seg.sum()) // 10} respuestas, {int(buy[seg].sum())} compras")
ll0, auc0, _, pb0 = cv(seg, [])
print(f"   {'señales de v33':<40}{ll0:>10.5f}{0.0:>8.2f}%{auc0:>8.4f}")
for name, extra in (("+ sub_buy", ["sub_buy"]), ("+ sub_cart", ["sub_cart"]), ("+ sub_buy, sub_cart", ["sub_buy", "sub_cart"]), ("+ cat_buy, cat_cart", ["cat_buy", "cat_cart"]),
                    ("+ marca_buy, marca_cart", ["marca_buy", "marca_cart"]), ("+ las seis", list(P))):
    ll, auc, w, pb = cv(seg, extra)
    print(f"   {name:<40}{ll:>10.5f}{100 * (ll0 - ll) / ll0:>8.2f}%{auc:>8.4f}   {sum(a < b for a, b in zip(pb, pb0))}/5 | " + ", ".join(f"{x:+.2f}" for x in w))
print(f"\ntotal {time.time() - t0:.0f}s")
