"""¿Importa cuántos productos tiene el comprador en el carrito / vistos (dilución)?

    python analysis/user_totals.py      # usa cache/live_matrix.npz (con urow) y cache/users_history_v2.pkl

Si el comprador compra UNO de los productos de su carrito, cada uno es más probable cuanto menos tiene. Se mira la
tasa de compra en bruto y la mejora de la predicción al añadir señales normalizadas por comprador a las de v33.
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
X, slot, buy, new, when, urow = d["X"].astype(np.float64), d["slot"], d["buy"], d["new"].astype(bool), d["when"], d["urow"]
FEATS = list(d["features"]); FI = {n: i for i, n in enumerate(FEATS)}
with (rec.CACHE / "users_history_v2.pkl").open("rb") as f:
    h = pickle.load(f)
n_users = len(h["row"])
MASK = (1 << 20) - 1
ncart, nview, nbuy, cart_events = np.zeros(n_users + 1), np.zeros(n_users + 1), np.zeros(n_users + 1), np.zeros(n_users + 1)
for r, mine in h["prod"].items():
    for packed in mine.values():
        a, b = (packed >> 20) & MASK, packed >> 40
        if a and not b:
            ncart[r] += 1; cart_events[r] += a
        elif b:
            nbuy[r] += 1
        else:
            nview[r] += 1
u = np.where((urow >= 0) & (urow < n_users), urow, n_users)
nc, nv, nb = ncart[u], nview[u], nbuy[u]
cart = (X[:, FI["p_cart"]] > 0) & (X[:, FI["p_buy"]] == 0)
view = (X[:, FI["p_view"]] > 0) & ~cart & (X[:, FI["p_buy"]] == 0)
known = ~new & (u < n_users)
print(f"compradores con historial: {n_users} | mediana de productos en carrito sin comprar: {np.median(ncart[:n_users]):.0f} (p90 {np.quantile(ncart[:n_users], 0.9):.0f}) | vistos sin más: {np.median(nview[:n_users]):.0f} | comprados: {np.median(nbuy[:n_users]):.0f}")
top = slot < 3
print("\ntasa de compra de un producto del carrito (puestos 1-3) según cuántos tiene el comprador en el carrito (historial):")
for name, m in (("1-2", nc <= 2), ("3-5", (nc >= 3) & (nc <= 5)), ("6-10", (nc >= 6) & (nc <= 10)), ("11-20", (nc >= 11) & (nc <= 20)), ("21-40", (nc >= 21) & (nc <= 40)), ("41+", nc >= 41)):
    k = known & top & cart & m
    k2 = known & top & ~cart & m
    req = known & m
    print(f"   carrito {name:<6} filas {int(k.sum()):>6} | carrito: {100 * buy[k].mean() if k.sum() else 0:.2f} % ({int(buy[k].sum())}) | resto: {100 * buy[k2].mean() if k2.sum() else 0:.2f} % ({int(buy[k2].sum())}) | "
          f"respuestas {int(req.sum()) // 10:>6}, conversión {1000 * buy[req].mean():.2f} % | del carrito en la lista: {10 * cart[req].mean():.1f} de 10")
k0 = known & (nc == 0)
print(f"   sin carrito en el historial: respuestas {int(k0.sum()) // 10}, conversión {1000 * buy[k0].mean():.2f} %")

P = {"carrito: -log(nº en carrito)": np.where(cart, -np.log(np.maximum(nc, 1)), 0.0), "carrito: 1/nº": np.where(cart, 1 / np.maximum(nc, 1), 0.0),
     "log(nº en carrito) del comprador": np.log1p(nc), "visto: -log(nº vistos)": np.where(view, -np.log(np.maximum(nv, 1)), 0.0),
     "cuota del carrito": np.where(cart, np.expm1(X[:, FI["p_cart"]]) / np.maximum(cart_events[u], 1), 0.0),
     "no carrito × log(nº en carrito)": np.where(~cart, np.log1p(nc), 0.0)}
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
print(f"   {'señales de v33':<52}{ll0:>10.5f}{0.0:>8.2f}%{auc0:>8.4f}")
for name, extra in (("+ carrito: -log(nº en carrito)", ["carrito: -log(nº en carrito)"]), ("+ carrito: 1/nº", ["carrito: 1/nº"]),
                    ("+ log(nº en carrito) del comprador (todas las filas)", ["log(nº en carrito) del comprador"]),
                    ("+ cuota del carrito", ["cuota del carrito"]), ("+ visto: -log(nº vistos)", ["visto: -log(nº vistos)"]),
                    ("+ carrito -log(nº) y no carrito × log(nº)", ["carrito: -log(nº en carrito)", "no carrito × log(nº en carrito)"]),
                    ("+ carrito -log(nº), visto -log(nº)", ["carrito: -log(nº en carrito)", "visto: -log(nº vistos)"])):
    ll, auc, w, pb = cv(seg, extra)
    print(f"   {name:<52}{ll:>10.5f}{100 * (ll0 - ll) / ll0:>8.2f}%{auc:>8.4f}   {sum(a < b for a, b in zip(pb, pb0))}/5 | " + ", ".join(f"{x:+.2f}" for x in w))
print(f"\ntotal {time.time() - t0:.0f}s")
