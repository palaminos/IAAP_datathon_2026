"""¿Tiene cada comprador un nivel de precio habitual? (señal: distancia entre el precio del producto y el suyo)

    python analysis/price_fit.py      # usa cache/live_matrix.npz (con urow) y cache/users_history_v2.pkl

Por comprador con historial: media y dispersión del log-precio de lo que metió en el carrito o compró (y de lo que vio).
Señal candidata para cada producto: |log precio - media del comprador| (en desviaciones). Validación cruzada en 5 bloques.
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
price = {r["p"]: float(r["price"]) for r in rec.query("""SELECT p, argMax(price, v) AS price FROM (
    SELECT product_id::String AS p, price_eur::Float64 AS price, '' AS v FROM shop.catalog
    UNION ALL SELECT product_id::String, price_eur::Float64, toString(ts::DateTime64(3)) FROM shop.catalog_updates) GROUP BY p""", timeout=120)}
with (rec.CACHE / "users_history_v2.pkl").open("rb") as f:
    h = pickle.load(f)
lp_hist = np.array([np.log(max(price.get(p, 0.0), 0.5)) for p in h["products"]])
lp_now = np.array([np.log(max(price.get(p, 0.0), 0.5)) for p in names.tolist()])
n_users = len(h["row"])
MASK = (1 << 20) - 1
mu_c, sd_c, n_c, mu_v, n_v = np.zeros(n_users + 1), np.ones(n_users + 1), np.zeros(n_users + 1), np.zeros(n_users + 1), np.zeros(n_users + 1)
for r, mine in h["prod"].items():
    ii = np.fromiter(mine.keys(), np.int64, len(mine))
    pk = np.fromiter(mine.values(), np.int64, len(mine))
    strong = ((pk >> 20) & MASK) + (pk >> 40) > 0
    if strong.sum() >= 2:
        x = lp_hist[ii[strong]]
        mu_c[r], sd_c[r], n_c[r] = x.mean(), max(x.std(), 0.15), strong.sum()
    if len(ii) >= 2:
        mu_v[r], n_v[r] = lp_hist[ii].mean(), len(ii)
u = np.where((urow >= 0) & (urow < n_users), urow, n_users)
lp = lp_now[prod]
has = n_c[u] >= 2
dist = np.where(has, np.abs(lp - mu_c[u]), 0.0)
distz = np.where(has, np.minimum(np.abs(lp - mu_c[u]) / sd_c[u], 6), 0.0)
signed = np.where(has, lp - mu_c[u], 0.0)
print(f"log-precio global: media {lp_now.mean():.2f}, desviación {lp_now.std():.2f} | desviación típica dentro de un comprador: mediana {np.median(sd_c[:n_users][n_c[:n_users] >= 2]):.2f}")
top, kn = slot < 3, ~new
touched = (X[:, FI["p_cart"]] > 0) | (X[:, FI["p_view"]] > 0)
print("tasa de compra (conocidos, puestos 1-3, productos que NO había tocado) según la distancia al precio habitual del comprador (en log):")
out = []
for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 9)):
    k = kn & top & has & ~touched & (dist >= lo) & (dist < hi)
    out.append(f"{lo}-{hi}: {100 * buy[k].mean():.3f} % ({int(buy[k].sum())}/{int(k.sum())})")
print("   " + " | ".join(out))
P = {"dist": dist, "distz": distz, "más caro": np.maximum(signed, 0), "más barato": np.maximum(-signed, 0), "tiene": has.astype(float), "dist2": dist ** 2}
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
print(f"   {'señales de v33':<44}{ll0:>10.5f}{0.0:>8.2f}%{auc0:>8.4f}")
for name, extra in (("+ distancia al precio habitual", ["dist"]), ("+ distancia en desviaciones", ["distz"]), ("+ más caro / más barato", ["más caro", "más barato"]),
                    ("+ distancia y distancia²", ["dist", "dist2"])):
    ll, auc, w, pb = cv(seg, extra)
    print(f"   {name:<44}{ll:>10.5f}{100 * (ll0 - ll) / ll0:>8.2f}%{auc:>8.4f}   {sum(a < b for a, b in zip(pb, pb0))}/5 | " + ", ".join(f"{x:+.2f}" for x in w))
print(f"\ntotal {time.time() - t0:.0f}s")
