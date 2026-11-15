"""¿Aporta la `propensity` de los carruseles antiguos (la probabilidad con que el sistema antiguo eligió cada producto)?

    python analysis/hist_propensity.py      # usa cache/live_matrix.npz (con urow); baja y guarda cache/hist_prop_users.npz

shop.history_sessions trae, para cada producto enseñado, la probabilidad con la que el sistema antiguo lo eligió.
En el histórico esa probabilidad predice la compra (puesto 1: <0,001 → 0,01 % de compra; >0,3 → 1,2 %): es la
"opinión" de otro modelo sobre cada pareja comprador-producto. Aquí se guarda, por pareja, la media y el máximo de
-log10(propensity) y se mide con el feedback del evento (validación cruzada en 5 bloques de tiempo).
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.simplefilter("ignore")
import model  # noqa: E402
import recommender as rec  # noqa: E402

HIST_PROP = """
SELECT u, groupArray(p) AS p, groupArray(v) AS v FROM (
    SELECT user_id::String AS u, pr.1 AS p,
           toUInt32(least(count(), 4095))
           + bitShiftLeft(toUInt32(round(least(greatest(-avg(log10(greatest(pr.2, 0.000001))), 0), 6) * 40)), 12)
           + bitShiftLeft(toUInt32(round(least(greatest(-log10(greatest(max(pr.2), 0.000001)), 0), 6) * 40)), 20) AS v
    FROM shop.history_sessions ARRAY JOIN arrayZip(shown::Array(String), propensity::Array(Float64)) AS pr
    WHERE cityHash64(user_id::String) % {chunks} = {i}
    GROUP BY u, p
) GROUP BY u"""

t0 = time.time()
d = np.load(rec.CACHE / "live_matrix.npz")
X, slot, buy, new, when, urow, prod, names = d["X"].astype(np.float64), d["slot"], d["buy"], d["new"].astype(bool), d["when"], d["urow"], d["prod"], d["names"]
FEATS = list(d["features"]); FI = {n: i for i, n in enumerate(FEATS)}
model.PRODUCTS.idx = {n: i for i, n in enumerate(names.tolist())}      # el catálogo con el que se hizo la tabla
model.PRODUCTS.names = names.tolist()
users = model.UserStore()
users.load_file(rec.CACHE / "users_history_v2.pkl")
n_hist = len(users.row)
indptr, idx, val = users._load_pairs(rec.query, HIST_PROP, rec.CACHE / "hist_prop_users.npz", 16)
print(f"propensity por pareja: {len(idx)} parejas ({time.time() - t0:.0f}s)")
rows_of = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
key = rows_of.astype(np.int64) * (1 << 20) + idx
order = np.argsort(key)
key, val = key[order], val[order]
ok = (urow >= 0) & (urow < n_hist)
q = urow.astype(np.int64) * (1 << 20) + prod
pos = np.minimum(np.searchsorted(key, q), len(key) - 1)
hit = ok & (key[pos] == q)
v = np.where(hit, val[pos], 0)
shown = (v & 0xFFF).astype(float)
lp_mean = np.where(hit, ((v >> 12) & 0xFF) / 40.0, 0.0)      # -log10(propensity), media
lp_max = np.where(hit, ((v >> 20) & 0xFF) / 40.0, 0.0)       # -log10 de la propensity máxima
print(f"filas con pareja vista en carruseles antiguos: {100 * hit[~new].mean():.0f} % (conocidos) | {100 * hit[new].mean():.1f} % (nuevos)")
top, kn = slot < 3, ~new
cart = (X[:, FI["p_cart"]] > 0) & (X[:, FI["p_buy"]] == 0)
print("\ntasa de compra (conocidos, puestos 1-3) según la propensity MÁXIMA con que el sistema antiguo enseñó ese producto a ese comprador:")
for label, m in (("todos", np.ones(len(X), bool)), ("del carrito", cart), ("no carrito", ~cart)):
    out = [f"nunca enseñado: {100 * buy[kn & top & m & ~hit].mean():.3f} % ({int(buy[kn & top & m & ~hit].sum())}/{int((kn & top & m & ~hit).sum())})"]
    for name, lo, hi in (("<0,01", 2.0, 9), ("0,01-0,03", 1.523, 2.0), ("0,03-0,1", 1.0, 1.523), ("0,1-0,3", 0.523, 1.0), (">0,3", -1, 0.523)):
        k = kn & top & m & hit & (lp_max > lo) & (lp_max <= hi)
        out.append(f"{name}: {100 * buy[k].mean() if k.sum() else 0:.3f} % ({int(buy[k].sum())}/{int(k.sum())})")
    print(f"   {label:<12}" + " | ".join(out))

P = {"enseñado": hit.astype(float), "log p media": np.where(hit, 6 - lp_mean, 0.0), "log p máx": np.where(hit, 6 - lp_max, 0.0),
     "p máx": np.where(hit, 10 ** -lp_max, 0.0), "p media": np.where(hit, 10 ** -lp_mean, 0.0),
     "esperado": np.where(hit, shown * 10 ** -lp_mean, 0.0)}
DROP = ("cart_no_buy", "viewed_no_cart", "organic", "on_sale", "cart_live", "view_live", "pop_log", "rate_buy")
BASE = [n for n in FEATS if n not in DROP]
onehot = np.eye(10)[slot][:, 1:]
block = np.minimum((np.argsort(np.argsort(when, kind="stable"), kind="stable") * 5 // len(when)), 4)


def cv(mask, extra, base=BASE, C=0.3):
    Z = np.stack([X[:, FI[n]] for n in base] + [P[e] for e in extra], 1)
    pr, yy = [], []
    for f in range(5):
        tr, te = mask & (block != f), mask & (block == f)
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        lr = LogisticRegression(C=C, max_iter=3000).fit(np.hstack([(Z[tr] - mu) / sd, onehot[tr]]), buy[tr])
        pr.append(lr.predict_proba(np.hstack([(Z[te] - mu) / sd, onehot[te]]))[:, 1]); yy.append(buy[te])
    p, y = np.concatenate(pr), np.concatenate(yy)
    mu, sd = Z[mask].mean(0), Z[mask].std(0) + 1e-9
    lr = LogisticRegression(C=C, max_iter=3000).fit(np.hstack([(Z[mask] - mu) / sd, onehot[mask]]), buy[mask])
    return log_loss(y, p), roc_auc_score(y, p), lr.coef_[0][len(base):len(base) + len(extra)] / sd[len(base):], [log_loss(b, a) for a, b in zip(pr, yy)]


seg = ~new
print(f"\n## COMPRADORES CONOCIDOS: {int(seg.sum()) // 10} respuestas, {int(buy[seg].sum())} compras")
ll0, auc0, _, pb0 = cv(seg, [])
print(f"   {'señales de v33':<50}{ll0:>10.5f}{0.0:>8.2f}%{auc0:>8.4f}")
NO_H = [n for n in BASE if not n.startswith(("hc_", "hs_"))]
for name, extra, base in (("+ log p máx (y enseñado)", ["enseñado", "log p máx"], BASE), ("+ log p media (y enseñado)", ["enseñado", "log p media"], BASE),
                          ("+ p máx", ["p máx"], BASE), ("+ p media", ["p media"], BASE), ("+ log p máx, log p media, enseñado", ["enseñado", "log p máx", "log p media"], BASE),
                          ("+ p máx y p media", ["p máx", "p media"], BASE), ("+ esperado (veces × p media)", ["esperado"], BASE),
                          ("v24 (sin hc_* ni hs_views)", [], NO_H), ("v24 + log p máx (y enseñado)", ["enseñado", "log p máx"], NO_H)):
    ll, auc, w, pb = cv(seg, extra, base)
    print(f"   {name:<50}{ll:>10.5f}{100 * (ll0 - ll) / ll0:>8.2f}%{auc:>8.4f}   {sum(a < b for a, b in zip(pb, pb0))}/5 | " + ", ".join(f"{x:+.2f}" for x in w))
print(f"\ntotal {time.time() - t0:.0f}s")
