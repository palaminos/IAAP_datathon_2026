"""¿Salen mejores pesos de COMPRA si se parte de los pesos aprendidos con CLICS?

    python analysis/click_base.py [--new]     # usa cache/live_matrix.npz

Hay 3,5 veces más clics que compras, toda compra pasa por un clic y los pesos de clic y de compra se parecen mucho
(correlación 0,98). Idea: aprender primero los pesos con los clics (más datos, menos ruido) y después los de compra
como "pesos de clic × escala + correcciones" (la misma fórmula que ya usa train_live.py con los pesos del histórico).
El resultado sigue siendo UN vector de pesos por segmento: producción no cambia. Validación cruzada, 5 bloques de tiempo.
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
import recommender as rec  # noqa: E402

t0 = time.time()
d = np.load(rec.CACHE / "live_matrix.npz")
X, slot, buy, click, new, when = d["X"].astype(np.float64), d["slot"], d["buy"], d["click"], d["new"].astype(bool), d["when"]
FEATS = list(d["features"]); FI = {n: i for i, n in enumerate(FEATS)}
DROP = ("cart_no_buy", "viewed_no_cart", "organic", "on_sale", "cart_live", "view_live", "pop_log", "rate_buy")
if "--new" in sys.argv:
    DROP += ("hc_buy", "hc_click", "hc_any", "hc_n", "hs_views")
seg = new if "--new" in sys.argv else ~new
Z = X[seg][:, [FI[n] for n in FEATS if n not in DROP]]
slot, buy, click, when = slot[seg], buy[seg], click[seg], when[seg]
onehot = np.eye(10)[slot][:, 1:]
block = np.minimum((np.argsort(np.argsort(when, kind="stable"), kind="stable") * 5 // len(when)), 4)
K = Z.shape[1]
BIG = 20.0


def plain(tr, te, y, C):
    mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
    lr = LogisticRegression(C=C, max_iter=5000, tol=1e-6).fit(np.hstack([(Z[tr] - mu) / sd, onehot[tr]]), y[tr])
    return lr.predict_proba(np.hstack([(Z[te] - mu) / sd, onehot[te]]))[:, 1], lr.coef_[0][:K] / sd


def corrected(tr, te, base, C):
    mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
    s = Z @ base
    m, v = s[tr].mean(), s[tr].std()
    z = lambda r: np.hstack([((s[r] - m) / v * BIG)[:, None], (Z[r] - mu) / sd, onehot[r]])  # noqa: E731
    lr = LogisticRegression(C=C, max_iter=5000, tol=1e-6).fit(z(tr), buy[tr])
    return lr.predict_proba(z(te))[:, 1], lr.coef_[0][0] * BIG / v * base + lr.coef_[0][1:K + 1] / sd


print(f"{'compradores NUEVOS' if '--new' in sys.argv else 'compradores CONOCIDOS'}: {len(Z) // 10} respuestas, {int(buy.sum())} compras, {int(click.sum())} clics")
res = {"directo (producción) C=0.3": [[], []]}
Cs = (0.001, 0.003, 0.01, 0.03, 0.1)
for C in Cs:
    res[f"base clics, C={C}"] = [[], []]
for f in range(5):
    tr, te = block != f, block == f
    p, _ = plain(tr, te, buy, 0.3)
    res["directo (producción) C=0.3"][0].append(p); res["directo (producción) C=0.3"][1].append(buy[te])
    _, wc = plain(tr, tr, click, 0.3)
    for C in Cs:
        p, _ = corrected(tr, te, wc, C)
        res[f"base clics, C={C}"][0].append(p); res[f"base clics, C={C}"][1].append(buy[te])
ll0 = None
for name, (pr, yy) in res.items():
    p, y = np.concatenate(pr), np.concatenate(yy)
    ll = log_loss(y, p)
    ll0 = ll0 or ll
    print(f"   {name:<28} log-loss {ll:.5f} ({100 * (ll0 - ll) / ll0:+.2f} %)  AUC {roc_auc_score(y, p):.4f} | " + " ".join(f"{log_loss(b, a):.5f}" for a, b in zip(pr, yy)))
print(f"total {time.time() - t0:.0f}s")
