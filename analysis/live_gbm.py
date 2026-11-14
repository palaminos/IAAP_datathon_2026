"""¿Mejora un modelo de árboles (GBM) a la regresión logística con las señales actuales? Datos de hoy.

    python analysis/live_matrix.py [--cf-fill]   # primero
    python analysis/live_gbm.py                  # ~2 min

Split temporal 75/25. El GBM ve el puesto como una señal más al entrenar; para puntuar se le pone a todos
los productos el puesto 1 (así la puntuación es "del producto", como con los pesos). Medidas en la parte
de prueba: AUC y log-loss de compra sabiendo el puesto, y puesto medio del producto comprado entre los 10.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.simplefilter("ignore")
import recommender as rec  # noqa: E402

d = np.load(rec.CACHE / "live_matrix.npz")
X, slot, click, buy, when, new = d["X"].astype(np.float32), d["slot"], d["click"], d["buy"], d["when"], d["new"].astype(bool)
FEATS = list(d["features"])
DROP = ["cart_no_buy", "viewed_no_cart", "organic", "on_sale"]
cols = [i for i, n in enumerate(FEATS) if n not in DROP]
onehot = lambda s: np.eye(10)[s][:, 1:]
q = np.quantile(when, 0.75)
tr, te = when < q, when >= q
t0 = time.time()


def judge(score, fit_rows, test_rows, target=buy):
    Z = np.hstack([score[:, None], onehot(slot)])
    lr = LogisticRegression(C=10.0, max_iter=2000).fit(Z[fit_rows], target[fit_rows])
    p = lr.predict_proba(Z[test_rows])[:, 1]
    return roc_auc_score(target[test_rows], p), log_loss(target[test_rows], p)


def ranking(score, rows):
    S, Y = score.reshape(-1, 10), buy.reshape(-1, 10)
    R = rows.reshape(-1, 10)[:, 0] & Y.any(1)
    S, Y = S[R], Y[R]
    sj = S[np.arange(len(S)), Y.argmax(1)][:, None]
    return float(((S > sj).sum(1) + ((S == sj).sum(1) - 1) / 2 + 1).mean())


def lr_score(mask, C=1.0):
    Xm = X[mask][:, cols].astype(np.float64)
    mu, sd = Xm.mean(0), Xm.std(0) + 1e-9
    lr = LogisticRegression(C=C, max_iter=4000).fit(np.hstack([(Xm - mu) / sd, onehot(slot[mask])]), buy[mask])
    return ((X[:, cols] - mu) / sd) @ lr.coef_[0][:len(cols)]


def gbm_score(mask, target=buy, **kw):
    params = dict(max_iter=200, learning_rate=0.06, max_depth=4, min_samples_leaf=300, l2_regularization=1.0, random_state=0)
    params.update(kw)
    Xs = np.hstack([X[:, cols], slot[:, None].astype(np.float32)])
    g = HistGradientBoostingClassifier(**params).fit(Xs[mask], target[mask])
    Xs[:, -1] = 0   # todos en el puesto 1
    return g.decision_function(Xs)


scores = {"regresión logística (como v13)": lr_score(tr)}
print(f"LR lista ({time.time() - t0:.0f}s)", flush=True)
scores["GBM compra (prof. 4, 200 árboles)"] = gbm_score(tr)
scores["GBM compra (prof. 3, 300 árboles, hojas 600)"] = gbm_score(tr, max_depth=3, max_iter=300, min_samples_leaf=600)
scores["GBM compra (prof. 6, 150 árboles)"] = gbm_score(tr, max_depth=6, max_iter=150)
print(f"GBM listos ({time.time() - t0:.0f}s)", flush=True)
z = lambda s: (s - s[tr].mean()) / (s[tr].std() + 1e-9)
scores["media LR + GBM (prof. 4)"] = z(scores["regresión logística (como v13)"]) + z(scores["GBM compra (prof. 4, 200 árboles)"])
print(f"\n{'':<46}{'AUC':>8}{'log-loss':>10}{'puesto':>8} | {'nuevos: AUC':>12}{'puesto':>8} | {'conocidos: AUC':>15}{'puesto':>8}")
for name, s in scores.items():
    a, ll = judge(s, tr, te)
    an, _ = judge(s, tr & new, te & new)
    ak, _ = judge(s, tr & ~new, te & ~new)
    print(f"{name:<46}{a:>8.4f}{ll:>10.5f}{ranking(s, te):>8.2f} | {an:>12.4f}{ranking(s, te & new):>8.2f} | {ak:>15.4f}{ranking(s, te & ~new):>8.2f}")
print(f"\npuesto = puesto medio (1-10) del producto comprado entre los 10 mostrados; menos es mejor ({time.time() - t0:.0f}s)")
