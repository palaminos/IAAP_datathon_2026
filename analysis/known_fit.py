"""¿Cómo conviene ajustar los pesos de los compradores CONOCIDOS con el feedback del evento?

    python analysis/known_fit.py      # usa cache/live_matrix.npz

Compara, por validación cruzada en 5 bloques de tiempo, las dos formas de entrenar de train_live.py:
  * "corregido": parte de los pesos del histórico y aprende correcciones (fit_corrected), con varias fuerzas C;
  * "solo evento": aprende los pesos solo con el feedback del evento (fit_plain, la opción --no-base).
Al principio de la noche había pocas compras de conocidos y ganaba "corregido"; con 1.300+ puede haber cambiado.
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

t0 = time.time()
d = np.load(rec.CACHE / "live_matrix.npz")
X, slot, buy, new, when = d["X"].astype(np.float64), d["slot"], d["buy"], d["new"].astype(bool), d["when"]
FEATS = list(d["features"]); FI = {n: i for i, n in enumerate(FEATS)}
assert FEATS == model.FEATURES
DROP = [FI[n] for n in ("cart_no_buy", "viewed_no_cart", "organic", "on_sale", "cart_live", "view_live", "pop_log", "rate_buy")]
X[:, DROP] = 0
seg = ~new if "--new" not in sys.argv else new
X, slot, buy, when = X[seg], slot[seg], buy[seg], when[seg]
onehot = np.eye(10)[slot][:, 1:]
block = np.minimum((np.argsort(np.argsort(when, kind="stable"), kind="stable") * 5 // len(when)), 4)
current = model.load_weights().astype(np.float64)
current[DROP] = 0
K = X.shape[1]
BIG = 20.0


def corrected(tr, te, C):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    s = X @ current
    m, v = s[tr].mean(), s[tr].std()
    z = lambda r: np.hstack([((s[r] - m) / v * BIG)[:, None], (X[r] - mu) / sd, onehot[r]])  # noqa: E731
    lr = LogisticRegression(C=C, max_iter=5000, tol=1e-6).fit(z(tr), buy[tr])
    w = lr.coef_[0][0] * BIG / v * current + lr.coef_[0][1:K + 1] / sd
    return lr.predict_proba(z(te))[:, 1], w


def plain(tr, te, C):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    z = lambda r: np.hstack([(X[r] - mu) / sd, onehot[r]])  # noqa: E731
    lr = LogisticRegression(C=C, max_iter=5000, tol=1e-6).fit(z(tr), buy[tr])
    return lr.predict_proba(z(te))[:, 1], lr.coef_[0][:K] / sd


print(f"{'compradores NUEVOS' if '--new' in sys.argv else 'compradores CONOCIDOS'}: {len(X) // 10} respuestas, {int(buy.sum())} compras")
print(f"{'ajuste':<28}{'log-loss':>10}{'AUC':>8}   por bloque")
allrows = np.ones(len(X), bool)
W = {}
for name, fn, Cs in (("corregido (base histórico)", corrected, (0.001, 0.01, 0.1, 1.0, 10.0)), ("solo evento", plain, (0.03, 0.1, 0.3, 1.0, 3.0))):
    for C in Cs:
        pr, yy = [], []
        for f in range(5):
            p, _ = fn(block != f, block == f, C)
            pr.append(p); yy.append(buy[block == f])
        p, y = np.concatenate(pr), np.concatenate(yy)
        print(f"{name + f' C={C}':<28}{log_loss(y, p):>10.5f}{roc_auc_score(y, p):>8.4f}   " + " ".join(f"{log_loss(b, a):.5f}" for a, b in zip(pr, yy)))
        W[(name, C)] = fn(allrows, allrows, C)[1]
show = ("p_cart", "p_view", "rate_cart", "cf", "cf_any", "sub_share", "is_new_prod", "hc_buy", "hc_click", "hc_any", "hc_n", "hs_views", "in_session", "is_last", "mf", "pop_glob", "pop_sub")
print("\npesos con todos los datos:")
print(f"{'':<28}" + "".join(f"{n[:9]:>10}" for n in show))
for (name, C), w in W.items():
    print(f"{name[:18] + f' C={C}':<28}" + "".join(f"{w[FI[n]]:>10.2f}" for n in show))
print(f"total {time.time() - t0:.0f}s")
