"""¿Es sólida la mejora de añadir la señal orgánica (GBM entrenado con vistas orgánicas)? Repite la
comparación de analysis/organic_test.py con varios cortes temporales, usando los ejemplos ya calculados
(cache/exp_organic.npz). También mide cuánto tardaría el GBM en producción. No lee de RawTree.

    python analysis/organic_splits.py     # desde la raíz del repo, después de organic_test.py
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

warnings.simplefilter("ignore")
d = np.load(Path(__file__).resolve().parent.parent / "cache" / "exp_organic.npz")
Xo, mso, yc, Xf, slot, click, buy, when = (d[k] for k in ("Xo", "mso", "yc", "Xf", "slot", "click", "buy", "when"))
K = Xf.shape[1]
onehot = lambda s: np.eye(10, dtype=np.float32)[s][:, 1:]
GB = dict(max_iter=200, learning_rate=0.08, max_depth=4, min_samples_leaf=200, l2_regularization=1.0, random_state=0)


def judge(score, target, tr, te):
    s = np.clip(score, -60, 60)
    Z = np.hstack([((s - s[tr].mean()) / (s[tr].std() + 1e-9))[:, None], onehot(slot)])
    p = LogisticRegression(C=10.0, max_iter=2000).fit(Z[tr], target[tr]).predict_proba(Z[te])[:, 1]
    return roc_auc_score(target[te], p), log_loss(target[te], p)


def hit1(score, te):
    S, Y = score[te].reshape(-1, 10), buy[te].reshape(-1, 10)
    has = Y.any(1)
    sj = S[has][np.arange(has.sum()), Y[has].argmax(1)][:, None]
    return float(((S[has] > sj).sum(1) == 0).mean())


print(f"{len(Xo)} vistas orgánicas, {len(Xf) // 10} respuestas nuestras, {int(buy.sum())} compras\n")
print(f"{'corte':<8}{'compras prueba':>15} | {'solo feedback':^34} | {'feedback + señal orgánica':^34} | diferencia")
print(f"{'':<24}| {'AUC compra':>11}{'log-loss':>10}{'AUC clic':>9}{'h@1':>5} | {'AUC compra':>11}{'log-loss':>10}{'AUC clic':>9}{'h@1':>5} | {'AUC':>7}{'log-loss':>10}")
gbm = None
for q in (0.5, 0.6, 0.7, 0.8):
    cut = np.quantile(when, q)
    tr, te, otr = when < cut, when >= cut, mso < cut
    gbm = HistGradientBoostingClassifier(**GB).fit(Xo[otr], yc[otr])
    org = gbm.decision_function(Xf)
    mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-9
    Z0 = (Xf - mu) / sd
    Z1 = np.hstack([Z0, ((org - org[tr].mean()) / (org[tr].std() + 1e-9))[:, None]])
    out = []
    for Z in (Z0, Z1):
        lr = LogisticRegression(C=0.1, max_iter=3000).fit(np.hstack([Z[tr], onehot(slot[tr])]), buy[tr])
        s = Z @ lr.coef_[0][:Z.shape[1]]
        (ab, lb), (ac, _) = judge(s, buy, tr, te), judge(s, click, tr, te)
        out.append((ab, lb, ac, hit1(s, te)))
    a, b = out
    print(f"{int(100 * (1 - q)):>3} % final{int(buy[te].sum()):>10} | {a[0]:>11.4f}{a[1]:>10.5f}{a[2]:>9.4f}{a[3]:>5.2f} | {b[0]:>11.4f}{b[1]:>10.5f}{b[2]:>9.4f}{b[3]:>5.2f} | {b[0] - a[0]:>+7.4f}{100 * (b[1] / a[1] - 1):>+9.1f} %")

F = Xf[:5200]
t = time.time()
for _ in range(20):
    gbm.decision_function(F)
print(f"\ncoste del GBM en producción: {(time.time() - t) * 1000 / 20:.1f} ms por request (5.200 productos, {GB['max_iter']} árboles)")
small = HistGradientBoostingClassifier(**dict(GB, max_iter=80)).fit(Xo, yc)
t = time.time()
for _ in range(20):
    small.decision_function(F)
print(f"con 80 árboles: {(time.time() - t) * 1000 / 20:.1f} ms por request")
