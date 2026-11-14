"""¿Conviene un juego de pesos distinto para compradores nuevos y conocidos (o por página)?

    python analysis/live_matrix.py    # primero (2-4 min): tabla de señales del feedback de hoy
    python analysis/live_models.py    # ~1 min

Split temporal: entrenar con el primer 75 % del día, probar con el último 25 %. Para cada segmento se
compara, SOLO con las filas de prueba de ese segmento, el modelo único con el modelo propio del segmento.
Medida: AUC y log-loss de compra sabiendo el puesto (como train_live.judge).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.simplefilter("ignore")
import model  # noqa: E402
import recommender as rec  # noqa: E402

d = np.load(rec.CACHE / "live_matrix.npz")
X, slot, click, buy, when = d["X"].astype(np.float64), d["slot"], d["click"], d["buy"], d["when"]
new, page = d["new"], d["page"]
FEATS = list(d["features"])
FI = {n: i for i, n in enumerate(FEATS)}
DROP = [a for a in sys.argv[1:] if a in FI] or ["cart_no_buy", "viewed_no_cart", "organic", "on_sale"]
cols = [i for i, n in enumerate(FEATS) if n not in DROP]
onehot = lambda s: np.eye(10)[s][:, 1:]
tr, te = when < np.quantile(when, 0.75), when >= np.quantile(when, 0.75)


def fit(mask, C=1.0):
    """Pesos (uno por señal, en sus unidades) de una regresión logística con el puesto como control."""
    Xm = X[mask][:, cols]
    mu, sd = Xm.mean(0), Xm.std(0) + 1e-9
    lr = LogisticRegression(C=C, max_iter=4000).fit(np.hstack([(Xm - mu) / sd, onehot(slot[mask])]), buy[mask])
    w = np.zeros(len(FEATS))
    w[cols] = lr.coef_[0][:len(cols)] / sd
    return w


def judge(score, fit_rows, test_rows):
    Z = np.hstack([score[:, None], onehot(slot)])
    lr = LogisticRegression(C=10.0, max_iter=2000).fit(Z[fit_rows], buy[fit_rows])
    p = lr.predict_proba(Z[test_rows])[:, 1]
    return roc_auc_score(buy[test_rows], p), log_loss(buy[test_rows], p)


def ranking(score, rows):
    """En las respuestas con compra: puesto medio que la puntuación da al producto comprado entre los 10 (1 = mejor)."""
    S, Y = score.reshape(-1, 10), buy.reshape(-1, 10)
    R = rows.reshape(-1, 10)[:, 0] & Y.any(1)
    S, Y = S[R], Y[R]
    sj = S[np.arange(len(S)), Y.argmax(1)][:, None]
    return float(((S > sj).sum(1) + 1).mean()), int(R.sum())


w_all = {C: fit(tr, C) for C in (0.1, 1.0)}
segs = {"compradores NUEVOS": new, "compradores CONOCIDOS": ~new,
        "página home": page == "home", "página category": page == "category", "página product": page == "product",
        "nuevos · product": new & (page == "product"), "conocidos · product": ~new & (page == "product")}
print(f"{'segmento':<24}{'compras tr/te':>14} | {'modelo único':^34} | {'modelo propio del segmento':^34}")
print(f"{'':<24}{'':>14} | {'AUC':>8}{'log-loss':>10}{'puesto':>8}{'':>8} | {'AUC':>8}{'log-loss':>10}{'puesto':>8}{'C':>6}")
own = {}
for name, m in segs.items():
    a, ll = judge(X @ w_all[1.0], tr & m, te & m)
    rk, n = ranking(X @ w_all[1.0], te & m)
    best = None
    for C in (0.03, 0.1, 1.0):
        w = fit(tr & m, C)
        r = judge(X @ w, tr & m, te & m) + (ranking(X @ w, te & m)[0], C, w)
        if best is None or r[1] < best[1]:
            best = r
    own[name] = best[4]
    print(f"{name:<24}{int(buy[tr & m].sum()):>7}/{int(buy[te & m].sum()):<6} | {a:>8.4f}{ll:>10.5f}{rk:>8.2f}{'':>8} | {best[0]:>8.4f}{best[1]:>10.5f}{best[2]:>8.2f}{best[3]:>6}")
print("\npuesto = puesto medio (1-10) que el modelo da al producto comprado entre los 10 mostrados; menos es mejor")
print("\nPesos (por desviación típica de la señal): único | nuevos | conocidos")
sd = X.std(0)
for i in np.argsort(-np.abs(w_all[1.0] * sd)):
    if FEATS[i] not in DROP:
        print(f"  {FEATS[i]:<16}{w_all[1.0][i] * sd[i]:>8.3f}{own['compradores NUEVOS'][i] * sd[i]:>9.3f}{own['compradores CONOCIDOS'][i] * sd[i]:>9.3f}")
