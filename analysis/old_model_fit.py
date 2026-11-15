"""Ajuste de log(propensity) del sistema antiguo con nuestras señales, dentro de cada carrusel.

    python analysis/old_model_fit.py      # usa cache/old_model_sample.npz (analysis/old_model.py)
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.simplefilter("ignore")
import recommender as rec  # noqa: E402

t0 = time.time()
d = np.load(rec.CACHE / "old_model_sample.npz")
X, y, slot, sid, click, buy, known = d["X"].astype(np.float64), d["y"], d["slot"], d["sid"], d["click"], d["buy"], d["known"]
FEATS = list(d["features"]); FI = {n: i for i, n in enumerate(FEATS)}
n_s = sid.max() + 1
print(f"{n_s} carruseles | compradores con historial: {100 * known.mean():.0f} % | log(propensity): media {y.mean():.2f}, desviación {y.std():.2f}, dentro de un carrusel {np.sqrt(((y - np.bincount(sid, y)[sid] / 10) ** 2).mean()):.2f}")
print("propensity media por puesto:", " ".join(f"{np.exp(y[slot == k]).mean():.3f}" for k in range(10)))


def within(a):
    return a - (np.bincount(sid, a, minlength=n_s) / 10)[sid]


use = [n for n in FEATS if X[:, FI[n]].std() > 0]
Z = np.stack([within(X[:, FI[n]]) for n in use] + [within((slot == k).astype(float)) for k in range(1, 10)], 1)
yw = within(y)
for label, m in (("todos", np.ones(len(y), bool)), ("con historial", known), ("sin historial", ~known)):
    w, *_ = np.linalg.lstsq(Z[m], yw[m], rcond=None)
    r2 = 1 - ((yw[m] - Z[m] @ w) ** 2).sum() / (yw[m] ** 2).sum()
    print(f"\n{label}: R² dentro del carrusel = {r2:.3f} ({int(m.sum()) // 10} carruseles)")
    sd = X[m][:, [FI[n] for n in use]].std(0)
    order = np.argsort(-np.abs(w[:len(use)] * sd))
    print("   señal: peso (peso × desviación)  " + " | ".join(f"{use[i]} {w[i]:+.2f} ({w[i] * sd[i]:+.2f})" for i in order[:16]))
    print("   puestos 2..10: " + " ".join(f"{x:+.2f}" for x in w[len(use):]))
res = yw - Z @ np.linalg.lstsq(Z, yw, rcond=None)[0]
print(f"\nresiduo: desviación {res.std():.2f} | por tipo de fila:")
for name, m in (("en la sesión", X[:, FI["in_session"]] > 0), ("última vista", X[:, FI["is_last"]] > 0), ("en carrito", X[:, FI["p_cart"]] > 0), ("ya comprado", X[:, FI["p_buy"]] > 0),
                ("visto antes", (X[:, FI["p_view"]] > 0) & (X[:, FI["p_cart"]] == 0)), ("misma subcat. que la última vista", X[:, FI["same_sub"]] > 0), ("resto", (X[:, FI["in_session"]] == 0) & (X[:, FI["p_cart"]] == 0) & (X[:, FI["p_view"]] == 0))):
    print(f"   {name:<34} filas {int(m.sum()):>7} | log(propensity) medio dentro del carrusel {yw[m].mean():+.2f} | residuo medio {res[m].mean():+.2f} (desv. {res[m].std():.2f}) | compra {100 * buy[m].mean():.3f} %")
# ¿predice la compra la puntuación antigua mejor que nuestras señales? AUC dentro de la muestra
from sklearn.metrics import roc_auc_score  # noqa: E402
print(f"\nAUC de compra (todas las filas): log(propensity) {roc_auc_score(buy, y):.4f} | dentro del carrusel {roc_auc_score(buy, yw):.4f} | nuestra aproximación lineal {roc_auc_score(buy, Z[:, :len(use)] @ np.linalg.lstsq(Z, yw, rcond=None)[0][:len(use)]):.4f}")
print(f"total {time.time() - t0:.0f}s")
