"""¿Aporta saber cómo le ha ido a cada producto EN NUESTRO carrusel (clics y compras cuando lo enseñamos)?

    python analysis/live_matrix.py --cf-fill     # primero
    python analysis/carousel_rate.py             # ~1 min, sin red

rate_cart mide lo que convence un producto en la tienda (carritos por vista). Aquí se prueba una señal
parecida pero medida en el propio carrusel: clics (o compras) que ha tenido el producto frente a los que
"tocaban" por los puestos en que se enseñó, suavizado. Para no hacer trampa, la señal de cada bloque de tiempo
se calcula SOLO con los otros cuatro bloques (validación cruzada por bloques, como analysis/pop_forms.py).
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
X, slot, buy, click, new, when, prod = d["X"].astype(np.float64), d["slot"], d["buy"], d["click"], d["new"].astype(bool), d["when"], d["prod"]
FEATS = list(d["features"]); FI = {n: i for i, n in enumerate(FEATS)}
n_prod = int(prod.max()) + 1
BASE = [n for n in FEATS if n not in ("cart_no_buy", "viewed_no_cart", "organic", "on_sale", "cart_live", "view_live", "pop_log", "rate_buy")]
onehot = np.eye(10)[slot][:, 1:]
block = np.minimum((np.argsort(np.argsort(when, kind="stable"), kind="stable") * 5 // len(when)), 4)


def carousel_feature(train, target, k):
    """log( (observado + k) / (esperado por puesto + k) ) por producto, con las filas `train`."""
    base = np.array([target[train & (slot == s)].mean() for s in range(10)])     # tasa media de cada puesto
    exp = np.bincount(prod[train], weights=base[slot[train]], minlength=n_prod)
    obs = np.bincount(prod[train], weights=target[train].astype(float), minlength=n_prod)
    return np.log((obs + k) / (exp + k))


def cv(mask, kind, C=0.3):
    ll, pr, yy = [], [], []
    for f in range(5):
        tr, te = mask & (block != f), mask & (block == f)
        cols = [X[:, FI[n]] for n in BASE]
        if kind:
            target, k = (click, 3.0) if kind == "clics" else (buy, 1.0)
            cols.append(carousel_feature(block != f, target, k)[prod])      # con TODOS los compradores de los otros bloques
        Z = np.stack(cols, 1)
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        lr = LogisticRegression(C=C, max_iter=3000).fit(np.hstack([(Z[tr] - mu) / sd, onehot[tr]]), buy[tr])
        pr.append(lr.predict_proba(np.hstack([(Z[te] - mu) / sd, onehot[te]]))[:, 1]); yy.append(buy[te])
        w = lr.coef_[0][len(BASE)] if kind else 0.0
    p, y = np.concatenate(pr), np.concatenate(yy)
    return log_loss(y, p), roc_auc_score(y, p), w


for label, seg in (("COMPRADORES NUEVOS", new), ("COMPRADORES CONOCIDOS", ~new)):
    print(f"\n## {label}: {int(seg.sum()) // 10} respuestas, {int(buy[seg].sum())} compras")
    print(f"   {'modelo':<52}{'log-loss':>10}{'mejora':>9}{'AUC':>8}{'peso':>8}")
    base_ll = None
    for name, kind in (("señales de v24 (con rate_cart)", None), ("+ clics del producto en nuestro carrusel", "clics"), ("+ compras del producto en nuestro carrusel", "compras")):
        ll, auc, w = cv(seg, kind)
        base_ll = base_ll or ll
        print(f"   {name:<52}{ll:>10.5f}{100 * (base_ll - ll) / base_ll:>8.2f}%{auc:>8.4f}{w:>8.2f}")
imp = np.bincount(prod, minlength=n_prod)
print(f"\nproductos enseñados alguna vez: {(imp > 0).sum()} | con 100+ impresiones: {(imp >= 100).sum()} | con 1.000+: {(imp >= 1000).sum()} | total {time.time() - t0:.0f}s")
