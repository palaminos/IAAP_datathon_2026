"""Señal orgánica: un GBM entrenado con la actividad orgánica de la tienda (shop.events), no con el carrusel.

    python analysis/organic_test.py    # 1) reproduce el día y guarda los ejemplos en cache/exp_organic.npz (~3 min)
    python train_organic.py            # 2) entrena el GBM y guarda organic_gbm.pkl (va en el repo)

Por qué: nuestro feedback tiene pocos cientos de compras; la tienda genera por su cuenta más de 100 veces
más. Cada vista orgánica (usuario, producto) es un ejemplo: ¿lo metió en el carrito en los 30 min siguientes?
Con ~15.000 carritos sí hay datos para que un modelo de árboles aprenda combinaciones de señales (p. ej.
afinidad × popularidad) que con el feedback solo no se pueden aprender. Su salida se usa como UNA señal
más ("organic") del modelo lineal; cuánto pesa lo decide train_live.py con el feedback real.
Validado en analysis/organic_splits.py: +0,017 a +0,021 de AUC de compra y -3,4 % de log-loss en 4 cortes.
"""
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import model

TREES = 100   # ~6 ms por request en producción (con 200 árboles, ~25 ms y la misma mejora)

if __name__ == "__main__":
    t0 = time.time()
    d = np.load(Path("cache") / "exp_organic.npz")
    Xo, mso, yc = d["Xo"], d["mso"], d["yc"]
    names = model.FEATURES[:Xo.shape[1]]          # las señales que existían cuando se calcularon los ejemplos
    params = dict(max_iter=TREES, learning_rate=0.12, max_depth=4, min_samples_leaf=200, l2_regularization=1.0, random_state=0)
    cut = np.quantile(mso, 0.8)                    # comprobación: entrenar con el 80 % inicial, medir en el 20 % final
    check = HistGradientBoostingClassifier(**params).fit(Xo[mso < cut], yc[mso < cut])
    auc = roc_auc_score(yc[mso >= cut], check.decision_function(Xo[mso >= cut]))
    gbm = HistGradientBoostingClassifier(**params).fit(Xo, yc)
    with model.ORGANIC_FILE.open("wb") as f:
        pickle.dump({"model": gbm, "features": names, "views": int(len(Xo)), "carts": int(yc.sum()),
                     "auc_cart_last_20pct": float(auc), "trees": TREES}, f, protocol=4)
    F = np.zeros((5200, model.K), np.float32)
    t = time.time()
    for _ in range(10):
        gbm.decision_function(F[:, :len(names)])
    print(f"{len(Xo)} vistas orgánicas, {int(yc.sum())} con carrito | AUC de carrito en el 20 % final: {auc:.4f} | "
          f"{(time.time() - t) * 100:.1f} ms por request | guardado {model.ORGANIC_FILE.name} ({time.time() - t0:.0f}s)")
