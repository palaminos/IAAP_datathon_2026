"""¿Aporta saber cómo le fue a cada producto en los carruseles del HISTÓRICO (2 M de listas del sistema antiguo)?

    python analysis/hist_carousel.py     # usa cache/hist_carousel.json (una fila por producto y puesto) y cache/live_matrix.npz

Señal candidata: log( (compras + k) / (compras esperadas por sus puestos + k) ) con los 20 M de impresiones del
histórico; lo mismo con clics. Se añade a las señales de v24 (que ya tienen rate_cart) y se mide con el
feedback de hoy por validación cruzada en 5 bloques de tiempo, como analysis/pop_forms.py. Aquí no hay trampa
posible: la señal sale solo del histórico.
"""
import json
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
X, slot, buy, new, when, names, prod = d["X"].astype(np.float64), d["slot"], d["buy"], d["new"].astype(bool), d["when"], d["names"], d["prod"]
FEATS = list(d["features"]); FI = {n: i for i, n in enumerate(FEATS)}
rows = json.loads((rec.CACHE / "hist_carousel.json").read_text())
tot_n, tot_b, tot_c = np.zeros(11), np.zeros(11), np.zeros(11)
for p, k, n, b, c in rows:
    tot_n[k] += n; tot_b[k] += b; tot_c[k] += c
rb, rc = tot_b / np.maximum(tot_n, 1), tot_c / np.maximum(tot_n, 1)
agg = {}
for p, k, n, b, c in rows:
    a = agg.setdefault(p, [0.0, 0.0, 0.0, 0.0, 0.0])
    a[0] += n; a[1] += b; a[2] += n * rb[k]; a[3] += c; a[4] += n * rc[k]


def feat(kind, k):
    out = np.zeros(len(names))
    for i, p in enumerate(names):
        a = agg.get(p)
        if a:
            out[i] = np.log((a[1] + k) / (a[2] + k)) if kind == "b" else np.log((a[3] + k) / (a[4] + k))
    return out[prod]


P = {"hist. carrusel: compras (k=2)": feat("b", 2.0), "hist. carrusel: compras (k=10)": feat("b", 10.0), "hist. carrusel: clics (k=5)": feat("c", 5.0),
     "impresiones en el histórico (log)": np.array([np.log1p(agg.get(p, [0])[0]) for p in names])[prod]}
BASE = [n for n in FEATS if n not in ("cart_no_buy", "viewed_no_cart", "organic", "on_sale", "cart_live", "view_live", "pop_log", "rate_buy")]
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
    return log_loss(y, p), roc_auc_score(y, p), lr.coef_[0][len(base):len(base) + len(extra)], dict(zip(base, lr.coef_[0][:len(base)]))


NO_RATE = [n for n in BASE if n != "rate_cart"]
for label, seg in (("COMPRADORES NUEVOS", new), ("COMPRADORES CONOCIDOS", ~new)):
    print(f"\n## {label}: {int(seg.sum()) // 10} respuestas, {int(buy[seg].sum())} compras")
    print(f"   {'modelo':<58}{'log-loss':>10}{'mejora':>9}{'AUC':>8}   pesos estandarizados")
    ll0, auc0, _, co = cv(seg, [])
    print(f"   {'señales de v24 (con rate_cart)':<58}{ll0:>10.5f}{0.0:>8.2f}%{auc0:>8.4f}   rate_cart {co['rate_cart']:+.2f}")
    for name, extra, base in (("+ compras en el carrusel del histórico (k=2)", ["hist. carrusel: compras (k=2)"], BASE),
                              ("+ compras en el carrusel del histórico (k=10)", ["hist. carrusel: compras (k=10)"], BASE),
                              ("+ clics en el carrusel del histórico", ["hist. carrusel: clics (k=5)"], BASE),
                              ("+ compras y clics del histórico", ["hist. carrusel: compras (k=10)", "hist. carrusel: clics (k=5)"], BASE),
                              ("+ impresiones en el histórico", ["impresiones en el histórico (log)"], BASE),
                              ("SIN rate_cart, con compras del carrusel del histórico", ["hist. carrusel: compras (k=10)"], NO_RATE)):
        ll, auc, w, co = cv(seg, extra, base)
        print(f"   {name:<58}{ll:>10.5f}{100 * (ll0 - ll) / ll0:>8.2f}%{auc:>8.4f}   " + ", ".join(f"{x:+.2f}" for x in w) + (f" | rate_cart {co['rate_cart']:+.2f}" if "rate_cart" in co else ""))
print(f"\ntotal {time.time() - t0:.0f}s")
