"""¿Hay productos "de moda" por horas? ¿Predice mejor la tasa RECIENTE de un producto que su tasa de siempre?

    python analysis/trend_rate.py     # una consulta agregada (producto × media hora) + cache/live_matrix.npz

rate_cart usa los 60 días del histórico y todo el evento: es estable. Si el simulador tiene picos (productos
que durante un rato convencen más), la tasa de la última hora debería predecir compras ADEMÁS de la de
siempre. Se mide con el feedback de hoy, validación cruzada en 5 bloques de tiempo. Para cada impresión se
usan solo los eventos de los 60 minutos ANTERIORES (por medias horas cerradas: sin trampa).
"""
import datetime as dt
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
rows = rec.query("""SELECT product_id::String AS p, toUnixTimestamp(toStartOfInterval(ts::DateTime64(3), INTERVAL 30 MINUTE)) AS h,
countIf(event_type::String = 'view') AS v, countIf(event_type::String = 'add_to_cart') AS a, countIf(event_type::String = 'purchase') AS b
FROM shop.events GROUP BY p, h""", timeout=300)
pi = {n: i for i, n in enumerate(names)}
h0 = min(int(r["h"]) for r in rows)
nb = (max(int(r["h"]) for r in rows) - h0) // 1800 + 1
V, A, B = np.zeros((len(names), nb)), np.zeros((len(names), nb)), np.zeros((len(names), nb))
for r in rows:
    i = pi.get(r["p"])
    if i is not None:
        k = (int(r["h"]) - h0) // 1800
        V[i, k], A[i, k], B[i, k] = float(r["v"]), float(r["a"]), float(r["b"])
print(f"{len(rows)} celdas producto × media hora, {nb} medias horas ({time.time() - t0:.0f}s)")
bucket = np.clip(((when / 1000 - h0) // 1800).astype(int), 0, nb - 1)        # media hora en curso de cada impresión


def recent(M, k_back):
    """Suma de las k_back medias horas CERRADAS anteriores a la de la impresión."""
    C = np.concatenate([np.zeros((len(names), 1)), np.cumsum(M, 1)], 1)
    hi = bucket; lo = np.maximum(bucket - k_back, 0)
    return C[prod, hi] - C[prod, lo]


rbar = A.sum() / V.sum()
P = {}
for name, kb in (("última hora", 2), ("últimas 3 horas", 6)):
    v, a = recent(V, kb), recent(A, kb)
    P[f"tasa carrito/vista, {name}"] = np.log((a + 10 * rbar) / (v + 10) / rbar)
    P[f"vistas, {name} (log)"] = np.log1p(v)
P["desvío última hora − siempre"] = P["tasa carrito/vista, última hora"] - X[:, FI["rate_cart"]]
BASE = [n for n in FEATS if n not in ("cart_no_buy", "viewed_no_cart", "organic", "on_sale", "cart_live", "view_live", "pop_log", "rate_buy")]
onehot = np.eye(10)[slot][:, 1:]
block = np.minimum((np.argsort(np.argsort(when, kind="stable"), kind="stable") * 5 // len(when)), 4)


def cv(mask, extra, C=0.3):
    Z = np.stack([X[:, FI[n]] for n in BASE] + [P[e] for e in extra], 1)
    pr, yy = [], []
    for f in range(5):
        tr, te = mask & (block != f), mask & (block == f)
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        lr = LogisticRegression(C=C, max_iter=3000).fit(np.hstack([(Z[tr] - mu) / sd, onehot[tr]]), buy[tr])
        pr.append(lr.predict_proba(np.hstack([(Z[te] - mu) / sd, onehot[te]]))[:, 1]); yy.append(buy[te])
    p, y = np.concatenate(pr), np.concatenate(yy)
    mu, sd = Z[mask].mean(0), Z[mask].std(0) + 1e-9
    lr = LogisticRegression(C=C, max_iter=3000).fit(np.hstack([(Z[mask] - mu) / sd, onehot[mask]]), buy[mask])
    return log_loss(y, p), roc_auc_score(y, p), lr.coef_[0][len(BASE):len(BASE) + len(extra)]


for label, seg in (("COMPRADORES NUEVOS", new), ("COMPRADORES CONOCIDOS", ~new)):
    print(f"\n## {label}: {int(seg.sum()) // 10} respuestas, {int(buy[seg].sum())} compras")
    print(f"   {'modelo':<50}{'log-loss':>10}{'mejora':>9}{'AUC':>8}   pesos estandarizados")
    ll0, auc0, _ = cv(seg, [])
    print(f"   {'señales de v24 (rate_cart de siempre)':<50}{ll0:>10.5f}{0.0:>8.2f}%{auc0:>8.4f}")
    for name, extra in (("+ tasa de la última hora", ["tasa carrito/vista, última hora"]), ("+ tasa de las últimas 3 horas", ["tasa carrito/vista, últimas 3 horas"]),
                        ("+ desvío (última hora − siempre)", ["desvío última hora − siempre"]), ("+ vistas de la última hora", ["vistas, última hora (log)"]),
                        ("+ tasa y vistas de la última hora", ["tasa carrito/vista, última hora", "vistas, última hora (log)"])):
        ll, auc, w = cv(seg, extra)
        print(f"   {name:<50}{ll:>10.5f}{100 * (ll0 - ll) / ll0:>8.2f}%{auc:>8.4f}   " + ", ".join(f"{x:+.2f}" for x in w))
# ¿cuánto cambia la tasa de un producto de una hora a otra? (productos con muchas vistas)
big = V.sum(1) > 3000
hr = (A[big][:, 0::2] + A[big][:, 1::2][:, :A[big][:, 0::2].shape[1]]) if False else None
ra = (A[big] + 5 * rbar) / (V[big] + 5)
print(f"\nproductos con más de 3.000 vistas hoy: {int(big.sum())} | variación de su tasa entre medias horas (desv. típica del log): {np.log(ra).std(1).mean():.3f} "
      f"| esperada solo por azar: {np.sqrt((1 / np.maximum(A[big], 1)).mean()):.3f}")
print(f"total {time.time() - t0:.0f}s")
