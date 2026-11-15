"""¿Cómo se mide mejor la popularidad de un producto para ordenar el carrusel? Datos de hoy. Solo lee.

    python analysis/live_matrix.py --cf-fill     # primero (guarda también los nombres de producto)
    python analysis/pop_forms.py

Hoy usamos el VOLUMEN (3·compras + carritos + 0,05·vistas, histórico de 14 días + última hora). Pero en un
carrusel todos los productos "se ven" igual: quizá importa más la TASA (compras por cada vista) que el
volumen. Aquí se compara, con validación cruzada por bloques de tiempo (5 bloques), cuánto mejora la
predicción de compra al añadir cada forma de popularidad a las señales de v14, para compradores nuevos y
conocidos. Dos consultas pequeñas a RawTree (una fila por producto).
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
X, slot, buy, new, when, names, prod = d["X"].astype(np.float64), d["slot"], d["buy"], d["new"].astype(bool), d["when"], d["names"], d["prod"]
FEATS = list(d["features"]); FI = {n: i for i, n in enumerate(FEATS)}
Q = """SELECT product_id::String AS p, countIf(event_type::String = 'view') AS v, countIf(event_type::String = 'add_to_cart') AS a,
countIf(event_type::String = 'purchase') AS b FROM {table} {where} GROUP BY p"""
hist = {r["p"]: (float(r["v"]), float(r["a"]), float(r["b"])) for r in rec.query(Q.format(table="shop.history_events", where=""), timeout=300)}
live = {r["p"]: (float(r["v"]), float(r["a"]), float(r["b"])) for r in rec.query(Q.format(table="shop.events", where=""), timeout=300)}
print(f"productos con eventos: histórico {len(hist)}, hoy {len(live)} ({time.time() - t0:.0f}s)")


def table(src):
    a = np.array([src.get(n, (0.0, 0.0, 0.0)) for n in names])
    v, c, b = a.T
    r_buy, r_cart = b.sum() / v.sum(), c.sum() / v.sum()
    k = 30.0   # suavizado: 30 vistas "de regalo" con la tasa media
    return {"log vistas": np.log1p(v), "log carritos": np.log1p(c), "log compras": np.log1p(b),
            "tasa compra/vista": np.log((b + k * r_buy) / (v + k)), "tasa carrito/vista": np.log((c + k * r_cart) / (v + k))}


P = {f"hist: {k}": v[prod] for k, v in table(hist).items()}
P.update({f"hoy: {k}": v[prod] for k, v in table(live).items()})
BASE = [n for n in FEATS if n not in ("cart_no_buy", "viewed_no_cart", "organic", "on_sale", "cart_live", "view_live", "pop_log")]
onehot = np.eye(10)[slot][:, 1:]
block = np.minimum((np.argsort(np.argsort(when, kind="stable"), kind="stable") * 5 // len(when)), 4)   # 5 bloques de tiempo


def cv(mask, extra, C=0.3):
    cols = [X[:, FI[n]] for n in BASE] + [X[:, FI[e]] if e in FI else P[e] for e in extra]
    Z = np.stack(cols, 1)
    ll, pr, yy = [], [], []
    for f in range(5):
        tr, te = mask & (block != f), mask & (block == f)
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        lr = LogisticRegression(C=C, max_iter=3000).fit(np.hstack([(Z[tr] - mu) / sd, onehot[tr]]), buy[tr])
        p = lr.predict_proba(np.hstack([(Z[te] - mu) / sd, onehot[te]]))[:, 1]
        pr.append(p); yy.append(buy[te])
    p, y = np.concatenate(pr), np.concatenate(yy)
    mu, sd = Z[mask].mean(0), Z[mask].std(0) + 1e-9
    lr = LogisticRegression(C=C, max_iter=3000).fit(np.hstack([(Z[mask] - mu) / sd, onehot[mask]]), buy[mask])
    return log_loss(y, p), roc_auc_score(y, p), lr.coef_[0][len(BASE):len(BASE) + len(extra)]


tests = [("señales de v14 (pop_glob y pop_sub)", []), ("+ pop_log (v23)", ["pop_log"]),
         ("+ hist: log compras", ["hist: log compras"]), ("+ hoy: log compras", ["hoy: log compras"]),
         ("+ hist: tasa compra/vista", ["hist: tasa compra/vista"]), ("+ hoy: tasa compra/vista", ["hoy: tasa compra/vista"]),
         ("+ hoy: tasa carrito/vista", ["hoy: tasa carrito/vista"]),
         ("+ pop_log + tasas de hoy", ["pop_log", "hoy: tasa compra/vista", "hoy: tasa carrito/vista"]),
         ("+ pop_log + tasas hist y hoy", ["pop_log", "hist: tasa compra/vista", "hoy: tasa compra/vista"]),
         ("+ todo (volumen y tasas, hist y hoy)", ["pop_log"] + list(P))]
for label, seg in (("COMPRADORES NUEVOS", new), ("COMPRADORES CONOCIDOS", ~new)):
    print(f"\n## {label}: {int(seg.sum()) // 10} respuestas, {int(buy[seg].sum())} compras (validación cruzada, 5 bloques de tiempo)")
    print(f"   {'modelo':<40}{'log-loss':>10}{'mejora':>9}{'AUC':>8}   pesos estandarizados de lo añadido")
    base_ll = None
    for name, extra in tests:
        ll, auc, co = cv(seg, extra)
        base_ll = base_ll or ll
        print(f"   {name:<40}{ll:>10.5f}{100 * (base_ll - ll) / base_ll:>8.2f}%{auc:>8.4f}   " + ", ".join(f"{e.split(': ')[-1][:18]} {c:+.2f}" for e, c in zip(extra, co))[:150])
print(f"\ntotal {time.time() - t0:.0f}s")
