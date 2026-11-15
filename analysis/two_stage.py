"""¿Predice mejor la compra un modelo en dos pasos (clic y, después, compra dado el clic)?

    python analysis/two_stage.py      # usa cache/live_matrix.npz

Hay unas 3 veces más clics que compras. Si la compra pasa por el clic, P(compra) = P(clic) × P(compra | clic), y
cada parte se puede aprender con más datos de los que tiene la compra directa. Validación cruzada en 5 bloques de
tiempo, compradores conocidos y nuevos por separado; se compara con el modelo directo (el de producción).
También mide qué orden deja cada modelo: de las compras, cuántas habrían quedado en los 3 primeros puestos si se
reordenan los 10 productos enseñados según cada modelo (sin efecto de posición).
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
X, slot, buy, click, new, when = d["X"].astype(np.float64), d["slot"], d["buy"], d["click"], d["new"].astype(bool), d["when"]
FEATS = list(d["features"]); FI = {n: i for i, n in enumerate(FEATS)}
DROP = ("cart_no_buy", "viewed_no_cart", "organic", "on_sale", "cart_live", "view_live", "pop_log", "rate_buy")
Z = X[:, [FI[n] for n in FEATS if n not in DROP]]
onehot = np.eye(10)[slot][:, 1:]
block = np.minimum((np.argsort(np.argsort(when, kind="stable"), kind="stable") * 5 // len(when)), 4)
req = np.arange(len(X)) // 10
print(f"compras sin clic: {int((buy & ~click).sum())} de {int(buy.sum())} | clics: {int(click.sum())} | compra dado clic: {100 * buy[click].mean():.1f} %")


def fit(tr, y, C, with_slot=True):
    mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
    A = np.hstack([(Z[tr] - mu) / sd, onehot[tr]]) if with_slot else (Z[tr] - mu) / sd
    lr = LogisticRegression(C=C, max_iter=3000).fit(A, y[tr])
    k = Z.shape[1]
    w, b = lr.coef_[0][:k] / sd, lr.intercept_[0] - (lr.coef_[0][:k] * mu / sd).sum()
    return w, b, (lr.coef_[0][k:] if with_slot else np.zeros(9))


def top3(score, mask):
    """De las compras de `mask`: % que quedaría en los 3 primeros puestos reordenando cada lista por `score`."""
    s = score.reshape(-1, 10)
    rank = (-s).argsort(1).argsort(1).reshape(-1)
    k = mask & buy
    return 100 * (rank[k] < 3).mean(), 100 * (rank[k] == 0).mean()


for label, seg in (("COMPRADORES CONOCIDOS", ~new), ("COMPRADORES NUEVOS", new)):
    print(f"\n## {label}: {int(seg.sum()) // 10} respuestas, {int(buy[seg].sum())} compras, {int(click[seg].sum())} clics")
    P = {k: np.zeros(len(X)) for k in ("directo", "dos pasos", "mezcla")}
    S = {k: np.zeros(len(X)) for k in ("directo", "dos pasos", "mezcla", "solo clic")}
    for f in range(5):
        tr, te = seg & (block != f), seg & (block == f)
        wd, bd, sl = fit(tr, buy, 0.3)
        slot_te = np.concatenate([[0.0], sl])[slot[te]]
        S["directo"][te] = Z[te] @ wd
        P["directo"][te] = 1 / (1 + np.exp(-(Z[te] @ wd + bd + slot_te)))
        wc, bc, slc = fit(tr, click, 0.3)
        wb, bb, _ = fit(tr & click, buy, 0.1, with_slot=False)
        pc = 1 / (1 + np.exp(-(Z[te] @ wc + bc + np.concatenate([[0.0], slc])[slot[te]])))
        pb = 1 / (1 + np.exp(-(Z[te] @ wb + bb)))
        P["dos pasos"][te] = np.clip(pc * pb, 1e-9, 1)
        S["dos pasos"][te] = Z[te] @ wc + np.log(pb)          # sin el efecto de la posición
        S["solo clic"][te] = Z[te] @ wc
        P["mezcla"][te] = np.sqrt(P["directo"][te] * P["dos pasos"][te])
        S["mezcla"][te] = 0.5 * (S["directo"][te] + S["dos pasos"][te])
    ll0 = log_loss(buy[seg], P["directo"][seg])
    for name in ("directo", "dos pasos", "mezcla"):
        ll, auc = log_loss(buy[seg], P[name][seg]), roc_auc_score(buy[seg], P[name][seg])
        a, b = top3(S[name], seg)
        print(f"   {name:<10} log-loss {ll:.5f} ({100 * (ll0 - ll) / ll0:+.2f} %)  AUC {auc:.4f} | reordenando: compras en el top-3 {a:.1f} %, en el 1.º {b:.1f} %")
    a, b = top3(S["solo clic"], seg)
    print(f"   {'solo clic':<10} {'':<41} | reordenando: compras en el top-3 {a:.1f} %, en el 1.º {b:.1f} %")
    a, b = top3(-slot.astype(float), seg)
    print(f"   {'(orden que se enseñó)':<52} | compras en el top-3 {a:.1f} %, en el 1.º {b:.1f} %")
print(f"\ntotal {time.time() - t0:.0f}s")
