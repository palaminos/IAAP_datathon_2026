"""¿Hay una forma mejor de calcular el "parecido entre productos"? EASE frente al coseno actual.

    python analysis/ease_test.py     # desde la raíz del repo. Solo lee (usa la caché de cf_lift.py). ~3 min

El coseno de co-ocurrencia (train_cf.py) mira los productos de dos en dos. EASE (Steck, 2019) aprende de
golpe, con una regresión cerrada, cuánto ayuda cada producto a predecir cada otro, descontando lo que ya
explican los demás: B = I - P / diag(P), con P = (X'X + lambda·I)^-1 y X = matriz comprador × producto de
carritos y compras. Suele ganar con claridad a los vecinos por coseno.

Se mide en el histórico del carrusel (1-14 nov, todo calculado con datos anteriores al 1-nov), solo con
productos que el comprador NO había tocado nunca (en vivo lo ya comprado no se recompra): para cada señal,
AUC de compra a igual puesto y tasa de compra del 5 % de impresiones con la señal más alta.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.simplefilter("ignore")
from experiments import *  # noqa: F401,F403,E402

t0 = time.time()
ctx = Ctx()
it, users = ctx.items, ctx.users
P = model.PRODUCTS.idx
M = model.MASK
n = it.n
# X: comprador × producto (1 si lo metió en el carrito o lo compró antes del corte)
ru, ci = [], []
liked_of = {}
for u, row in users.row.items():
    mine = users.prod.get(row, {})
    lk = [i for i, pk in mine.items() if i < n and ((pk >> 20) & M or pk >> 40)]
    liked_of[u] = lk
    ru += [row] * len(lk); ci += lk
X = csr_matrix((np.ones(len(ru), np.float32), (ru, ci)), shape=(max(users.row.values()) + 1, n))
G = (X.T @ X).toarray().astype(np.float64)
cnt = np.diag(G).copy()
print(f"X: {X.shape[0]} compradores × {n} productos, {X.nnz} pares; productos con algún comprador: {(cnt > 0).sum()} ({time.time() - t0:.0f}s)", flush=True)

# coseno actual (todos los pares con n>=2, top 100 por producto), como train_cf.py
Gc = G.copy(); np.fill_diagonal(Gc, 0); Gc[Gc < 2] = 0
keep = np.argsort(-Gc, axis=1)[:, :100]
Scos = np.zeros((n, n), np.float32); rows = np.arange(n)[:, None]
Scos[rows, keep] = Gc[rows, keep] / np.sqrt(np.maximum(cnt[:, None] * cnt[keep], 1))
variants = {"coseno top-100 (actual)": Scos}
del Gc


def topk(B, k):
    """Se queda con los k valores positivos más altos de cada fila (como se guardará para producción)."""
    out = np.zeros((n, n), np.float32)
    keep = np.argpartition(-B, k, axis=1)[:, :k]
    out[rows, keep] = np.maximum(B[rows, keep], 0)
    return out


for lam in (300.0, 1000.0):
    Pm = np.linalg.inv(G + lam * np.eye(n))
    B = (-Pm / np.diag(Pm)[None, :]).astype(np.float32)
    np.fill_diagonal(B, 0)
    del Pm
    if lam == 300.0:
        variants["EASE lambda=300, entera"] = B
        for k in (100, 200, 500):
            variants[f"EASE lambda=300, top-{k} por fila"] = topk(B, k)
    else:
        variants["EASE lambda=1000, top-200 por fila"] = topk(B, 200)
    print(f"EASE lambda={lam:.0f} lista ({time.time() - t0:.0f}s)", flush=True)
del G

# impresiones de productos nunca tocados, de compradores con algún carrito o compra
recs, feats = [], {k: [] for k in variants}
for s in ctx.sessions:
    lk = liked_of.get(s["u"])
    if not lk:
        continue
    row = users.row[s["u"]]
    mine = users.prod.get(row, {})
    idx = np.array([P[p] for p in s["shown"]])
    w = 1.0 if s["clicked"] else 100 / NEG_PCT
    fresh = np.array([i not in mine for i in idx])
    for k, S in variants.items():
        feats[k].append(S[lk][:, idx].sum(0))
    for j in range(10):
        recs.append((s["purchased"] == s["shown"][j], j, w, fresh[j], s["week"]))
R = np.array(recs, float)
buy, slot, w, fresh, week = R[:, 0] > 0, R[:, 1].astype(int), R[:, 2], R[:, 3] > 0, R[:, 4]
print(f"{len(R)} impresiones de compradores con historial; nunca tocadas: {int(fresh.sum())} con {int(buy[fresh].sum())} compras ({time.time() - t0:.0f}s)\n")
onehot = np.eye(10)[slot][:, 1:]
tr, te = fresh & (week == 0), fresh & (week == 1)
print(f"{'señal (productos nunca tocados)':<34}{'AUC semana 2':>13}{'log-loss':>10}{'compra top 5 %':>16}{'compra resto':>14}{'veces':>7}")
base = LogisticRegression(C=10, max_iter=2000).fit(onehot[tr], buy[tr], sample_weight=w[tr])
p0 = base.predict_proba(onehot[te])[:, 1]
print(f"{'solo el puesto':<34}{roc_auc_score(buy[te], p0, sample_weight=w[te]):>13.4f}{log_loss(buy[te], p0, sample_weight=w[te]):>10.5f}")
for k in variants:
    f = np.concatenate(feats[k])
    # dos transformaciones: valor tal cual (estandarizado) y su rango dentro de la muestra
    best = None
    for name, x in (("", f), (" (log)", np.log1p(np.maximum(f, 0) * 15))):
        z = (x - x[tr].mean()) / (x[tr].std() + 1e-9)
        Z = np.hstack([z[:, None], onehot])
        lr = LogisticRegression(C=10, max_iter=2000).fit(Z[tr], buy[tr], sample_weight=w[tr])
        p = lr.predict_proba(Z[te])[:, 1]
        r = (roc_auc_score(buy[te], p, sample_weight=w[te]), log_loss(buy[te], p, sample_weight=w[te]), name)
        if best is None or r[1] < best[1]:
            best = r
    cut = np.quantile(f[te], 0.95)
    top = te & (f >= cut); rest = te & (f < cut)
    rt, rr = (buy[top] * w[top]).sum() / w[top].sum(), (buy[rest] * w[rest]).sum() / w[rest].sum()
    print(f"{k + best[2]:<34}{best[0]:>13.4f}{best[1]:>10.5f}{100 * rt:>15.3f}%{100 * rr:>13.3f}%{rt / rr:>7.1f}")
print(f"\ntotal {time.time() - t0:.0f}s")
