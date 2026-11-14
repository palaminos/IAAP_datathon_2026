"""Variantes de EASE en el histórico: ¿con qué datos se calcula mejor el parecido? Solo lee (caché de cf_lift.py).

    python analysis/ease_variants.py     # ~3 min

Misma prueba que analysis/ease_test.py (productos nunca tocados, compradores con algún carrito o compra;
semana 1 ajusta, semana 2 mide). Variantes de la matriz comprador × producto con la que se calcula EASE:
  A  carrito o compra = 1 (la de cf_ease.npz)          B  solo compras
  C  carrito o compra o 3+ vistas                      D  carrito = 1, compra = 2
y combinaciones de dos señales a la vez (EASE + coseno; EASE de compras + EASE de carritos).
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
P, M, n = model.PRODUCTS.idx, model.MASK, ctx.items.n
nu = max(users.row.values()) + 1
trip = {k: ([], [], []) for k in "ABCD"}
liked_of = {}
for u, row in users.row.items():
    mine = users.prod.get(row, {})
    lk = []
    for i, pk in mine.items():
        if i >= n:
            continue
        v, a, b = pk & M, (pk >> 20) & M, pk >> 40
        if a or b:
            lk.append(i)
            for k, val in (("A", 1.0), ("D", 2.0 if b else 1.0)):
                trip[k][0].append(row); trip[k][1].append(i); trip[k][2].append(val)
        if b:
            trip["B"][0].append(row); trip["B"][1].append(i); trip["B"][2].append(1.0)
        if a or b or v >= 3:
            trip["C"][0].append(row); trip["C"][1].append(i); trip["C"][2].append(1.0)
    liked_of[u] = lk
rows_idx = np.arange(n)[:, None]


def ease(k, lam=300.0, top=100):
    r, c, v = trip[k]
    X = csr_matrix((np.array(v, np.float32), (r, c)), shape=(nu, n))
    G = (X.T @ X).toarray().astype(np.float64)
    Pm = np.linalg.inv(G + lam * np.eye(n))
    B = (-Pm / np.diag(Pm)[None, :]).astype(np.float32)
    np.fill_diagonal(B, 0)
    keep = np.argpartition(-B, top, axis=1)[:, :top]
    vals = np.maximum(B[rows_idx, keep], 0)
    return csr_matrix((vals.ravel(), (np.repeat(np.arange(n), top), keep.ravel())), shape=(n, n)), G


tables = {}
tables["A"], G = ease("A")
cnt = np.diag(G).copy(); np.fill_diagonal(G, 0); G[G < 2] = 0
keep = np.argsort(-G, axis=1)[:, :100]
tables["coseno"] = csr_matrix(((G[rows_idx, keep] / np.sqrt(np.maximum(cnt[:, None] * cnt[keep], 1))).ravel().astype(np.float32),
                               (np.repeat(np.arange(n), 100), keep.ravel())), shape=(n, n))
del G
for k in "BCD":
    tables[k], _ = ease(k)
tables["A lambda=100"], _ = ease("A", 100.0)
tables["A lambda=1000"], _ = ease("A", 1000.0)
tables["A top-50"], _ = ease("A", 300.0, 50)
print(f"tablas listas ({time.time() - t0:.0f}s)", flush=True)

feats = {k: [] for k in tables}
recs = []
for s in ctx.sessions:
    lk = liked_of.get(s["u"])
    if not lk:
        continue
    mine = users.prod.get(users.row[s["u"]], {})
    idx = np.array([P[p] for p in s["shown"]])
    w = 1.0 if s["clicked"] else 100 / NEG_PCT
    for k, S in tables.items():
        feats[k].append(np.asarray(S[lk][:, idx].sum(0)).ravel())
    for j in range(10):
        recs.append((s["purchased"] == s["shown"][j], j, w, idx[j] not in mine, s["week"]))
R = np.array(recs, float)
buy, slot, w, fresh, week = R[:, 0] > 0, R[:, 1].astype(int), R[:, 2], R[:, 3] > 0, R[:, 4]
onehot = np.eye(10)[slot][:, 1:]
tr, te = fresh & (week == 0), fresh & (week == 1)
F = {k: np.log1p(np.concatenate(v) * 15) for k, v in feats.items()}
print(f"{int(fresh.sum())} impresiones nunca tocadas, {int(buy[fresh].sum())} compras ({time.time() - t0:.0f}s)\n")


def score(cols):
    Z = np.hstack([np.stack([(F[c] - F[c][tr].mean()) / (F[c][tr].std() + 1e-9) for c in cols], 1), onehot])
    lr = LogisticRegression(C=10, max_iter=2000).fit(Z[tr], buy[tr], sample_weight=w[tr])
    p = lr.predict_proba(Z[te])[:, 1]
    return roc_auc_score(buy[te], p, sample_weight=w[te]), log_loss(buy[te], p, sample_weight=w[te])


names = {"A": "A  carrito o compra (actual)", "B": "B  solo compras", "C": "C  + 3 o más vistas", "D": "D  carrito 1, compra 2"}
print(f"{'señal':<44}{'AUC semana 2':>13}{'log-loss':>10}")
for cols in (["coseno"], ["A"], ["B"], ["C"], ["D"], ["A lambda=100"], ["A lambda=1000"], ["A top-50"], ["A", "coseno"], ["A", "B"], ["A", "C"], ["A", "B", "C", "coseno"]):
    a, ll = score(cols)
    print(f"{' + '.join(names.get(c, c) for c in cols):<44}{a:>13.4f}{ll:>10.5f}")
print(f"\ntotal {time.time() - t0:.0f}s")
