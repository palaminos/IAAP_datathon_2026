"""¿Aporta una señal de "usuarios parecidos" (factorización de matrices) al modelo actual? Solo lee.

    python analysis/mf_test.py     # desde la raíz del repo

Con los eventos orgánicos ANTERIORES al corte (1-nov) se monta la tabla usuario × producto (cuánto ha
interactuado cada usuario con cada producto) y se resume en K factores (SVD). La afinidad de un usuario
por un producto es el producto de sus factores: alta si usuarios con gustos parecidos interactuaron con
él. Se añade como señal extra y se mide con el mismo split y la misma métrica que el resto de ideas
(experiments.py): se elige la regularización en validación y se mide una vez en la semana de prueba.
"""
import math
import pickle
import sys
import time
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import *  # noqa: F401,F403,E402

warnings.simplefilter("ignore")
PAIRS = """
SELECT u, groupArray(p) AS p, groupArray(w) AS w FROM (
    SELECT user_id::String AS u, product_id::String AS p,
           countIf(event_type = 'view') + 3 * countIf(event_type = 'add_to_cart') + 5 * countIf(event_type = 'purchase') AS w
    FROM shop.history_events
    WHERE ts::DateTime64(3) < '{until}' AND cityHash64(user_id::String) % 16 = {i}
    GROUP BY u, p
) GROUP BY u"""

ctx = Ctx()
X, slot, buy, click, weight, week = build_train(ctx)


def load_pairs():
    users, rows, cols, vals = {}, [], [], []
    with ThreadPoolExecutor(4) as pool:
        for part in pool.map(lambda i: rec.query(PAIRS.format(until=UNTIL, i=i), timeout=600), range(16)):
            for r in part:
                ui = users.setdefault(r["u"], len(users))
                for p, w in zip(r["p"], r["w"]):
                    j = model.PRODUCTS.idx.get(p)
                    if j is not None:
                        rows.append(ui), cols.append(j), vals.append(math.log1p(float(w)))
    return users, np.array(rows, np.int32), np.array(cols, np.int32), np.array(vals, np.float32)


t0 = time.time()
users, rows, cols, vals = cached("mf_pairs", load_pairs)
M = csr_matrix((vals, (rows, cols)), shape=(len(users), ctx.items.n))
print(f"tabla usuario × producto: {M.shape[0]} usuarios × {M.shape[1]} productos, {M.nnz} pares ({time.time() - t0:.0f}s)")

is_val_s = np.array([zlib.crc32(s["u"].encode()) % 5 == 0 for s in ctx.sessions])
val_rows = np.repeat(is_val_s, 10)
A = week == 0
base_cols = cols_of(BASE)


def mf_columns(U, V):
    """Para cada carrusel: afinidad MF de sus 10 productos (cruda y normalizada dentro del usuario)."""
    out = np.zeros((len(ctx.sessions) * 10, 2), np.float32)
    for g, s in enumerate(ctx.sessions):
        ui = users.get(s["u"])
        if ui is not None:
            sc = V @ U[ui]
            idx = [model.PRODUCTS.idx[p] for p in s["shown"]]
            out[g * 10:(g + 1) * 10, 0] = sc[idx]
            out[g * 10:(g + 1) * 10, 1] = (sc[idx] - sc.mean()) / (sc.std() + 1e-9)
    return out


def fit(Z, mask, C):
    mu, sd = Z[mask].mean(0), Z[mask].std(0) + 1e-9
    D = np.hstack([(Z[mask] - mu) / sd, np.eye(10, dtype=np.float32)[slot[mask]][:, 1:]])
    lr = LogisticRegression(C=C, max_iter=5000, tol=1e-6).fit(D, buy[mask], sample_weight=weight[mask])
    return (lr.coef_[0][:Z.shape[1]] / sd).astype(np.float32)


def ndcgs(cands, sessions, U, V):
    out = {k: [] for k in cands}
    for s in sessions:
        F = ctx.features(s)[:, base_cols]
        ui = users.get(s["u"])
        mf = np.zeros((len(F), 2), np.float32)
        if ui is not None:
            sc = V @ U[ui]
            mf[:, 0], mf[:, 1] = sc, (sc - sc.mean()) / (sc.std() + 1e-9)
        full = np.hstack([F, mf])
        t = model.PRODUCTS.idx[s["purchased"]]
        for k, w in cands.items():
            sc = full[:, :len(w)] @ w
            r = 1 + int((sc > sc[t]).sum())
            out[k].append(1 / math.log2(1 + r) if r <= 10 else 0.0)
    return {k: np.array(v) for k, v in out.items()}


val = [s for s, v in zip(ctx.sessions, is_val_s) if v and s["week"] == 0 and s["purchased"]]
test = [s for s in ctx.sessions if s["week"] == 1 and s["purchased"]]
CS = (0.01, 0.1, 1.0)
KS = (4, 8, 12, 16, 24, 32, 64)
rng = np.random.default_rng(0)
nb = len(base_cols)
factors, chosen = {}, {}
print("\nVALIDACIÓN (NDCG@10) — se elige aquí el nº de factores y la regularización:")
base_val = {C: fit(X[:, base_cols], A & ~val_rows, C) for C in CS}
rv = ndcgs({("base", C): w for C, w in base_val.items()}, val, np.zeros((len(users), 1), np.float32), np.zeros((ctx.items.n, 1), np.float32))
print(f"  {'base (16 señales)':<22}" + "".join(f"  C={C}: {rv[('base', C)].mean():.4f}" for C in CS))
best_base_C = max(CS, key=lambda C: rv[("base", C)].mean())
for K in KS:
    svd = TruncatedSVD(n_components=K, random_state=0).fit(M)
    U, V = svd.transform(M).astype(np.float32), svd.components_.T.astype(np.float32)
    Z = np.hstack([X[:, base_cols], mf_columns(U, V)])
    factors[K] = (U, V, Z)
    r = ndcgs({C: fit(Z, A & ~val_rows, C) for C in CS}, val, U, V)
    chosen[K] = max(((r[C].mean(), C) for C in CS))
    print(f"  {'base + MF ' + str(K) + ' factores':<22}" + "".join(f"  C={C}: {r[C].mean():.4f}" for C in CS))
K = max(chosen, key=lambda k: chosen[k][0])
C = chosen[K][1]
U, V, Z = factors[K]
final = {"base": fit(Z[:, :nb], A, best_base_C), "base+MF": fit(Z, A, C)}
rt = ndcgs(final, test, U, V)
d = rt["base+MF"] - rt["base"]
boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
home = np.array([s["page"] == "home" for s in test])
print(f"\nElegido en validación: {K} factores, C={C}")
print(f"PRUEBA ({len(test)} compras, una sola medida):")
print(f"  base     C={best_base_C}: NDCG@10 {rt['base'].mean():.4f}  Recall@10 {(rt['base'] > 0).mean():.4f}  | home {rt['base'][home].mean():.3f}  resto {rt['base'][~home].mean():.3f}")
print(f"  base+MF  C={C}: NDCG@10 {rt['base+MF'].mean():.4f}  Recall@10 {(rt['base+MF'] > 0).mean():.4f}  | home {rt['base+MF'][home].mean():.3f}  resto {rt['base+MF'][~home].mean():.3f}")
print(f"  diferencia NDCG@10: {d.mean():+.4f} ({100 * d.mean() / rt['base'].mean():+.1f} %)  IC95 [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]")
print("  pesos:", ", ".join(f"{n} {w:+.2f}" for n, w in zip(BASE + ["mf", "mf_z"], final["base+MF"])))
