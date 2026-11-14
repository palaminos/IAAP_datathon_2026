"""Parecido entre productos con EASE (Steck, 2019), en el mismo formato que cf_sim.npz. Solo lee de RawTree.

    python train_ease.py                 # histórico de 60 días -> cf_ease.npz (va en el repo)
    python train_ease.py --live-until "2026-11-15 06:00:00" --out cf_ease_live.npz    # + carritos y compras de hoy

Qué hace: con la matriz X (comprador × producto, 1 si lo metió en el carrito o lo compró) calcula de una vez
cuánto ayuda cada producto a predecir cada otro, descontando lo que ya explican los demás:
    P = (X'X + lambda·I)^-1 ;  B = -P / diag(P) ;  diag(B) = 0
Se guardan los 100 valores positivos más altos de cada producto. En producción se usa igual que cf_sim.npz:
la señal `cf` de un producto es la suma de B con todo lo que el comprador metió en el carrito o compró.

Por qué: el coseno de train_cf.py mira los productos de dos en dos; EASE los mira todos a la vez. En el
histórico del carrusel (analysis/ease_test.py, productos que el comprador nunca había tocado, 1.909 compras),
el 5 % de impresiones con mejor nota se compra 19 veces más que el resto con EASE top-100, frente a 14 veces
con el coseno (log-loss 0,00351 frente a 0,00362; solo con el puesto, 0,00417).
"""
import sys
import time

import numpy as np

import model
import train_cf
from monitor import query

LAMBDA = 300.0
TOP = 100

if __name__ == "__main__":
    t0 = time.time()
    live_until = sys.argv[sys.argv.index("--live-until") + 1] if "--live-until" in sys.argv else None
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "cf_ease.npz"
    liked = train_cf.LIKED.format(cond="", live=train_cf.LIVE.format(live_until=live_until) if live_until else "")
    rows = []
    for k in range(4):   # por trozos: el resultado entero pasa del límite de 100 MiB
        rows += query(f"SELECT u, p FROM ({liked}) WHERE cityHash64(u) % 4 = {k}", timeout=600)
    users = {u: i for i, u in enumerate(sorted({r["u"] for r in rows}))}
    products = sorted({r["p"] for r in rows})
    pi = {p: i for i, p in enumerate(products)}
    n = len(products)
    from scipy.sparse import csr_matrix
    X = csr_matrix((np.ones(len(rows), np.float32), ([users[r["u"]] for r in rows], [pi[r["p"]] for r in rows])), shape=(len(users), n))
    X.data[:] = 1.0
    G = (X.T @ X).toarray().astype(np.float64)
    print(f"{len(users)} compradores × {n} productos, {X.nnz} pares ({time.time() - t0:.0f}s)", flush=True)
    P = np.linalg.inv(G + LAMBDA * np.eye(n))
    B = (-P / np.diag(P)[None, :]).astype(np.float32)
    np.fill_diagonal(B, 0)
    del P, G
    keep = np.argpartition(-B, TOP, axis=1)[:, :TOP]
    r = np.repeat(np.arange(n), TOP)
    c = keep.ravel()
    v = B[r, c]
    ok = v > 0
    i, j, v = r[ok].astype(np.int32), c[ok].astype(np.int32), v[ok].astype(np.float32)
    S = csr_matrix((v, (i, j)), shape=(n, n))
    rng = np.random.default_rng(0)   # misma escala que train_cf: un producto "relacionado" típico queda en ~1 tras log1p
    sums = np.concatenate([np.asarray(S[rng.choice(n, 25, replace=False)].sum(0)).ravel() for _ in range(200)])
    scale = float((np.e - 1) / np.median(sums[sums > 0]))
    train_cf.save({"products": products, "i": i, "j": j, "v": v, "scale": scale}, model.WEIGHTS_FILE.with_name(out))
    print(f"{len(v)} pares, valor mediano {np.median(v):.4f}, escala {scale:.1f} ({time.time() - t0:.0f}s) -> {out}")
