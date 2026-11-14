"""Señal de "usuarios parecidos": factoriza la tabla usuario × producto del histórico orgánico.

    python train_mf.py        # solo lee de RawTree; guarda mf_factors.npz (~2 MB, va en el repo)

Con shop.history_events se monta la tabla usuario × producto (cuánto ha interactuado cada usuario con
cada producto: vistas + 3·carritos + 5·compras, en logaritmo) y se resume en K factores con SVD. Cada
usuario y cada producto queda descrito por K números ("tipos de gusto"); su afinidad es el producto de
ambos. K=8 se eligió en validación (analysis/mf_test.py): +4,1 % de NDCG@10 offline.
"""
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

import model
import recommender as rec

K = 8
PAIRS = """
SELECT u, groupArray(p) AS p, groupArray(w) AS w FROM (
    SELECT user_id::String AS u, product_id::String AS p,
           countIf(event_type = 'view') + 3 * countIf(event_type = 'add_to_cart') + 5 * countIf(event_type = 'purchase') AS w
    FROM shop.history_events
    WHERE cityHash64(user_id::String) % 16 = {i}
    GROUP BY u, p
) GROUP BY u"""

if __name__ == "__main__":
    t0 = time.time()
    users, pids, rows, cols, vals = {}, {}, [], [], []
    with ThreadPoolExecutor(4) as pool:
        for part in pool.map(lambda i: rec.query(PAIRS.format(i=i), timeout=600), range(16)):
            for r in part:
                ui = users.setdefault(r["u"], len(users))
                for p, w in zip(r["p"], r["w"]):
                    rows.append(ui), cols.append(pids.setdefault(p, len(pids))), vals.append(math.log1p(float(w)))
    M = csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(len(users), len(pids)))
    svd = TruncatedSVD(n_components=K, random_state=0).fit(M)
    U, V = svd.transform(M).astype(np.float32), svd.components_.T.astype(np.float32)
    np.savez_compressed(model.MF_FILE, users=np.array(list(users)), U=U.astype(np.float16), products=np.array(list(pids)), V=V)
    print(f"{M.shape[0]} usuarios × {M.shape[1]} productos, {M.nnz} pares → {K} factores "
          f"(varianza explicada {svd.explained_variance_ratio_.sum():.2f}) en {time.time() - t0:.0f}s → {model.MF_FILE.name}")
