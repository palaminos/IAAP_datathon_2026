"""Filtrado colaborativo de producto a producto: "quien metió X en el carrito o lo compró, también Y".

    python train_cf.py        # solo lee de RawTree; guarda cf_sim.npz (va en el repo)

Con los 60 días de shop.history_events se cuenta, para cada par de productos, cuántos usuarios metieron
los dos en el carrito o los compraron. La similitud es el coseno de esa co-ocurrencia (se guardan los 100
más parecidos de cada producto). En producción, la señal `cf` de un producto para un usuario es la suma de
su similitud con todo lo que ese usuario ha metido en el carrito o comprado (60 días + hoy).

Por qué: en el histórico del carrusel (analysis/cf_lift.py), los productos con similitud muy alta se
compran 15 veces más que la media y los que no tienen ninguna, casi nunca (0,001 %); y sigue funcionando
con productos que el usuario nunca había tocado y dentro de las subcategorías que ya le gustan.
"""
import time

import numpy as np

import model
from monitor import query

PAIRS = """
SELECT a.p AS p, b.p AS q, count() AS n
FROM (SELECT DISTINCT user_id::String AS u, product_id::String AS p FROM shop.history_events
      WHERE event_type::String IN ('purchase', 'add_to_cart') {cond}) AS a
INNER JOIN (SELECT DISTINCT user_id::String AS u, product_id::String AS p FROM shop.history_events
            WHERE event_type::String IN ('purchase', 'add_to_cart') {cond}) AS b ON a.u = b.u
WHERE a.p != b.p GROUP BY p, q HAVING n >= 2 ORDER BY n DESC LIMIT 100 BY p"""
COUNTS = """SELECT product_id::String AS p, uniqExact(user_id::String) AS n FROM shop.history_events
WHERE event_type::String IN ('purchase', 'add_to_cart') {cond} GROUP BY p"""


def build(until: str | None = None) -> dict:
    """Similitud producto-producto con los carritos y compras del histórico (anteriores a `until`, si se da:
    así se puede entrenar con el histórico sin usar información posterior a cada carrusel)."""
    from scipy.sparse import csr_matrix
    cond = f"AND ts::DateTime64(3) < '{until}'" if until else ""
    cnt = {r["p"]: float(r["n"]) for r in query(COUNTS.format(cond=cond), timeout=600)}
    pairs = query(PAIRS.format(cond=cond), timeout=900)
    products = sorted(cnt)
    pi = {p: i for i, p in enumerate(products)}
    i = np.array([pi[r["p"]] for r in pairs], np.int32)
    j = np.array([pi[r["q"]] for r in pairs], np.int32)
    v = np.array([float(r["n"]) / np.sqrt(cnt[r["p"]] * cnt[r["q"]]) for r in pairs], np.float32)
    # Escala: que la similitud total típica de un producto "relacionado" quede en torno a 1 tras log1p.
    # Se estima con la suma de similitudes de 25 productos al azar (un usuario medio tiene ~25 en carrito/compra).
    rng = np.random.default_rng(0)
    S = csr_matrix((v, (i, j)), shape=(len(products), len(products)))
    sums = np.concatenate([np.asarray(S[rng.choice(len(products), 25, replace=False)].sum(0)).ravel() for _ in range(200)])
    scale = float((np.e - 1) / np.median(sums[sums > 0]))
    return {"products": products, "i": i, "j": j, "v": v, "scale": scale}


def save(cf: dict, path) -> None:
    np.savez_compressed(path, products=np.array(cf["products"]), i=cf["i"], j=cf["j"], v=cf["v"], scale=np.float32(cf["scale"]))


if __name__ == "__main__":
    t0 = time.time()
    cf = build()
    save(cf, model.CF_FILE)
    print(f"{len(cf['products'])} productos, {len(cf['v'])} pares, similitud mediana {np.median(cf['v']):.3f}, "
          f"escala {cf['scale']:.1f} ({time.time() - t0:.0f}s) -> {model.CF_FILE.name}")
