"""Calcula features con SQL dentro de RawTree (la T) y las inserta en team_NN.scratch (la L).

    python build_scratch_features.py

Ejecutar UNA vez: cada ejecución añade filas (la clave no tiene permiso para INSERT ... SELECT ni
TRUNCATE, así que el resultado del SELECT se inserta por la API). Solo lee de shop.*.
"""
import time

from monitor import API, TEAM, http, query

FEATURE_SETS = {
    # Una fila por producto: popularidad y posición en ventas (últimos 14 días del histórico)
    "item_popularity_v1": """
SELECT 'item_popularity_v1' AS feature_set, p AS product_id, sub AS subcategory, cat AS category,
       b AS purchases_14d, a AS carts_14d, v AS views_14d, s AS pop_score,
       row_number() OVER (ORDER BY s DESC) AS rank_global,
       row_number() OVER (PARTITION BY sub ORDER BY s DESC) AS rank_in_subcategory,
       row_number() OVER (PARTITION BY cat ORDER BY s DESC) AS rank_in_category
FROM (
  SELECT c.product_id::String AS p, any(c.subcategory::String) AS sub, any(c.category::String) AS cat,
         countIf(e.event_type = 'purchase') AS b, countIf(e.event_type = 'add_to_cart') AS a,
         countIf(e.event_type = 'view') AS v, 3 * b + a + 0.05 * v AS s
  FROM shop.catalog AS c
  LEFT JOIN (SELECT product_id::String AS pid, event_type::String AS event_type FROM shop.history_events
             WHERE ts::DateTime64(3) >= (SELECT max(ts::DateTime64(3)) FROM shop.history_events) - INTERVAL 14 DAY) AS e
    ON e.pid = c.product_id::String
  GROUP BY p)""",
    # Una fila por usuario: actividad, precio habitual y subcategorías favoritas (60 días)
    "user_profile_v1": """
SELECT 'user_profile_v1' AS feature_set, u AS user_id, sum(n) AS n_events, sum(nb) AS n_purchases,
       sum(slp) / sum(n) AS avg_log_price,
       arraySlice(arrayMap(x -> x.2, arrayReverseSort(groupArray((n, sub)))), 1, 3) AS top_subcategories,
       arraySlice(arrayMap(x -> round(x.1 / sum(n), 4), arrayReverseSort(groupArray((n, sub)))), 1, 3) AS top_subcategory_shares
FROM (
  SELECT e.user_id::String AS u, c.subcategory::String AS sub, count() AS n,
         countIf(e.event_type = 'purchase') AS nb, sum(log(e.price_eur::Float64)) AS slp
  FROM shop.history_events AS e INNER JOIN shop.catalog AS c ON c.product_id::String = e.product_id::String
  GROUP BY u, sub)
GROUP BY u""",
    # Pares vista→compra en la misma sesión: quien vio product_id acabó comprando bought_product_id
    "cobuy_v1": """
SELECT 'cobuy_v1' AS feature_set, a.p AS product_id, b.q AS bought_product_id, count() AS sessions,
       row_number() OVER (PARTITION BY a.p ORDER BY count() DESC) AS rank
FROM (SELECT DISTINCT session_id::String AS s, product_id::String AS p FROM shop.history_events WHERE event_type = 'view') AS a
INNER JOIN (SELECT DISTINCT session_id::String AS s, product_id::String AS q FROM shop.history_events WHERE event_type = 'purchase') AS b
  ON a.s = b.s
WHERE a.p != b.q
GROUP BY product_id, bought_product_id
ORDER BY sessions DESC
LIMIT 10 BY product_id""",
}

if __name__ == "__main__":
    for name, sql in FEATURE_SETS.items():
        t = time.time()
        rows = query(sql, timeout=900)
        for i in range(0, len(rows), 5000):
            http.post(f"{API}/v1/tables/scratch", params={"database": TEAM}, json=rows[i:i + 5000], timeout=120).raise_for_status()
        print(f"{name}: {len(rows)} filas insertadas en {TEAM}.scratch ({time.time() - t:.0f}s)")
