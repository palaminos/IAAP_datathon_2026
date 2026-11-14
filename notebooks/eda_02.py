"""EDA 02 · preguntas que deciden el diseño del modelo (sin fuga temporal).

    python notebooks/eda_02.py > eda_02.txt      # desde la raíz del repo (lee .env)

1. Cobertura de candidatos: ¿la compra era algo que el usuario ya había visto/comprado ANTES? ¿su categoría top previa?
2. Efecto puro de la posición (descontando la popularidad del producto).
3. Afinidad usuario-producto y ventana de recompra: P(compra | mostrado, relación previa con el producto).
4. Rebajas: mismo producto con y sin descuento (stream en vivo).
5. Cold start: productos nuevos vs su subcategoría; qué compran los usuarios nuevos.

Reutiliza las vistas base y el formato de salida de eda_01.py. ~2-4 min.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eda_01 import CAT, EVH, EVL, S, TEAM, run  # noqa: E402

BUYS = f"(SELECT rid, u, t, pur, page FROM {S} WHERE pur != '')"
HOURS = """multiIf({h} < 1, 'a<1h', {h} < 24, 'b<1d', {h} < 168, 'c<7d', {h} < 720, 'd<30d', 'e30d+')"""
NEW_P = """(SELECT toString(product_id) AS pid, min(ts::DateTime64(3)) AS launch, any(toString(subcategory)) AS sub,
           any(toString(category)) AS cat FROM shop.catalog_updates WHERE toString(change) = 'new_product' GROUP BY pid)"""


def main() -> None:
    t0 = time.time()
    print(f"# EDA_02 team={TEAM}")

    # ------------------------------------------------------------------ 1. cobertura de candidatos
    print("\n# 1. COBERTURA: la compra del carrusel vs historial PREVIO del usuario (fila page='' = total)")
    run("compra ya vista/comprada antes, por antigüedad", f"""
        WITH b AS {BUYS},
             e AS (SELECT u, pid, et, t FROM {EVH} WHERE u IN (SELECT u FROM b) AND et IN ('view', 'purchase'))
        SELECT page, count() AS buys,
            avg(vb > 0) AS viewed_before, avg(v1 > 0) AS viewed_1d, avg(v7 > 0) AS viewed_7d,
            avg(bb > 0) AS bought_before, avg(b7 > 0) AS bought_7d, avg(b30 > 0) AS bought_30d
        FROM (SELECT b.rid, any(b.page) AS page,
                countIf(e.et = 'view' AND e.t < b.t) AS vb,
                countIf(e.et = 'view' AND e.t < b.t AND e.t >= b.t - INTERVAL 1 DAY) AS v1,
                countIf(e.et = 'view' AND e.t < b.t AND e.t >= b.t - INTERVAL 7 DAY) AS v7,
                countIf(e.et = 'purchase' AND e.t < b.t) AS bb,
                countIf(e.et = 'purchase' AND e.t < b.t AND e.t >= b.t - INTERVAL 7 DAY) AS b7,
                countIf(e.et = 'purchase' AND e.t < b.t AND e.t >= b.t - INTERVAL 30 DAY) AS b30
              FROM b LEFT JOIN e ON e.u = b.u AND e.pid = b.pur GROUP BY b.rid)
        GROUP BY ROLLUP(page) ORDER BY page""")
    run("categoria top PREVIA del usuario (sin fuga)", f"""
        WITH b AS {BUYS},
             e AS (SELECT e.u AS u, e.t AS t, c.cat AS cat FROM {EVH} AS e INNER JOIN {CAT} AS c ON c.pid = e.pid
                   WHERE e.u IN (SELECT u FROM b)),
             per AS (SELECT b.rid AS rid, e.cat AS cat, count() AS c FROM b INNER JOIN e ON e.u = b.u
                     WHERE e.t < b.t GROUP BY rid, cat),
             top AS (SELECT rid, argMax(cat, c) AS topcat, max(c) / sum(c) AS top_share FROM per GROUP BY rid)
        SELECT b.page AS page, count() AS buys, countIf(top.rid != '') / buys AS has_history,
               countIf(cp.cat = top.topcat) / countIf(top.rid != '') AS buy_in_prev_topcat,
               avgIf(top.top_share, top.rid != '') AS avg_topcat_share_of_events
        FROM b LEFT JOIN top ON top.rid = b.rid LEFT JOIN {CAT} AS cp ON cp.pid = b.pur
        GROUP BY ROLLUP(b.page) ORDER BY page""")

    # ------------------------------------------------------------------ 2. posición pura
    print("\n# 2. POSICION: compras observadas / esperadas por la tasa media del producto (mult relativo al slot 1)")
    run("multiplicador de posicion", f"""
        WITH imp AS (SELECT p, i AS slot, pur = p AS b FROM {S} ARRAY JOIN shown_a AS p, arrayEnumerate(shown_a) AS i),
             pr AS (SELECT p, avg(b) AS rate FROM imp GROUP BY p)
        SELECT slot, count() AS impressions, sum(b) AS buys, sum(pr.rate) AS expected, buys / expected AS mult,
               mult / first_value(mult) OVER (ORDER BY slot) AS mult_vs_slot1
        FROM imp INNER JOIN pr ON pr.p = imp.p GROUP BY slot ORDER BY slot""")

    # ------------------------------------------------------------------ 3. afinidad usuario-producto
    print("\n# 3. AFINIDAD: P(compra | producto mostrado, relacion previa del usuario con el) · muestra 25 % sesiones")
    lb, lv = HOURS.format(h="dateDiff('hour', lb, st)"), HOURS.format(h="dateDiff('hour', lv, st)")
    run("tasa de compra por relacion previa y si esta en recent_views", f"""
        WITH s AS (SELECT rid, u, t, shown_a, pur, rv_ids FROM {S} WHERE cityHash64(rid) % 4 = 0),
             imp AS (SELECT rid, u, t, p, i, pur = p AS b, has(rv_ids, p) AS in_rv
                     FROM s ARRAY JOIN shown_a AS p, arrayEnumerate(shown_a) AS i),
             e AS (SELECT u, pid, et, t FROM {EVH} WHERE et IN ('view', 'purchase') AND u IN (SELECT u FROM s)),
             j AS (SELECT imp.rid, imp.i, any(imp.b) AS b, any(imp.in_rv) AS in_rv, any(imp.t) AS st,
                          maxIf(e.t, e.et = 'purchase' AND e.t < imp.t) AS lb, maxIf(e.t, e.et = 'view' AND e.t < imp.t) AS lv
                   FROM imp LEFT JOIN e ON e.u = imp.u AND e.pid = imp.p GROUP BY imp.rid, imp.i)
        SELECT multiIf(lb > toDateTime64('2000-01-01', 3), concat('bought_', {lb}),
                       lv > toDateTime64('2000-01-01', 3), concat('viewed_', {lv}), 'never') AS relation,
               in_rv, count() AS impressions, sum(b) AS buys, buys / impressions AS buy_rate
        FROM j GROUP BY relation, in_rv ORDER BY in_rv, relation""")
    run("recompra organica: horas entre compras del mismo producto", f"""
        SELECT count() AS repeat_buys, quantiles(0.1, 0.25, 0.5, 0.75, 0.9)(gap_h) AS gap_hours_q,
               countIf(gap_h < 24) / repeat_buys AS within_1d, countIf(gap_h < 168) / repeat_buys AS within_7d
        FROM (SELECT dateDiff('hour', lagInFrame(t) OVER w, t) AS gap_h, row_number() OVER w AS rn
              FROM {EVH} WHERE et = 'purchase' WINDOW w AS (PARTITION BY u, pid ORDER BY t
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)) WHERE rn > 1""")

    # ------------------------------------------------------------------ 4. rebajas
    print("\n# 4. REBAJAS (shop.events en vivo; precio del evento vs precio base del catalogo)")
    run("compras/vista mismo producto con y sin descuento", f"""
        SELECT count() AS products_with_both, sum(vn) AS views_normal, sum(vd) AS views_disc,
               sum(bn) / sum(vn) AS buy_rate_normal, sum(bd) / sum(vd) AS buy_rate_disc,
               buy_rate_disc / buy_rate_normal AS lift_buy_rate,
               (sum(vd) / sum(hd)) / (sum(vn) / sum(hn)) AS lift_views_per_hour
        FROM (SELECT e.pid,
                countIf(e.et = 'view' AND e.price >= c.price * 0.95) AS vn, countIf(e.et = 'purchase' AND e.price >= c.price * 0.95) AS bn,
                countIf(e.et = 'view' AND e.price < c.price * 0.95) AS vd, countIf(e.et = 'purchase' AND e.price < c.price * 0.95) AS bd,
                uniqExactIf(toStartOfHour(e.t), e.price >= c.price * 0.95) AS hn, uniqExactIf(toStartOfHour(e.t), e.price < c.price * 0.95) AS hd
              FROM {EVL} AS e INNER JOIN {CAT} AS c ON c.pid = e.pid WHERE c.price > 0 GROUP BY e.pid)
        WHERE vn > 0 AND vd > 0""")

    # ------------------------------------------------------------------ 5. cold start
    print("\n# 5. COLD START")
    run("productos nuevos vs existentes de su misma subcategoria (live)", f"""
        WITH allp AS (SELECT pid, sub FROM {CAT} UNION ALL SELECT pid, sub FROM {NEW_P}),
             ev AS (SELECT e.pid AS pid, e.et AS et, e.t AS t, a.sub AS sub, e.pid IN (SELECT pid FROM {NEW_P}) AS is_new
                    FROM {EVL} AS e INNER JOIN allp AS a ON a.pid = e.pid),
             subrate AS (SELECT sub, countIf(et = 'purchase') / countIf(et = 'view') AS r,
                         countIf(et = 'view') / uniqExact(pid) AS views_per_prod FROM ev WHERE NOT is_new GROUP BY sub)
        SELECT is_new, uniqExact(ev.pid) AS products, countIf(et = 'view') AS views, countIf(et = 'purchase') AS buys,
               buys / views AS buy_rate, buys / sumIf(subrate.r, et = 'view') AS buy_rate_vs_subcat,
               views / products AS views_per_product
        FROM ev INNER JOIN subrate ON subrate.sub = ev.sub GROUP BY is_new""")
    run("productos nuevos: vistas por hora desde el lanzamiento", f"""
        SELECT dateDiff('hour', n.launch, e.t) AS hours_since_launch, uniqExact(e.pid) AS products,
               countIf(e.et = 'view') / products AS views_per_product, countIf(e.et = 'purchase') / products AS buys_per_product
        FROM {EVL} AS e INNER JOIN {NEW_P} AS n ON n.pid = e.pid WHERE e.t >= n.launch
        GROUP BY hours_since_launch ORDER BY hours_since_launch LIMIT 8""")
    run("usuarios nuevos vs conocidos (live events): que compran", f"""
        WITH top AS (SELECT pid FROM {EVH} WHERE et = 'purchase' GROUP BY pid ORDER BY count() DESC LIMIT 100)
        SELECT u IN (SELECT DISTINCT u FROM {EVH}) AS known_user, uniqExact(u) AS users, countIf(et = 'purchase') AS buys,
               buys / countIf(et = 'view') AS buy_per_view, countIf(et = 'purchase' AND pid IN top) / buys AS buys_in_hist_top100,
               countIf(et = 'view') / users AS views_per_user
        FROM {EVL} GROUP BY known_user""")

    print(f"\n# FIN {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()