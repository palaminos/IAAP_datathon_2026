"""EDA 01 · diagnóstico numérico orientado a modelado (sin gráficas, sin descargar datos).

    python notebooks/eda_01.py            # desde la raíz del repo (lee .env)
    python notebooks/eda_01.py > eda_01.txt

Todo se agrega dentro de RawTree (SQL): solo viajan resúmenes. Salida compacta pensada para pegarla en el chat.
Cada bloque es independiente: si uno falla imprime el error en una línea y sigue.
Tiempo esperado: 2-6 min (history_events tiene ~12 M filas).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests

# --------------------------------------------------------------------------- conexión
for d in [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent.parent]:
    if (d / ".env").exists():
        for line in (d / ".env").read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() and not k.strip().startswith("#"):
                os.environ.setdefault(k.strip(), v.strip())
        break
API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ.get("TEAM_ID", "team_23")
http = requests.Session()
http.headers["Authorization"] = f"Bearer {os.environ.get('RAWTREE_API_KEY', '')}"
MAX_ROWS = 40


def query(sql: str) -> list[dict]:
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=900)
    if not r.ok:
        raise RuntimeError(f"{r.status_code} {r.text[:200]}")
    return r.json()["data"]


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, list):
        return "[" + ",".join(fmt(x) for x in v) + "]"
    if isinstance(v, str):
        try:  # RawTree devuelve a veces números como texto
            f = float(v)
            return v if f.is_integer() and "." not in v else f"{f:.4g}"
        except ValueError:
            return v[:40]
    return str(v)


def run(name: str, sql: str) -> list[dict] | None:
    t = time.time()
    try:
        rows = query(sql)
    except Exception as e:
        print(f"## {name} | ERROR {str(e)[:220]}")
        return None
    print(f"## {name} ({time.time() - t:.0f}s, {len(rows)} filas)")
    if rows:
        cols = list(rows[0])
        print("|".join(cols))
        for r in rows[:MAX_ROWS]:
            print("|".join(fmt(r[c]) for c in cols))
        if len(rows) > MAX_ROWS:
            print(f"... (+{len(rows) - MAX_ROWS})")
    return rows


def nz(col: str) -> str:
    """Dynamic que puede ser NULL / '' / 'null' -> '' o el string."""
    return f"if(isNull({col}) OR toString({col}) IN ('', 'null', 'None', '\\\\N'), '', toString({col}))"


# --------------------------------------------------------------------------- vistas base (subconsultas)
CAT = """(SELECT toString(product_id) AS pid, toString(category) AS cat, toString(subcategory) AS sub,
        toString(brand) AS brand, toFloat64OrNull(toString(price_eur)) AS price,
        toString(is_active) IN ('true', '1') AS active, toString(description) AS descr,
        parseDateTime64BestEffortOrNull(toString(created_at), 3) AS created
    FROM shop.catalog)"""


def rv_cols() -> str:
    return """ifNull(toJSONString(recent_views), '[]') AS rv,
        JSONLength(rv) AS n_rv,
        arrayMap(x -> JSONExtractString(x, 'product_id'), JSONExtractArrayRaw(rv)) AS rv_ids,
        JSONExtractString(rv, -1, 'product_id') AS last_view,
        parseDateTime64BestEffortOrNull(JSONExtractString(rv, -1, 'ts'), 3) AS last_view_ts,
        JSONExtractString(ifNull(toJSONString(context), '{}'), 'device') AS device,
        JSONExtractString(ifNull(toJSONString(context), '{}'), 'region') AS region,
        JSONExtractString(ifNull(toJSONString(context), '{}'), 'page') AS page"""


S = f"""(SELECT toString(request_id) AS rid, ts::DateTime64(3) AS t, toString(user_id) AS u, {rv_cols()},
        JSONExtract(ifNull(toJSONString(shown), '[]'), 'Array(String)') AS shown_a,
        JSONExtract(ifNull(toJSONString(propensity), '[]'), 'Array(Float64)') AS prop_a,
        {nz('clicked')} AS clk, {nz('purchased')} AS pur,
        toFloat64OrZero(toString(revenue_eur)) AS rev
    FROM shop.history_sessions)"""

R = f"""(SELECT toString(request_id) AS rid, ts::DateTime64(3) AS t, toString(user_id) AS u,
        toString(is_new_user) IN ('true', '1') AS is_new, {rv_cols()}
    FROM shop.requests)"""


def EV(tbl: str) -> str:
    return f"""(SELECT ts::DateTime64(3) AS t, toString(event_type) AS et, toString(user_id) AS u,
        toString(session_id) AS s, toString(product_id) AS pid, toFloat64OrNull(toString(price_eur)) AS price,
        toString(device) AS device, toString(region) AS region
    FROM {tbl})"""


EVH, EVL = EV("shop.history_events"), EV("shop.events")
S_MAX = "(SELECT max(ts::DateTime64(3)) FROM shop.history_sessions)"
NEW_PIDS = """(SELECT groupUniqArray(toString(product_id)) FROM shop.catalog_updates
              WHERE toString(change) = 'new_product')"""


def main() -> None:
    t0 = time.time()
    print(f"# EDA_01 team={TEAM} api={API}")

    # ------------------------------------------------------------------ 0. inventario
    print("\n# 0. INVENTARIO")
    for tbl in ["shop.catalog", "shop.history_sessions", "shop.history_events", "shop.requests",
                "shop.events", "shop.catalog_updates", f"{TEAM}.feedback", f"{TEAM}.submissions"]:
        ts = "created_at" if tbl == "shop.catalog" else "ts"
        run(f"tabla {tbl}", f"""SELECT count() AS n, min({ts}::DateTime64(3)) AS min_ts,
            max({ts}::DateTime64(3)) AS max_ts, dateDiff('second', max({ts}::DateTime64(3)), now()) AS lag_s
            FROM {tbl}""")
    run("tipos dinámicos history_sessions", """SELECT dynamicType(clicked) AS clicked_t, dynamicType(purchased) AS pur_t,
        dynamicType(shown) AS shown_t, dynamicType(propensity) AS prop_t, dynamicType(recent_views) AS rv_t,
        dynamicType(context) AS ctx_t, count() AS n FROM shop.history_sessions GROUP BY ALL ORDER BY n DESC LIMIT 6""")

    # ------------------------------------------------------------------ 1. catálogo
    print("\n# 1. CATALOGO")
    run("catalogo global", f"""SELECT count() AS n, uniqExact(pid) AS uniq_pid, uniqExact(cat) AS cats,
        uniqExact(sub) AS subcats, uniqExact(brand) AS brands, avg(active) AS active_rate,
        countIf(price IS NULL OR price <= 0) AS bad_price, countIf(descr = '') AS empty_descr,
        avg(length(descr)) AS descr_len, quantiles(0.05, 0.25, 0.5, 0.75, 0.95)(price) AS price_q,
        countIf(created > (SELECT max(created) FROM {CAT}) - INTERVAL 60 DAY) AS created_last60d
        FROM {CAT}""")
    run("catalogo por categoria", f"""SELECT cat, count() AS n, uniqExact(sub) AS subs, uniqExact(brand) AS brands,
        median(price) AS p50, avg(active) AS active FROM {CAT} GROUP BY cat ORDER BY n DESC""")

    # ------------------------------------------------------------------ 2. target (history_sessions)
    print("\n# 2. TARGET history_sessions")
    run("target global", f"""SELECT count() AS n, uniqExact(rid) AS uniq_rid, uniqExact(u) AS users,
        countIf(clk != '') / n AS ctr, countIf(pur != '') / n AS conv,
        countIf(pur != '') / countIf(clk != '') AS conv_given_click,
        countIf(pur != '' AND clk = '') AS pur_without_click,
        countIf(pur != '' AND clk != '' AND pur != clk) AS pur_ne_click,
        countIf(pur != '' AND indexOf(shown_a, pur) = 0) AS pur_not_shown,
        countIf(clk != '' AND indexOf(shown_a, clk) = 0) AS clk_not_shown,
        sum(rev) / n AS eur_req, sum(rev) / countIf(pur != '') AS eur_per_buy,
        countIf(pur = '' AND rev > 0) AS rev_without_buy FROM {S}""")
    run("calidad sessions", f"""SELECT countIf(length(shown_a) != 10) AS shown_not10,
        countIf(length(arrayDistinct(shown_a)) != length(shown_a)) AS shown_dups,
        countIf(length(prop_a) != length(shown_a)) AS prop_len_mismatch,
        countIf(arrayExists(x -> x <= 0 OR x > 1, prop_a)) AS prop_out_0_1,
        quantiles(0.001, 0.01, 0.5)(arrayMin(prop_a)) AS min_prop_q,
        countIf(n_rv = 0) / count() AS empty_rv, quantiles(0.5, 0.9, 0.99)(n_rv) AS n_rv_q,
        countIf(last_view_ts > t) AS rv_after_request_LEAK,
        quantiles(0.5, 0.9)(dateDiff('second', last_view_ts, t)) AS sec_since_last_view_q,
        countIf(arrayExists(p -> p NOT IN (SELECT pid FROM {CAT}), shown_a)) AS shown_unknown_pid FROM {S}""")
    run("posicion (slot): sesgo y conversion", f"""SELECT i AS slot, avg(prop_a[i]) AS prop_mean,
        min(prop_a[i]) AS prop_min, countIf(clk = shown_a[i]) / count() AS ctr,
        countIf(pur = shown_a[i]) / count() AS conv FROM {S} ARRAY JOIN arrayEnumerate(shown_a) AS i
        GROUP BY slot ORDER BY slot""")
    for dim in ["page", "device", "region"]:
        run(f"target por {dim}", f"""SELECT {dim}, count() AS n, n / sum(n) OVER () AS share,
            countIf(clk != '') / n AS ctr, countIf(pur != '') / n AS conv, sum(rev) / n AS eur_req,
            avg(n_rv) AS avg_rv FROM {S} GROUP BY {dim} ORDER BY n DESC""")
    run("target por n_recent_views", f"""SELECT least(n_rv, 8) AS n_rv_cap, count() AS n, n / sum(n) OVER () AS share,
        countIf(pur != '') / n AS conv, countIf(pur != '' AND has(rv_ids, pur)) / countIf(pur != '') AS buy_in_rv
        FROM {S} GROUP BY n_rv_cap ORDER BY n_rv_cap""")
    run("target por semana", f"""SELECT toMonday(t) AS week, count() AS n, countIf(clk != '') / n AS ctr,
        countIf(pur != '') / n AS conv, sum(rev) / n AS eur_req FROM {S} GROUP BY week ORDER BY week""")
    run("target por hora UTC (final = 09-11 UTC)", f"""SELECT toHour(t) AS h, count() AS n,
        countIf(pur != '') / n AS conv FROM {S} GROUP BY h ORDER BY h""")
    run("split temporal: ultimos 7d vs previo", f"""SELECT if(t >= {S_MAX} - INTERVAL 7 DAY, 'test_7d', 'train') AS split,
        count() AS n, uniqExact(u) AS users, countIf(clk != '') / n AS ctr, countIf(pur != '') / n AS conv,
        sum(rev) / n AS eur_req, countIf(n_rv = 0) / n AS empty_rv FROM {S} GROUP BY split""")
    run("usuarios en test_7d ya vistos en train", f"""SELECT count() AS test_users,
        countIf(u IN (SELECT u FROM {S} WHERE t < {S_MAX} - INTERVAL 7 DAY)) / test_users AS seen_in_train_sessions,
        countIf(u IN (SELECT DISTINCT u FROM {EVH} WHERE t < {S_MAX} - INTERVAL 7 DAY)) / test_users AS seen_in_train_events
        FROM (SELECT DISTINCT u FROM {S} WHERE t >= {S_MAX} - INTERVAL 7 DAY)""")

    # ------------------------------------------------------------------ 3. relación compra ↔ sesión / usuario
    print("\n# 3. SEÑALES: compra vs vistas, categoría, usuario")
    run("compra vs ultima vista", f"""SELECT count() AS buys_with_rv, countIf(has(s.rv_ids, s.pur)) / buys_with_rv AS in_rv,
        countIf(s.pur = s.last_view) / buys_with_rv AS eq_last_view,
        countIf(cl.sub != '' AND cp.sub = cl.sub) / buys_with_rv AS same_sub_last,
        countIf(cl.cat != '' AND cp.cat = cl.cat) / buys_with_rv AS same_cat_last,
        countIf(cl.brand != '' AND cp.brand = cl.brand) / buys_with_rv AS same_brand_last
        FROM (SELECT * FROM {S} WHERE pur != '' AND n_rv > 0) AS s
        LEFT JOIN {CAT} AS cp ON cp.pid = s.pur LEFT JOIN {CAT} AS cl ON cl.pid = s.last_view""")
    run("base: shown del sistema viejo en misma subcat que ultima vista", f"""SELECT
        avg(arrayCount(p -> p = s.last_view, s.shown_a)) AS last_view_in_shown,
        count() AS n FROM (SELECT * FROM {S} WHERE n_rv > 0) AS s""")
    run("afinidad: compra en la categoria top del usuario (eventos organicos)", f"""
        WITH ucat AS (SELECT u, argMax(cat, c) AS topcat, sum(c) AS n_ev FROM
            (SELECT e.u AS u, c.cat AS cat, count() AS c FROM {EVH} AS e INNER JOIN {CAT} AS c ON c.pid = e.pid
             WHERE e.et IN ('view', 'add_to_cart', 'purchase') GROUP BY u, cat) GROUP BY u)
        SELECT count() AS buys_known_user, countIf(cp.cat = ucat.topcat) / buys_known_user AS buy_in_topcat,
               median(ucat.n_ev) AS median_user_events
        FROM (SELECT u, pur FROM {S} WHERE pur != '') AS s INNER JOIN ucat ON ucat.u = s.u
        LEFT JOIN {CAT} AS cp ON cp.pid = s.pur""")
    run("recompra: compra carrusel ya comprada antes organicamente", f"""SELECT count() AS buys,
        countIf((u, pur) IN (SELECT u, pid FROM {EVH} WHERE et = 'purchase')) / buys AS also_bought_organic,
        countIf((u, pur) IN (SELECT u, pid FROM {EVH} WHERE et = 'view')) / buys AS viewed_organic
        FROM {S} WHERE pur != ''""")
    run("sesiones por usuario", f"""SELECT count() AS users, quantiles(0.5, 0.9, 0.99)(c) AS sess_q, max(c) AS max_sess,
        countIf(c > 1) / users AS share_repeat, countIf(b > 0) / users AS share_buyers
        FROM (SELECT u, count() AS c, countIf(pur != '') AS b FROM {S} GROUP BY u)""")
    run("concentracion compras (sessions vs events) y exposicion", f"""SELECT 'sessions_buys' AS src, length(a) AS products,
            arraySum(arraySlice(a, 1, 10)) / arraySum(a) AS top10, arraySum(arraySlice(a, 1, 100)) / arraySum(a) AS top100,
            arraySum(arraySlice(a, 1, 500)) / arraySum(a) AS top500
        FROM (SELECT arrayReverseSort(groupArray(c)) AS a FROM (SELECT pur, count() AS c FROM {S} WHERE pur != '' GROUP BY pur))
        UNION ALL SELECT 'events_buys', length(a), arraySum(arraySlice(a, 1, 10)) / arraySum(a),
            arraySum(arraySlice(a, 1, 100)) / arraySum(a), arraySum(arraySlice(a, 1, 500)) / arraySum(a)
        FROM (SELECT arrayReverseSort(groupArray(c)) AS a FROM (SELECT pid, count() AS c FROM {EVH} WHERE et = 'purchase' GROUP BY pid))
        UNION ALL SELECT 'sessions_shown', length(a), arraySum(arraySlice(a, 1, 10)) / arraySum(a),
            arraySum(arraySlice(a, 1, 100)) / arraySum(a), arraySum(arraySlice(a, 1, 500)) / arraySum(a)
        FROM (SELECT arrayReverseSort(groupArray(c)) AS a FROM (SELECT p, count() AS c FROM {S} ARRAY JOIN shown_a AS p GROUP BY p))""")
    run("conversion carrusel por producto mostrado: dispersion", f"""SELECT count() AS products_shown_100plus,
        quantiles(0.1, 0.5, 0.9, 0.99)(cv) AS conv_per_impr_q, corr(log(imp), cv) AS corr_logexpo_conv
        FROM (SELECT p, count() AS imp, countIf(pur = p) / imp AS cv FROM {S} ARRAY JOIN shown_a AS p
              GROUP BY p HAVING imp >= 100)""")

    # ------------------------------------------------------------------ 4. eventos orgánicos
    print("\n# 4. history_events")
    run("embudo por tipo", f"""SELECT et, count() AS n, uniqExact(u) AS users, uniqExact(s) AS sessions,
        uniqExact(pid) AS products FROM {EVH} GROUP BY et ORDER BY n DESC""")
    run("eventos por sesion y compras", f"""SELECT count() AS sessions, quantiles(0.5, 0.9, 0.99)(n) AS ev_q,
        countIf(b > 0) / sessions AS sess_with_buy, avg(b) AS buys_per_sess FROM
        (SELECT s, count() AS n, countIf(et = 'purchase') AS b FROM {EVH} GROUP BY s)""")
    run("compra precedida de vista/carrito en misma sesion", f"""SELECT count() AS buys,
        countIf((s, pid) IN (SELECT s, pid FROM {EVH} WHERE et = 'view')) / buys AS viewed_same_sess,
        countIf((s, pid) IN (SELECT s, pid FROM {EVH} WHERE et = 'add_to_cart')) / buys AS carted_same_sess
        FROM {EVH} WHERE et = 'purchase'""")
    run("recompra organica (mismo user y producto)", f"""SELECT count() AS user_prod_pairs,
        countIf(c > 1) / user_prod_pairs AS repeat_share, max(c) AS max_c
        FROM (SELECT u, pid, count() AS c FROM {EVH} WHERE et = 'purchase' GROUP BY u, pid)""")
    run("conversion organica por precio", f"""SELECT multiIf(c.price < 10, 'a<10', c.price < 25, 'b10-25',
        c.price < 50, 'c25-50', c.price < 100, 'd50-100', c.price < 250, 'e100-250', 'f250+') AS price_b,
        countIf(e.et = 'view') AS views, countIf(e.et = 'purchase') AS buys, buys / views AS buy_per_view
        FROM {EVH} AS e INNER JOIN {CAT} AS c ON c.pid = e.pid GROUP BY price_b ORDER BY price_b""")
    run("conversion organica por device", f"""SELECT device, count() AS n, countIf(et = 'purchase') / countIf(et = 'view') AS buy_per_view
        FROM {EVH} GROUP BY device ORDER BY n DESC""")
    run("precio evento vs catalogo", f"""SELECT count() AS n, quantiles(0.01, 0.5, 0.99)(e.price / c.price) AS ratio_q,
        countIf(abs(e.price / c.price - 1) > 0.01) / n AS share_diff FROM {EVH} AS e
        INNER JOIN {CAT} AS c ON c.pid = e.pid WHERE c.price > 0""")
    run("cobertura catalogo", f"""SELECT count() AS catalog,
        countIf(pid IN (SELECT pid FROM {EVH} WHERE et = 'view')) / catalog AS ever_viewed,
        countIf(pid IN (SELECT pid FROM {EVH} WHERE et = 'purchase')) / catalog AS ever_bought,
        countIf(pid IN (SELECT pur FROM {S} WHERE pur != '')) / catalog AS ever_bought_carousel FROM {CAT}""")

    # ------------------------------------------------------------------ 5. drift histórico vs live
    print("\n# 5. DRIFT historico vs live")
    run("live requests: cold start y calidad", f"""SELECT count() AS n, uniqExact(rid) / n AS uniq_rid_ratio,
        countIf(is_new) / n AS new_user, countIf(hasAny(rv_ids, {NEW_PIDS})) / n AS rv_has_new_product,
        countIf(is_new OR hasAny(rv_ids, {NEW_PIDS})) / n AS COLD_share,
        countIf(n_rv = 0) / n AS empty_rv, avg(n_rv) AS avg_rv, countIf(last_view_ts > t) AS rv_after_ts,
        countIf(u IN (SELECT DISTINCT u FROM {S}) OR u IN (SELECT DISTINCT u FROM {EVH})) / n AS user_in_history,
        countIf(is_new AND u IN (SELECT DISTINCT u FROM {EVH})) AS new_but_in_history,
        countIf(NOT is_new AND u NOT IN (SELECT DISTINCT u FROM {EVH}) AND u NOT IN (SELECT DISTINCT u FROM {S})) / n AS old_but_unseen
        FROM {R}""")
    run("live requests: rate por hora", f"""SELECT toStartOfHour(t) AS h, count() AS n, countIf(is_new) / n AS new_user,
        countIf(n_rv = 0) / n AS empty_rv FROM {R} GROUP BY h ORDER BY h DESC LIMIT 12""")
    for dim in ["page", "device", "region"]:
        run(f"mix {dim}: historico vs live", f"""SELECT {dim}, sumIf(n, src = 'hist') / sum(sumIf(n, src = 'hist')) OVER () AS hist,
            sumIf(n, src = 'live') / sum(sumIf(n, src = 'live')) OVER () AS live FROM
            (SELECT 'hist' AS src, {dim}, count() AS n FROM {S} GROUP BY {dim}
             UNION ALL SELECT 'live', {dim}, count() FROM {R} GROUP BY {dim})
            GROUP BY {dim} ORDER BY hist DESC""")
    run("mix compras por categoria: train / ultimos 7d / live", f"""SELECT c.cat AS cat,
        countIf(per = 'a_train') / any(tot_a) AS train, countIf(per = 'b_last7') / any(tot_b) AS last7,
        countIf(per = 'c_live') / any(tot_c) AS live FROM
        (SELECT pid, multiIf(src = 'live', 'c_live', t >= mx - INTERVAL 7 DAY, 'b_last7', 'a_train') AS per,
                countIf(per = 'a_train') OVER () AS tot_a, countIf(per = 'b_last7') OVER () AS tot_b,
                countIf(per = 'c_live') OVER () AS tot_c FROM
           (SELECT 'hist' AS src, t, pid, (SELECT max(t) FROM {EVH}) AS mx FROM {EVH} WHERE et = 'purchase'
            UNION ALL SELECT 'live', t, pid, now() FROM {EVL} WHERE et = 'purchase')) AS e
        INNER JOIN {CAT} AS c ON c.pid = e.pid GROUP BY cat ORDER BY train DESC""")
    run("live events: volumen", f"""SELECT et, count() AS n, uniqExact(u) AS users, uniqExact(pid) AS products,
        countIf(pid NOT IN (SELECT pid FROM {CAT})) AS pid_not_in_catalog,
        countIf(u IN (SELECT u FROM {R})) / n AS from_request_users FROM {EVL} GROUP BY et""")
    run("catalog_updates por tipo", """SELECT toString(change) AS change, count() AS n, uniqExact(toString(product_id)) AS products,
        min(ts::DateTime64(3)) AS first, max(ts::DateTime64(3)) AS last FROM shop.catalog_updates GROUP BY change""")
    run("cambios de precio: ratio nuevo/original", f"""SELECT count() AS n, quantiles(0.05, 0.25, 0.5, 0.75, 0.95)(toFloat64OrNull(toString(u.price_eur)) / c.price) AS ratio_q
        FROM shop.catalog_updates AS u INNER JOIN {CAT} AS c ON c.pid = toString(u.product_id)
        WHERE toString(u.change) = 'price_change' AND c.price > 0""")
    run("productos nuevos: categoria y parecido al catalogo", f"""SELECT toString(category) AS cat, count() AS n,
        countIf(toString(subcategory) IN (SELECT sub FROM {CAT})) / n AS known_sub,
        countIf(toString(brand) IN (SELECT brand FROM {CAT})) / n AS known_brand,
        median(toFloat64OrNull(toString(price_eur))) AS p50 FROM shop.catalog_updates
        WHERE toString(change) = 'new_product' GROUP BY cat ORDER BY n DESC""")

    # ------------------------------------------------------------------ 6. nuestro feedback
    print("\n# 6. FEEDBACK propio")
    run("feedback por served_by/motivo", f"""SELECT toString(served_by) AS served_by, toString(fallback_reason) AS reason,
        count() AS n, countIf({nz('clicked')} != '') / n AS ctr,
        countIf({nz('purchased')} != '') / n AS conv, sum(toFloat64OrZero(toString(revenue_eur))) / n AS eur_req,
        quantile(0.95)(toFloat64OrZero(toString(latency_ms))) AS p95_ms
        FROM {TEAM}.feedback WHERE toString(request_id) != '__init__' GROUP BY served_by, reason""")

    print(f"\n# FIN {time.time() - t0:.0f}s")


if __name__ == "__main__":
    if not os.environ.get("RAWTREE_API_KEY"):
        sys.exit("Falta RAWTREE_API_KEY (.env)")
    main()