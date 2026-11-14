"""¿Qué señales predicen una compra en el carrusel? Análisis sobre el 10 % de shop.history_sessions.

    python analysis/signal_lift.py

Para cada producto mostrado en un carrusel antiguo (una fila por producto y posición) mira si se clicó o
compró, y lo cruza con lo que ese usuario hizo por su cuenta en shop.history_events. Solo lee.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monitor import query  # noqa: E402

BASE = """
WITH cat AS (SELECT product_id::String p, subcategory::String sub, category::String cat, brand::String brand, price_eur::Float64 price FROM shop.catalog),
ev AS (SELECT e.user_id::String u, e.product_id::String p, c.sub sub, c.cat cat, c.brand brand, e.event_type::String t FROM shop.history_events e INNER JOIN cat c ON c.p = e.product_id::String),
ua AS (SELECT u, sub, count() n FROM ev GROUP BY u, sub),
uc AS (SELECT u, cat, count() n FROM ev GROUP BY u, cat),
ub AS (SELECT u, brand, count() n FROM ev GROUP BY u, brand),
up AS (SELECT u, p, countIf(t='view') hv, countIf(t='purchase') hb FROM ev GROUP BY u, p),
ut AS (SELECT u, sum(n) tot FROM ua GROUP BY u),
pop0 AS (SELECT p, sub, countIf(t='purchase') nb FROM ev GROUP BY p, sub),
pop AS (SELECT p, row_number() OVER (PARTITION BY sub ORDER BY nb DESC) rk FROM pop0),
imp AS (
  SELECT user_id::String u, context.page::String page,
         arrayMap(x -> x.product_id::String, recent_views::Array(JSON)) rv,
         item, slot, ifNull(clicked::String = item, 0) clk, ifNull(purchased::String = item, 0) buy
  FROM shop.history_sessions
  ARRAY JOIN shown::Array(String) AS item, arrayEnumerate(shown::Array(String)) AS slot
  WHERE cityHash64(request_id::String) % 10 = 0),
x AS (
  SELECT imp.u u, imp.page page, imp.rv rv, imp.item item, imp.slot slot, imp.clk clk, imp.buy buy, ci.price price, ci.sub isub, cl.sub lsub, pop.rk rk, ifNull(ut.tot,0) tot,
         ifNull(ua.n,0)/greatest(ifNull(ut.tot,0),1) aff, ifNull(uc.n,0)/greatest(ifNull(ut.tot,0),1) affc, ifNull(ub.n,0)/greatest(ifNull(ut.tot,0),1) affb,
         ifNull(up.hv,0) pv, ifNull(up.hb,0) pb
  FROM imp
  LEFT JOIN cat ci ON ci.p = imp.item
  LEFT JOIN cat cl ON cl.p = imp.rv[-1]
  LEFT JOIN pop ON pop.p = imp.item
  LEFT JOIN ua ON ua.u = imp.u AND ua.sub = ci.sub
  LEFT JOIN uc ON uc.u = imp.u AND uc.cat = ci.cat
  LEFT JOIN ub ON ub.u = imp.u AND ub.brand = ci.brand
  LEFT JOIN up ON up.u = imp.u AND up.p = imp.item
  LEFT JOIN ut ON ut.u = imp.u)
"""
M = "count() imps, round(100*avg(clk),3) ctr, round(100*avg(buy),3) compra"
PRICE = "multiIf(price<10,'1 <10',price<20,'2 10-20',price<40,'3 20-40',price<80,'4 40-80','5 80+')"
QUERIES = {
    "afinidad del usuario por la subcategoría del producto": f"SELECT multiIf(tot=0,'0 sin historial', aff=0,'1 nunca la miró', aff<0.05,'2 <5%', aff<0.15,'3 5-15%', aff<0.3,'4 15-30%','5 >30%') g, {M} FROM x GROUP BY g ORDER BY g",
    "producto ya visto en la sesión (recent_views)": f"SELECT has(rv, item) visto, {M} FROM x WHERE length(rv)>0 GROUP BY visto",
    "misma subcategoría que la última vista": f"SELECT isub = lsub misma_sub, {M} FROM x WHERE length(rv)>0 AND NOT has(rv,item) GROUP BY misma_sub",
    "el usuario vio ESE producto en el histórico (nº veces)": f"SELECT least(pv,3) vistas, {M} FROM x GROUP BY vistas ORDER BY vistas",
    "el usuario compró ESE producto en el histórico": f"SELECT least(pb,2) compras, {M} FROM x GROUP BY compras ORDER BY compras",
    "subcat afín (>=15%) × producto visto en el histórico": f"SELECT aff>=0.15 afin, pv>0 visto, {M} FROM x GROUP BY afin, visto ORDER BY afin, visto",
    "afinidad de CATEGORÍA cuando la subcat no es afín": f"SELECT multiIf(affc<0.1,'1 <10%',affc<0.3,'2 10-30%','3 >30%') g, {M} FROM x WHERE aff<0.05 GROUP BY g ORDER BY g",
    "afinidad de MARCA cuando la subcat no es afín": f"SELECT multiIf(affb=0,'0',affb<0.03,'1 <3%','2 >3%') g, {M} FROM x WHERE aff<0.05 GROUP BY g ORDER BY g",
    "dentro de subcat afín: ranking de ventas en su subcat": f"SELECT multiIf(rk<=3,'1 top3',rk<=10,'2 top10',rk<=25,'3 top25','4 resto') g, {M} FROM x WHERE aff>=0.15 GROUP BY g ORDER BY g",
    "precio (todos)": f"SELECT {PRICE} g, {M} FROM x GROUP BY g ORDER BY g",
    "precio dentro de subcat afín": f"SELECT {PRICE} g, {M} FROM x WHERE aff>=0.15 GROUP BY g ORDER BY g",
    "posición dentro de subcat afín": f"SELECT slot, {M} FROM x WHERE aff>=0.15 GROUP BY slot ORDER BY slot",
    "por página": f"SELECT page, multiIf(tot=0,'sin hist', aff>=0.15,'afín','no afín') g, {M} FROM x GROUP BY page, g ORDER BY page, g",
}

if __name__ == "__main__":
    only = sys.argv[1:]
    for title, sel in QUERIES.items():
        if only and not any(o in title for o in only):
            continue
        t0 = time.time()
        print(f"\n## {title}")
        try:
            for r in query(BASE + sel, timeout=600):
                print(json.dumps(r, ensure_ascii=False, default=str))
        except Exception as e:
            print("ERR", str(e)[:500])
        print(f"   ({time.time() - t0:.0f}s)")
