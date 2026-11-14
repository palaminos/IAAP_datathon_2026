"""¿Importan las TENDENCIAS y los CAMBIOS DE PRECIO para lo que se compra en el carrusel? Solo lee.

    python analysis/trends_prices.py     # desde la raíz del repo

1. Cambios de precio: cuándo ocurren, cuántos productos hay rebajados ahora, y si un producto rebajado se
   compra más (por su cuenta en la tienda y cuando lo enseñamos en el carrusel).
2. Tendencias: para cada producto que enseñamos, cuánto ha subido su subcategoría en los 15 minutos
   anteriores respecto a las 3 horas previas, y si eso cambia la probabilidad de clic y de compra.
"""
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monitor import TEAM, query  # noqa: E402

BASE = "(SELECT product_id::String AS p, price_eur::Float64 AS base, subcategory::String AS sub FROM shop.catalog)"


def show(title, sql):
    print(f"\n## {title}")
    for r in query(sql, timeout=600):
        print("  ", json.dumps(r, ensure_ascii=False, default=str))


show("cambios de precio por hora (UTC) y tamaño medio del cambio", f"""
SELECT toString(toStartOfHour(u.ts::DateTime64(3))) hora, count() cambios, countIf(u.price_eur::Float64 < c.base * 0.95) rebajas,
       countIf(u.price_eur::Float64 > c.base * 1.05) subidas, round(avg(u.price_eur::Float64 / c.base), 3) precio_sobre_base
FROM shop.catalog_updates u INNER JOIN {BASE} c ON c.p = u.product_id::String
WHERE u.change::String = 'price_change' GROUP BY hora ORDER BY hora""")
show("productos rebajados AHORA (precio actual frente al de shop.catalog)", f"""
SELECT countIf(cur < base * 0.95) rebajados, countIf(cur > base * 1.05) encarecidos, count() productos
FROM (SELECT c.base base, if(u.p = '', c.base, u.cur) cur FROM {BASE} c
      LEFT JOIN (SELECT product_id::String p, argMax(price_eur::Float64, ts::DateTime64(3)) cur FROM shop.catalog_updates GROUP BY p) u ON u.p = c.p)""")
show("ORGÁNICO hoy: ¿se compra más un producto rebajado? (precio del evento frente al precio base)", f"""
SELECT multiIf(e.price_eur::Float64 < c.base * 0.95, '1 rebajado', e.price_eur::Float64 > c.base * 1.05, '3 encarecido', '2 precio normal') precio,
       countIf(e.event_type = 'view') vistas, round(100 * countIf(e.event_type = 'add_to_cart') / vistas, 2) carritos_por_100_vistas,
       round(100 * countIf(e.event_type = 'purchase') / vistas, 2) compras_por_100_vistas
FROM shop.events e INNER JOIN {BASE} c ON c.p = e.product_id::String GROUP BY precio ORDER BY precio""")
show("ORGÁNICO: los mismos productos, con y sin rebaja (solo los que han estado rebajados hoy)", f"""
SELECT multiIf(e.price_eur::Float64 < c.base * 0.95, '1 rebajado', '2 precio normal') precio, uniqExact(c.p) productos,
       countIf(e.event_type = 'view') vistas, round(100 * countIf(e.event_type = 'purchase') / vistas, 2) compras_por_100_vistas
FROM shop.events e INNER JOIN {BASE} c ON c.p = e.product_id::String
WHERE c.p IN (SELECT DISTINCT e2.product_id::String FROM shop.events e2 INNER JOIN {BASE} c2 ON c2.p = e2.product_id::String WHERE e2.price_eur::Float64 < c2.base * 0.95)
GROUP BY precio ORDER BY precio""")
show("CARRUSEL: productos rebajados en el momento de enseñarlos (todas las respuestas, también fallback)", f"""
SELECT multiIf(pos = 1, '1', pos <= 3, '2-3', '4-10') puesto, disc AS rebajado, count() impresiones,
       round(100 * countIf(c) / impresiones, 3) ctr, round(100 * countIf(b) / impresiones, 3) compra, countIf(b) compras
FROM (SELECT imp.pos pos, imp.c c, imp.b b, (u.price > 0 AND u.price < base.base * 0.95) AS disc
      FROM (SELECT r.ts::DateTime64(3) ts, item, pos, f.clicked::String = item AS c, f.purchased::String = item AS b
            FROM {TEAM}.feedback f INNER JOIN shop.requests r ON r.request_id::String = f.request_id::String
            ARRAY JOIN f.shown::Array(String) AS item, arrayEnumerate(f.shown::Array(String)) AS pos) imp
      ASOF LEFT JOIN (SELECT product_id::String p, ts::DateTime64(3) ts, price_eur::Float64 price FROM shop.catalog_updates WHERE change::String = 'price_change') u
        ON u.p = imp.item AND u.ts <= imp.ts
      INNER JOIN {BASE} base ON base.p = imp.item)
GROUP BY puesto, rebajado ORDER BY puesto, rebajado""")

# ------------------------------------------------------------------ tendencias por subcategoría
print("\n## TENDENCIAS: subida de la subcategoría en los 15 min previos frente a las 3 h anteriores")
sub_of = {r["p"]: r["sub"] for r in query("""SELECT p, argMax(sub, v) AS sub FROM (
    SELECT product_id::String AS p, subcategory::String AS sub, '' AS v FROM shop.catalog
    UNION ALL SELECT product_id::String, subcategory::String, toString(ts::DateTime64(3)) FROM shop.catalog_updates) GROUP BY p""")}
subs = sorted(set(sub_of.values()))
si = {s: i for i, s in enumerate(subs)}
rows = query("""SELECT toUnixTimestamp(toStartOfFiveMinutes(ts::DateTime64(3))) b, product_id::String p, countIf(event_type = 'view') v
FROM shop.events GROUP BY b, p HAVING v > 0""", timeout=600)
b0 = min(int(r["b"]) for r in rows)
nb = (max(int(r["b"]) for r in rows) - b0) // 300 + 1
V = np.zeros((nb, len(subs)))
for r in rows:
    if r["p"] in sub_of:
        V[(int(r["b"]) - b0) // 300, si[sub_of[r["p"]]]] += float(r["v"])
C = np.vstack([np.zeros((1, len(subs))), np.cumsum(V, 0)])   # C[k] = suma de los k primeros tramos de 5 min
fb = [r for i in range(4) for r in query(f"""
SELECT toUnixTimestamp(r.ts::DateTime64(3)) t, f.shown::Array(String) shown, f.clicked::String clicked, f.purchased::String purchased
FROM {TEAM}.feedback f INNER JOIN shop.requests r ON r.request_id::String = f.request_id::String
WHERE f.served_by::String = 'team' AND cityHash64(f.request_id::String) % 4 = {i}""", timeout=600)]
tab = defaultdict(lambda: np.zeros(3))
for r in fb:
    k = (int(r["t"]) - b0) // 300          # tramo en curso; se usan solo tramos ya terminados
    if k < 39 or k > nb:
        continue
    recent, before = C[k] - C[k - 3], C[k - 3] - C[k - 39]
    lift = (recent / recent.sum()) / np.maximum(before / before.sum(), 1e-9)
    for pos, p in enumerate(r["shown"]):
        s = sub_of.get(p)
        if s is None:
            continue
        x = lift[si[s]]
        g = "1 baja (<0,8)" if x < 0.8 else "2 normal (0,8-1,2)" if x < 1.2 else "3 sube (1,2-1,5)" if x < 1.5 else "4 sube mucho (>1,5)"
        slot = "puesto 1" if pos == 0 else "puestos 2-3" if pos < 3 else "puestos 4-10"
        for key in ((slot, g), ("todos los puestos", g)):
            tab[key] += (1, p == r["clicked"], p == r["purchased"])
last = None
for (slot, g), (n, c, b) in sorted(tab.items()):
    if slot != last:
        print(f"  {slot}:"); last = slot
    print(f"     subcategoría {g:<22} impresiones={int(n):>7}  clic={100 * c / n:6.3f} %  compra={100 * b / n:6.3f} %  (compras={int(b)})")
lifts = []
for k in range(39, nb):
    recent, before = C[k] - C[k - 3], C[k - 3] - C[k - 39]
    lifts.append((recent / recent.sum()) / np.maximum(before / before.sum(), 1e-9))
L = np.array(lifts)
print(f"\n  cuánto se mueven las subcategorías: en un tramo cualquiera, el {100 * (L > 1.5).mean():.1f} % de las subcategorías sube más de un 50 % "
      f"y el {100 * (L > 2).mean():.1f} % se duplica; máximo visto: x{L.max():.1f}")

# ------------------------------------------------------------------ tendencias por producto
print("\n## TENDENCIAS POR PRODUCTO: cuota de vistas en la última hora frente a su cuota en los 14 días del histórico")
hist = {r["p"]: float(r["v"]) for r in query("""SELECT product_id::String p, countIf(event_type = 'view') v FROM shop.history_events
WHERE ts::DateTime64(3) > (SELECT max(ts::DateTime64(3)) FROM shop.history_events) - INTERVAL 14 DAY GROUP BY p""", timeout=600)}
prods = sorted(hist)
pi = {p: i for i, p in enumerate(prods)}
H = np.array([hist[p] for p in prods])
H = H / H.sum()
P = np.zeros((nb, len(prods)), np.float32)
for r in rows:
    if r["p"] in pi:
        P[(int(r["b"]) - b0) // 300, pi[r["p"]]] += float(r["v"])
CP = np.vstack([np.zeros((1, len(prods)), np.float32), np.cumsum(P, 0)])
tab = defaultdict(lambda: np.zeros(3))
for r in fb:
    k = (int(r["t"]) - b0) // 300
    if k < 12 or k > nb:
        continue
    recent = CP[k] - CP[k - 12]
    tot = float(recent.sum())
    for pos, p in enumerate(r["shown"]):
        j = pi.get(p)
        if j is None:
            continue   # producto nuevo: sin histórico
        x = (recent[j] + 1) / (tot * H[j] + 1)      # suavizado con 1 vista
        g = "1 muy por debajo (<0,6)" if x < 0.6 else "2 por debajo (0,6-0,85)" if x < 0.85 else "3 normal (0,85-1,2)" if x < 1.2 else "4 por encima (1,2-1,6)" if x < 1.6 else "5 muy por encima (>1,6)"
        slot = "puesto 1" if pos == 0 else "puestos 2-3" if pos < 3 else "puestos 4-10"
        for key in ((slot, g), ("todos los puestos", g)):
            tab[key] += (1, p == r["clicked"], p == r["purchased"])
last = None
for (slot, g), (n, c, b) in sorted(tab.items()):
    if slot != last:
        print(f"  {slot}:"); last = slot
    print(f"     producto {g:<26} impresiones={int(n):>7}  clic={100 * c / n:6.3f} %  compra={100 * b / n:6.3f} %  (compras={int(b)})")
