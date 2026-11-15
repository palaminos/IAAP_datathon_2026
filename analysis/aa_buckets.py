"""¿Son comparables los dos grupos de compradores del reparto A/B? Prueba A/A con los compradores NUEVOS.

    python analysis/aa_buckets.py ['2026-11-15 03:31:00']     # hora UTC desde la que mirar (por defecto, la ronda actual)

El reparto es por md5(user_id) % 100: la primera versión de VARIANTS se queda los grupos bajos. En la ronda 13,
v24 (grupos 0-49) y v33 (50-99) dan EXACTAMENTE las mismas listas a los compradores nuevos, así que cualquier
diferencia entre grupos con ellos es composición o suerte, no modelo. Se mira:
  * clics y compras por tramos de grupo (0-24, 25-49, 50-74, 75-99);
  * cuánto se concentran clics y compras en pocos compradores (si un comprador repite mucho, las requests no son
    independientes y el margen de error real es mayor que el que sale de contar requests).
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import recommender as rec  # noqa: E402

since = sys.argv[1] if len(sys.argv) > 1 else "2026-11-15 03:31:00"
rows = rec.query(f"""
SELECT r.user_id::String AS u, any(r.is_new_user::Bool) AS is_new, count() AS n, countIf(f.clicked::String != '') AS clicks,
       countIf(f.purchased::String != '') AS buys, groupUniqArray(f.model_version::String) AS vs
FROM {rec.TEAM}.feedback AS f INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
WHERE f.ts::DateTime64(3) >= toDateTime64('{since}', 3) AND f.served_by::String = 'team'
GROUP BY u""", timeout=300)
b = np.array([int(hashlib.md5(r["u"].encode()).hexdigest(), 16) % 100 for r in rows])
new = np.array([str(r["is_new"]).lower() in ("true", "1") for r in rows])
n, c, y = (np.array([int(r[k]) for r in rows], float) for k in ("n", "clicks", "buys"))
print(f"desde {since} UTC: {len(rows)} compradores, {int(n.sum())} respuestas | versiones: {sorted({v for r in rows for v in r['vs']})}")
for label, seg in (("NUEVOS", new), ("CONOCIDOS", ~new)):
    print(f"\n{label}: {int(seg.sum())} compradores, {int(n[seg].sum())} respuestas, {n[seg].sum() / seg.sum():.1f} respuestas por comprador (máx. {int(n[seg].max())})")
    for lo, hi in ((0, 25), (25, 50), (50, 75), (75, 100), (0, 50), (50, 100)):
        k = seg & (b >= lo) & (b < hi)
        print(f"   grupos {lo:>2}-{hi - 1:<3} compradores {int(k.sum()):>5} | respuestas {int(n[k].sum()):>6} | CTR {100 * c[k].sum() / n[k].sum():.2f} % | conversión {100 * y[k].sum() / n[k].sum():.2f} % ({int(y[k].sum())} compras)")
    # sobredispersión: varianza real de la tasa por grupo (100 grupos) frente a la que daría contar requests independientes
    for name, v in (("clics", c), ("compras", y)):
        p = v[seg].sum() / n[seg].sum()
        nb = np.bincount(b[seg], weights=n[seg], minlength=100)
        vb = np.bincount(b[seg], weights=v[seg], minlength=100)
        z = (vb - p * nb) / np.sqrt(np.maximum(p * (1 - p) * nb, 1e-9))
        print(f"   {name}: dispersión entre los 100 grupos = {z.var():.2f} veces la esperada si las requests fueran independientes (1,0 = lo esperado)")
    top = np.sort(y[seg])[::-1]
    k_users = max(int(0.01 * seg.sum()), 1)
    print(f"   el 1 % de compradores con más compras ({k_users}) se lleva el {100 * top[:k_users].sum() / max(y[seg].sum(), 1):.0f} % de las compras; con 2 o más compras: {int((y[seg] >= 2).sum())} compradores")
    # comparación entre mitades con el error calculado POR COMPRADOR (no por request)
    a, bb = seg & (b < 50), seg & (b >= 50)
    for name, v in (("clics", c), ("compras", y)):
        ra, rb = v[a].sum() / n[a].sum(), v[bb].sum() / n[bb].sum()
        # error por comprador (linealización de la razón)
        ea = np.sqrt(((v[a] - ra * n[a]) ** 2).sum()) / n[a].sum()
        eb = np.sqrt(((v[bb] - rb * n[bb]) ** 2).sum()) / n[bb].sum()
        naive = np.sqrt(ra * (1 - ra) / n[a].sum() + rb * (1 - rb) / n[bb].sum())
        print(f"   {name}: grupos 50-99 menos 0-49 = {100 * (rb - ra):+.2f} puntos | error contando requests {100 * naive:.2f} (z = {(rb - ra) / naive:+.1f}) | error por comprador {100 * np.hypot(ea, eb):.2f} (z = {(rb - ra) / np.hypot(ea, eb):+.1f})")
