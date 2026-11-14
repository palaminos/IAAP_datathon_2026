"""¿Se comportan EN VIVO las señales igual que en el histórico? Usa nuestro feedback real de hoy.

    python analysis/live_lift.py     # desde la raíz del repo. Solo lee.

Cada producto mostrado en una respuesta nuestra es una fila: sus señales (historial del usuario de 60 días)
y si se clicó o compró.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model  # noqa: E402
import recommender as rec  # noqa: E402

ROWS = """
SELECT r.user_id::String AS u, r.is_new_user::Bool AS nuevo, r.context.page::String AS page,
       arrayMap(x -> x.product_id::String, r.recent_views::Array(JSON)) AS rv,
       f.shown::Array(String) AS shown, f.clicked::String AS clicked, f.purchased::String AS purchased
FROM {team}.feedback AS f INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
WHERE f.served_by::String = 'team' AND cityHash64(f.request_id::String) % {k} = {i}"""

if __name__ == "__main__":
    state = rec.build_state(rec.hist_popularity())
    users = model.UserStore()
    users.load_file(rec.CACHE / "users_history.pkl")
    cov = model.index_coviews(rec.load_coviews())
    rows = [r for i in range(4) for r in rec.query(ROWS.format(team=rec.TEAM, k=4, i=i), timeout=300)]
    FI = model.FI
    tab = defaultdict(lambda: np.zeros(3))
    for r in rows:
        idx = [model.PRODUCTS.idx.get(p) for p in r["shown"]]
        if None in idx:
            continue
        F = model.featurize(r["u"], [{"product_id": p} for p in r["rv"]], state.items, users, cov)[idx]
        known = r["u"] in users.row
        for k, p in enumerate(r["shown"]):
            f = F[k]
            share = f[FI["sub_share"]]
            keys = [("posición", f"{k + 1:02d}"),
                    ("usuario", "con historial" if known else "sin historial"),
                    ("ya lo compró (60 días)", "sí" if f[FI["p_buy"]] > 0 else "no") if known else None,
                    ("lo tuvo en el carrito sin comprarlo", "sí" if f[FI["p_cart"]] > 0 and f[FI["p_buy"]] == 0 else "no") if known else None,
                    ("lo vio 2+ veces sin carrito ni compra", "sí" if f[FI["p_view"]] > 0 and f[FI["p_cart"]] == 0 and f[FI["p_buy"]] == 0 else "no") if known else None,
                    ("afinidad con la subcategoría", "0 nunca" if share == 0 else "1 <5%" if share < .05 else "2 5-15%" if share < .15 else "3 >15%") if known else None,
                    ("visto en esta sesión", "sí" if f[FI["in_session"]] else "no") if r["rv"] else None,
                    ("misma subcategoría que lo último visto", "sí" if f[FI["same_sub"]] else "no") if r["rv"] and not f[FI["in_session"]] else None,
                    ("misma categoría que lo último visto", "sí" if f[FI["same_cat"]] else "no") if r["rv"] and not f[FI["in_session"]] else None,
                    ("ventas en su subcategoría", "1 top3" if f[FI["pop_sub"]] >= 0.57 else "2 top10" if f[FI["pop_sub"]] >= 0.31 else "3 resto")]
            for key in filter(None, keys):
                tab[key] += (1, p == r["clicked"], p == r["purchased"])
    print(f"{len(rows)} respuestas nuestras servidas hoy\n")
    last = None
    for (group, value), (n, c, b) in sorted(tab.items()):
        if group != last:
            print(f"\n## {group}")
            last = group
        print(f"  {value:<16} impresiones={int(n):>7}  clic={100 * c / n:6.3f}%  compra={100 * b / n:6.3f}%  (compras={int(b)})")
