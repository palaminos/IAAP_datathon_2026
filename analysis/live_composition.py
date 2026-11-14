"""¿De qué tipo de producto salen las compras, y en qué puesto los estamos enseñando? Datos de hoy.

    python analysis/live_matrix.py --cf-fill     # primero
    python analysis/live_composition.py          # segundos

Para compradores nuevos y conocidos: cada tipo de producto (en el carrito hoy, en el carrito solo en el
histórico, de la sesión, solo visto, nunca tocado) con sus impresiones, en qué puestos cae, sus compras y su
tasa de compra en los puestos 1-3 y 4-10. Sirve para ver dónde se ganan o se pierden compras.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import recommender as rec  # noqa: E402

d = np.load(rec.CACHE / (sys.argv[1] if len(sys.argv) > 1 else "live_matrix.npz"))
X, slot, buy, click, new, when = d["X"], d["slot"], d["buy"], d["click"], d["new"].astype(bool), d["when"]
FI = {n: i for i, n in enumerate(d["features"])}
recent = when >= np.quantile(when, 0.5)          # la segunda mitad del día (versiones más parecidas a las actuales)
cart_live = X[:, FI["cart_live"]] > 0
cart = (X[:, FI["p_cart"]] > 0) & (X[:, FI["p_buy"]] == 0)
sess = X[:, FI["in_session"]] > 0
viewed = (X[:, FI["p_view"]] > 0) | (X[:, FI["view_live"]] > 0)
bought = X[:, FI["p_buy"]] > 0
types = [("ya comprado", bought), ("carrito de HOY", cart_live & ~bought), ("carrito solo del histórico", cart & ~cart_live),
         ("de la sesión (sin carrito)", sess & ~cart & ~bought), ("solo visto antes", viewed & ~sess & ~cart & ~bought),
         ("nunca tocado", ~viewed & ~sess & ~cart & ~bought)]
for label, seg in (("COMPRADORES NUEVOS", new), ("COMPRADORES CONOCIDOS", ~new)):
    m0 = seg & recent
    nreq = m0.sum() / 10
    print(f"\n## {label}: {nreq:.0f} respuestas, conversión {100 * buy[m0].sum() / nreq:.2f} %")
    print(f"   {'tipo de producto':<28}{'% huecos':>9}{'por lista':>10}{'% en 1-3':>9}{'compras':>8}{'% compras':>10}{'compra 1.º':>11}{'compra 2-3':>11}{'compra 4-10':>12}")
    for name, t in types:
        m = m0 & t
        if not m.sum():
            continue
        r = lambda mm: 100 * buy[mm].mean() if mm.sum() else float("nan")
        print(f"   {name:<28}{100 * m.sum() / m0.sum():>8.1f}%{m.sum() / nreq:>10.2f}{100 * (slot[m] < 3).mean():>8.0f}%{int(buy[m].sum()):>8}"
              f"{100 * buy[m].sum() / max(buy[m0].sum(), 1):>9.0f}%{r(m & (slot == 0)):>10.2f}%{r(m & (slot > 0) & (slot < 3)):>10.2f}%{r(m & (slot >= 3)):>11.3f}%")
