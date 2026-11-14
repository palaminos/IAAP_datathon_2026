"""¿Qué dicen los datos EN VIVO (nuestro feedback de hoy) de las señales de parecido entre productos?

    python analysis/live_matrix.py   # primero: reproduce el día y guarda la tabla (2-4 min)
    python analysis/live_cf.py       # segundos

Para cada señal: tasa de compra y de clic por tramos, corrigiendo por el puesto en que se enseñó
(regresión con el puesto como control: se muestra el multiplicador de la probabilidad de compra).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.simplefilter("ignore")
import model  # noqa: E402
import recommender as rec  # noqa: E402

d = np.load(rec.CACHE / "live_matrix.npz")
X, slot, click, buy, when = d["X"], d["slot"], d["click"], d["buy"], d["when"]
new, page, version = d["new"], d["page"], d["version"]
FI = {n: i for i, n in enumerate(d["features"])}
n_req = len(X) // 10
print(f"{n_req} respuestas, {int(buy.sum())} compras, {int(click.sum())} clics\n")

print("## Conversión por tipo de comprador y página (por respuesta)")
B, C = buy.reshape(-1, 10).any(1), click.reshape(-1, 10).any(1)
N, PG = new.reshape(-1, 10)[:, 0], page.reshape(-1, 10)[:, 0]
for label, m in (("conocidos", ~N), ("nuevos", N)):
    print(f"  {label:<10} respuestas {m.sum():>6} | conversión {100 * B[m].mean():.2f} % | CTR {100 * C[m].mean():.2f} %   " +
          " | ".join(f"{pg}: {100 * B[m & (PG == pg)].mean():.2f} % ({(m & (PG == pg)).sum()})" for pg in ("home", "category", "product")))
print()


def lift(name, x, mask, target=buy, edges=None):
    """Multiplicador de la probabilidad (odds) de cada tramo frente al tramo 0, controlando el puesto."""
    pos = x[mask & (x > 0)]
    if len(pos) < 200:
        print(f"## {name}: sin datos\n"); return
    qs = np.quantile(pos, [0.5, 0.8, 0.95]) if edges is None else edges
    b = np.where(x <= 0, 0, 1 + np.searchsorted(qs, x, side="right"))
    Z = np.hstack([np.eye(10)[slot[mask]][:, 1:], np.eye(len(qs) + 2)[b[mask]][:, 1:]])
    lr = LogisticRegression(C=100.0, max_iter=3000).fit(Z, target[mask])
    odds = np.exp(np.concatenate([[0.0], lr.coef_[0][9:]]))
    print(f"## {name}  (impresiones {mask.sum()}, compras {int(buy[mask].sum())})")
    labels = ["0 (sin señal)", "1 bajo", "2 medio", "3 alto (top 20 %)", "4 muy alto (top 5 %)"]
    for g in range(len(qs) + 2):
        m = mask & (b == g)
        if m.sum():
            print(f"   {labels[g]:<22} impresiones={m.sum():>8}  compra={100 * buy[m].mean():6.3f} %  clic={100 * click[m].mean():6.3f} %  "
                  f"x{odds[g]:5.2f} a igual puesto  compras={int(buy[m].sum())}")
    print()


cf, cfs, cfv = X[:, FI["cf"]], X[:, FI["cf_sess"]], X[:, FI["cf_view"]]
touched = (X[:, FI["p_view"]] > 0) | (X[:, FI["p_cart"]] > 0) | (X[:, FI["p_buy"]] > 0)
insess = X[:, FI["in_session"]] > 0
has_liked = np.repeat((X[:, FI["cf_any"]].reshape(-1, 10) > 0).any(1), 10)   # aprox.: algún producto mostrado tiene cf
has_rv = np.repeat((X[:, [FI["in_session"], FI["same_cat"], FI["cf_sess_any"]]].reshape(-1, 30) > 0).any(1), 10)
fresh = ~touched & ~insess
lift("cf (parecido a carritos/compras): todas las impresiones", cf, np.ones(len(X), bool))
lift("cf: productos que el comprador nunca había tocado ni visto en la sesión", cf, fresh)
lift("cf: nunca tocados, compradores CONOCIDOS con alguna señal cf en la lista", cf, fresh & ~new & has_liked)
lift("cf_sess (parecido a lo visto en la sesión): todas las impresiones con sesión", cfs, has_rv)
lift("cf_sess: producto no visto en la sesión ni tocado antes", cfs, has_rv & fresh)
lift("cf_sess: compradores NUEVOS, producto no visto en la sesión", cfs, has_rv & new & ~insess)
lift("cf_sess: conocidos, nunca tocado, y CON parecido a carritos/compras", cfs, has_rv & ~new & fresh & (cf > 0))
lift("cf_sess: nunca tocado y SIN parecido a carritos/compras", cfs, has_rv & fresh & (cf <= 0))
lift("cf_view (parecido a lo visto otros días): nunca tocados", cfv, fresh)
lift("cf_view: nunca tocados y SIN parecido a carritos/compras", cfv, fresh & (cf <= 0))
