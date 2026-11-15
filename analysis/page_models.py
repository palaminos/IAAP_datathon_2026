"""¿Conviene tener pesos distintos según la página (home, product, cart...)?

    python analysis/page_models.py      # usa cache/live_matrix.npz

Validación cruzada en 5 bloques de tiempo: un solo modelo (con el efecto de la posición por página) frente a un
modelo por página, para compradores conocidos y para nuevos. Si un modelo por página predice bastante mejor, las
señales pesan distinto según dónde está el comprador.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.simplefilter("ignore")
import recommender as rec  # noqa: E402

t0 = time.time()
d = np.load(rec.CACHE / "live_matrix.npz")
X, slot, buy, new, when, page = d["X"].astype(np.float64), d["slot"], d["buy"], d["new"].astype(bool), d["when"], d["page"]
FEATS = list(d["features"]); FI = {n: i for i, n in enumerate(FEATS)}
DROP = ("cart_no_buy", "viewed_no_cart", "organic", "on_sale", "cart_live", "view_live", "pop_log", "rate_buy")
BASE = [FI[n] for n in FEATS if n not in DROP]
Z = X[:, BASE]
pages = sorted(set(page.tolist()))
block = np.minimum((np.argsort(np.argsort(when, kind="stable"), kind="stable") * 5 // len(when)), 4)
onehot = np.eye(10)[slot][:, 1:]
# efecto de la posición por página (10 puestos × página), para que el modelo único no salga perjudicado por eso
pos_page = np.zeros((len(X), 10 * len(pages)))
pidx = np.array([pages.index(p) for p in page])
pos_page[np.arange(len(X)), pidx * 10 + slot] = 1
pos_page = pos_page[:, 1:]


def fit_predict(tr, te, extra, C=0.3):
    mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
    lr = LogisticRegression(C=C, max_iter=3000).fit(np.hstack([(Z[tr] - mu) / sd, extra[tr]]), buy[tr])
    return lr.predict_proba(np.hstack([(Z[te] - mu) / sd, extra[te]]))[:, 1]


for label, seg in (("COMPRADORES CONOCIDOS", ~new), ("COMPRADORES NUEVOS", new)):
    print(f"\n## {label}")
    print("   página: respuestas | compras | conversión por respuesta | % de las compras en puestos 1-3")
    for p in pages:
        k = seg & (page == p)
        if k.sum():
            print(f"   {p or '(vacía)':<12}{int(k.sum()) // 10:>8} | {int(buy[k].sum()):>5} | {1000 * buy[k].mean():.2f} % | {100 * buy[k & (slot < 3)].sum() / max(buy[k].sum(), 1):.0f} %")
    P = {"un modelo, posición común": np.zeros(len(X)), "un modelo, posición por página": np.zeros(len(X)), "un modelo por página": np.zeros(len(X))}
    for f in range(5):
        tr, te = seg & (block != f), seg & (block == f)
        P["un modelo, posición común"][te] = fit_predict(tr, te, onehot)
        P["un modelo, posición por página"][te] = fit_predict(tr, te, pos_page)
        for p in pages:
            a, b = tr & (page == p), te & (page == p)
            if b.sum() == 0:
                continue
            if buy[a].sum() < 30:      # muy pocas compras en esa página: usa el modelo común
                P["un modelo por página"][b] = fit_predict(tr, b, pos_page)
            else:
                P["un modelo por página"][b] = fit_predict(a, b, onehot, C=0.1)
    ll0 = None
    for name, p in P.items():
        ll, auc = log_loss(buy[seg], p[seg]), roc_auc_score(buy[seg], p[seg])
        ll0 = ll0 or ll
        per = " ".join(f"{pg or '-'}:{log_loss(buy[seg & (page == pg)], p[seg & (page == pg)], labels=[0, 1]):.5f}" for pg in pages if (seg & (page == pg)).sum())
        print(f"   {name:<32} log-loss {ll:.5f} ({100 * (ll0 - ll) / ll0:+.2f} %)  AUC {auc:.4f} | {per}")
print(f"\ntotal {time.time() - t0:.0f}s")
