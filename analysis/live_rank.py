"""¿Se aprenden mejor los pesos comparando SOLO productos de una misma lista? (logit condicional)

    python analysis/live_matrix.py --cf-fill    # primero
    python analysis/live_rank.py                # segundos

Lo que hacemos en producción es ORDENAR los productos de una request. La regresión logística de
train_live.py mezcla dos cosas: qué producto de la lista se compra (lo que nos interesa) y qué requests
acaban en compra (compradores nuevos, página... que no ayuda a ordenar y contamina los pesos). El logit
condicional solo mira las listas con compra y pregunta: de estos 10, ¿por qué se compró este? El puesto
entra como control (arriba se compra más) y se descarta al servir.

Medida (último 25 % del día, listas con compra): probabilidad que el modelo da al producto que se compró,
sabiendo el puesto de cada uno (log-verosimilitud media; más alto = mejor) y % de aciertos del primero.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.simplefilter("ignore")
import recommender as rec  # noqa: E402

d = np.load(rec.CACHE / (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".npz") else "live_matrix.npz"))
FEATS = list(d["features"])
DROP = ["cart_no_buy", "viewed_no_cart", "organic", "on_sale"]
cols = [i for i, n in enumerate(FEATS) if n not in DROP]
X = d["X"][:, cols].astype(np.float64)
mu, sd = X.mean(0), X.std(0) + 1e-9
Z = ((X - mu) / sd).reshape(-1, 10, len(cols))          # requests × 10 × señales (estandarizadas)
buy, click = d["buy"].reshape(-1, 10), d["click"].reshape(-1, 10)
when, new = d["when"].reshape(-1, 10)[:, 0], d["new"].reshape(-1, 10)[:, 0].astype(bool)
slot_flat = d["slot"]
q = np.quantile(when, 0.75)
tr, te = when < q, when >= q
has_b, has_c = buy.any(1), click.any(1)
S = np.eye(10)[:, 1:]                                    # puesto (one-hot sin el primero), igual en todas las listas
print(f"listas: {len(Z)} | con compra {has_b.sum()} | con clic {has_c.sum()} | compra sin clic en el mismo producto: "
      f"{int((buy & ~click).any(1).sum())}")


def cl_fit(groups, C=1.0, with_slot=True):
    """groups = [(Z_r [n,10,F], elegido [n], peso)]. Devuelve (w, b_puesto)."""
    F = groups[0][0].shape[2]

    def loss(theta):
        w, b = theta[:F], (theta[F:] if with_slot else np.zeros(9))
        total, gw, gb = 0.0, np.zeros(F), np.zeros(9)
        for Zr, c, wt in groups:
            sc = Zr @ w + S @ b
            lse = logsumexp(sc, axis=1)
            p = np.exp(sc - lse[:, None])
            n = np.arange(len(c))
            total -= wt * (sc[n, c] - lse).sum()
            gw += wt * (np.einsum("ni,nif->f", p, Zr) - Zr[n, c].sum(0))
            onehot_c = np.zeros_like(p); onehot_c[n, c] = 1
            gb += wt * ((p - onehot_c) @ S).sum(0)
        total += 0.5 * (w @ w) / C
        g = np.concatenate([gw + w / C, gb]) if with_slot else gw + w / C
        return total, g

    r = minimize(loss, np.zeros(F + (9 if with_slot else 0)), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return r.x[:F], (r.x[F:] if with_slot else np.zeros(9))


def lr_fit(mask_req, C=1.0):
    rows = np.repeat(mask_req, 10)
    lr = LogisticRegression(C=C, max_iter=4000).fit(np.hstack([Z.reshape(-1, Z.shape[2])[rows], np.eye(10)[slot_flat[rows]][:, 1:]]), buy.reshape(-1)[rows])
    return lr.coef_[0][:Z.shape[2]]


def evaluate(score, seg):
    """score: requests × 10. Calibra a·score + puesto con las listas con compra de entrenamiento del segmento."""
    a_tr, a_te = tr & has_b & seg, te & has_b & seg
    g = [(score[a_tr][:, :, None], buy[a_tr].argmax(1), 1.0)]
    a, b = cl_fit(g, C=1e6)
    sc = a[0] * score[a_te] + S @ b
    c = buy[a_te].argmax(1)
    n = np.arange(len(c))
    ll = (sc[n, c] - logsumexp(sc, axis=1)).mean()
    top1 = (sc.argmax(1) == c).mean()
    s0 = score[a_te]
    rank = ((s0 > s0[n, c][:, None]).sum(1) + ((s0 == s0[n, c][:, None]).sum(1) - 1) / 2 + 1).mean()
    return ll, 100 * top1, rank, int(a_te.sum())


segs = {"NUEVOS": new, "CONOCIDOS": ~new}
models = {}
models["solo el puesto (sin modelo)"] = {k: np.zeros(Z.shape[2]) for k in segs}
w = lr_fit(tr)
models["regresión logística única (como v13)"] = {k: w for k in segs}
models["regresión logística por segmento (como v14)"] = {k: lr_fit(tr & m, 1.0 if k == "CONOCIDOS" else 0.1) for k, m in segs.items()}
for C in (0.1, 1.0, 10.0):
    w, _ = cl_fit([(Z[tr & has_b], buy[tr & has_b].argmax(1), 1.0)], C)
    models[f"logit condicional único, compras (C={C})"] = {k: w for k in segs}
for C in (0.1, 1.0, 10.0):
    models[f"logit condicional por segmento, compras (C={C})"] = {k: cl_fit([(Z[tr & has_b & m], buy[tr & has_b & m].argmax(1), 1.0)], C)[0] for k, m in segs.items()}
for C in (0.1, 1.0):
    models[f"logit condicional por segmento, clics (C={C})"] = {k: cl_fit([(Z[tr & has_c & m], click[tr & has_c & m].argmax(1), 1.0)], C)[0] for k, m in segs.items()}
for alpha in (0.3, 1.0):
    models[f"logit cond. por segmento, compras + {alpha}×clics (C=1)"] = {
        k: cl_fit([(Z[tr & has_b & m], buy[tr & has_b & m].argmax(1), 1.0), (Z[tr & has_c & m], click[tr & has_c & m].argmax(1), alpha)], 1.0)[0]
        for k, m in segs.items()}
print(f"\n{'':<56}" + "".join(f"| {k + ': log-ver.':>20}{'acierto 1.º':>12}{'puesto':>8} " for k in segs))
for name, ws in models.items():
    line = f"{name:<56}"
    for k, m in segs.items():
        ll, top1, rank, n = evaluate(Z @ ws[k], m)
        line += f"| {ll:>20.4f}{top1:>11.1f}%{rank:>8.2f} "
    print(line)
print("\nlog-ver. = log de la probabilidad dada al producto comprado (más alto, mejor; el azar puro sería -2,30)")
print("acierto 1.º = % de listas en que el producto comprado era el favorito del modelo (contando el puesto)")
