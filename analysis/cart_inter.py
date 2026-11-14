"""Compradores conocidos: ¿hay que puntuar distinto los productos que tienen en el carrito? Histórico. Solo lee.

    python analysis/cart_inter.py      # ~4 min

El 72 % de las compras de los conocidos sale de productos del carrito, y en el histórico esos productos se
compran según cosas que el modelo trata igual para todos: lo vendido que es (top 1 %: x2,8), el precio...
Aquí se prueba añadir al modelo señales que solo valen para los productos del carrito (carrito × popularidad,
carrito × precio, carrito × parecido...). Histórico sin recompras; semana 1 entrena, semana 2 mide:
log-loss y AUC de compra sabiendo el puesto, y puesto medio del producto comprado dentro de su lista.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model  # noqa: E402
from experiments import *  # noqa: F401,F403,E402  (usa la caché local: no descarga nada grande)

t0 = time.time()
ctx = Ctx()
cf = model.load_cf(rec.CACHE / f"cf_sim_until_{CUT}.npz")
items = rec.make_state(ctx.state.product, set(ctx.state.product), cached("pop14", lambda: None), {}, cf=cf).items
Xs, slots, buys, ws, weeks = [], [], [], [], []
for s in ctx.sessions:
    idx = [model.PRODUCTS.idx[p] for p in s["shown"]]
    F = model.featurize(s["u"], [{"product_id": p} for p in s["rv"]], items, ctx.users, ctx.cov_idx)[idx]
    Xs.append(F); slots.append(np.arange(10)); buys.append([p == s["purchased"] for p in s["shown"]])
    ws.append(np.full(10, 1.0 if s["clicked"] else 100 / NEG_PCT) * (F[:, model.FI["p_buy"]] == 0))   # sin recompras
    weeks.append(np.full(10, s["week"]))
X, slot, buy, w, week = np.concatenate(Xs).astype(np.float64), np.concatenate(slots), np.concatenate(buys).astype(bool), np.concatenate(ws), np.concatenate(weeks)
FI = model.FI
ok = w > 0
onehot = np.eye(10)[slot][:, 1:]
tr, te = ok & (week == 0), ok & (week == 1)
cart = ((X[:, FI["p_cart"]] > 0) & (X[:, FI["p_buy"]] == 0)).astype(np.float64)
seen = ((X[:, FI["p_view"]] > 0) & (X[:, FI["p_cart"]] == 0) & (X[:, FI["p_buy"]] == 0)).astype(np.float64)
BASE = [n for n in model.FEATURES if n not in ("cart_no_buy", "viewed_no_cart", "organic", "on_sale", "cart_live", "view_live",
                                                "pop_log", "mf", "mf_z", "is_new_prod", "p_buy")]
col = lambda n: X[:, FI[n]]
extra = {
    "pop_log": col("pop_log"),
    "cart": cart,
    "cart×pop_log": cart * col("pop_log"), "cart×pop_sub": cart * col("pop_sub"), "cart×pop_glob": cart * col("pop_glob"),
    "cart×log_price": cart * col("log_price"), "cart×cf": cart * col("cf"), "cart×sub_share": cart * col("sub_share"),
    "cart×p_view": cart * col("p_view"), "cart×same_cat": cart * col("same_cat"), "cart×in_session": cart * col("in_session"),
    "cart×cf_sess": cart * col("cf_sess"),
    "seen": seen, "seen×pop_log": seen * col("pop_log"), "seen×cf": seen * col("cf"),
}
print(f"{int(ok.sum())} impresiones sin recompra, {int(buy[ok].sum())} compras; de productos del carrito: {int((buy & (cart > 0))[ok].sum())} ({time.time() - t0:.0f}s)\n")
group = np.repeat(np.arange(len(X) // 10), 10)


def run(name, names, C=1.0):
    Z = np.stack([col(n) if n in FI and n not in extra else extra[n] for n in names], 1)
    mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
    Zs = np.hstack([(Z - mu) / sd, onehot])
    lr = LogisticRegression(C=C, max_iter=4000).fit(Zs[tr], buy[tr], sample_weight=w[tr])
    p = lr.predict_proba(Zs[te])[:, 1]
    s = ((Z - mu) / sd) @ lr.coef_[0][:len(names)]
    # puesto del producto comprado dentro de su lista (entre los productos sin recompra), semana 2
    ranks, ranks_cart = [], []
    for g in np.unique(group[te & buy]):
        m = (group == g) & ok
        j = np.flatnonzero(m & buy)[0]
        r = 1 + (s[m] > s[j]).sum() + ((s[m] == s[j]).sum() - 1) / 2
        ranks.append(r)
        if cart[j]:
            ranks_cart.append(r)
    c = te & (cart > 0)
    print(f"{name:<52}{log_loss(buy[te], p, sample_weight=w[te]):>10.5f}{roc_auc_score(buy[te], p, sample_weight=w[te]):>8.4f}{np.mean(ranks):>8.3f}"
          f"{log_loss(buy[c], lr.predict_proba(Zs[c])[:, 1], sample_weight=w[c]):>12.5f}{np.mean(ranks_cart):>10.3f}")
    return dict(zip(names, lr.coef_[0][:len(names)]))


print(f"{'modelo':<52}{'log-loss':>10}{'AUC':>8}{'puesto':>8}{'ll carrito':>12}{'puesto c.':>10}")
run("base (señales de v14)", BASE)
run("+ pop_log", BASE + ["pop_log"])
run("+ cart (bandera)", BASE + ["cart"])
INT = ["cart", "cart×pop_log", "cart×pop_sub", "cart×log_price", "cart×cf", "cart×sub_share", "cart×p_view", "cart×same_cat", "cart×in_session", "cart×cf_sess"]
run("+ cart y sus cruces", BASE + INT)
co = run("+ pop_log + cart y sus cruces", BASE + ["pop_log"] + INT)
run("+ pop_log + cart y cruces + visto (y cruces)", BASE + ["pop_log"] + INT + ["seen", "seen×pop_log", "seen×cf"])
run("+ pop_log + cart × (pop_log, log_price, cf)", BASE + ["pop_log", "cart", "cart×pop_log", "cart×log_price", "cart×cf"])
print("\npuesto = puesto medio del producto comprado en su lista (menos es mejor); 'c.' = solo compras de productos del carrito")
print("pesos estandarizados del modelo '+ pop_log + cart y sus cruces':")
for k, v in sorted(co.items(), key=lambda kv: -abs(kv[1])):
    print(f"   {k:<18}{v:>8.3f}")
print(f"total {time.time() - t0:.0f}s")
