"""¿Aporta saber en cuántas sesiones ANTIGUAS vio cada comprador cada producto (señal hs_views)?

    python analysis/hist_views.py     # usa cache/live_matrix.npz (hecha con analysis/live_matrix.py --cf-fill)

Los recent_views de las requests del histórico (shop.history_sessions) no están en shop.history_events: son
una fuente distinta. Aquí se añade la señal a las de v32 y se mide con el feedback del evento por validación
cruzada en 5 bloques de tiempo (como analysis/pop_forms.py). No hay trampa posible: sale solo del histórico.
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
X, slot, buy, new, when = d["X"].astype(np.float64), d["slot"], d["buy"], d["new"].astype(bool), d["when"]
FEATS = list(d["features"]); FI = {n: i for i, n in enumerate(FEATS)}
cnt = np.rint(np.expm1(X[:, FI["hs_views"]]))          # nº de sesiones antiguas en que vio el producto
ins = X[:, FI["in_session"]] > 0
cart = X[:, FI["p_cart"]] > 0
P = {"hs log(1+n)": np.log1p(cnt), "hs log(1+min(n,3))": np.log1p(np.minimum(cnt, 3)), "hs visto alguna vez": (cnt >= 1).astype(float),
     "hs visto 2+ veces": (cnt >= 2).astype(float), "hs × en la sesión": np.log1p(np.minimum(cnt, 3)) * ins,
     "hs × en el carrito": np.log1p(np.minimum(cnt, 3)) * cart}
DROP = ("cart_no_buy", "viewed_no_cart", "organic", "on_sale", "cart_live", "view_live", "pop_log", "rate_buy", "hs_views")
BASE = [n for n in FEATS if n not in DROP]
onehot = np.eye(10)[slot][:, 1:]
block = np.minimum((np.argsort(np.argsort(when, kind="stable"), kind="stable") * 5 // len(when)), 4)


def cv(mask, extra, base=BASE, C=0.3):
    Z = np.stack([X[:, FI[n]] for n in base] + [P[e] for e in extra], 1)
    pr, yy = [], []
    for f in range(5):
        tr, te = mask & (block != f), mask & (block == f)
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        lr = LogisticRegression(C=C, max_iter=3000).fit(np.hstack([(Z[tr] - mu) / sd, onehot[tr]]), buy[tr])
        pr.append(lr.predict_proba(np.hstack([(Z[te] - mu) / sd, onehot[te]]))[:, 1]); yy.append(buy[te])
    p, y = np.concatenate(pr), np.concatenate(yy)
    per_block = [log_loss(b, a) for a, b in zip(pr, yy)]
    mu, sd = Z[mask].mean(0), Z[mask].std(0) + 1e-9
    lr = LogisticRegression(C=C, max_iter=3000).fit(np.hstack([(Z[mask] - mu) / sd, onehot[mask]]), buy[mask])
    return log_loss(y, p), roc_auc_score(y, p), lr.coef_[0][len(base):len(base) + len(extra)] / sd[len(base):], per_block


print("tasa de compra por nº de sesiones antiguas en que vio el producto (compradores conocidos, puestos 1-3):")
for label, m in (("no está en la sesión actual", ~ins), ("está en la sesión actual", ins)):
    out = []
    for name, c in (("0", cnt == 0), ("1", cnt == 1), ("2-3", (cnt >= 2) & (cnt <= 3)), ("4+", cnt >= 4)):
        k = ~new & (slot < 3) & m & c
        out.append(f"{name}: {100 * buy[k].mean() if k.sum() else 0:.2f} % ({int(buy[k].sum())}/{int(k.sum())})")
    print(f"   {label:<28}" + " | ".join(out))
NO_HC = [n for n in BASE if not n.startswith("hc_")]
for label, seg in (("COMPRADORES CONOCIDOS", ~new), ("COMPRADORES NUEVOS", new)):
    print(f"\n## {label}: {int(seg.sum()) // 10} respuestas, {int(buy[seg].sum())} compras")
    print(f"   {'modelo':<52}{'log-loss':>10}{'mejora':>9}{'AUC':>8}   bloques en que mejora | pesos")
    ll0, auc0, _, pb0 = cv(seg, [])
    print(f"   {'señales de v32':<52}{ll0:>10.5f}{0.0:>8.2f}%{auc0:>8.4f}")
    for name, extra, base in (("+ hs log(1+n)", ["hs log(1+n)"], BASE),
                              ("+ hs log(1+min(n,3))", ["hs log(1+min(n,3))"], BASE),
                              ("+ hs visto alguna vez", ["hs visto alguna vez"], BASE),
                              ("+ hs visto alguna vez y 2+ veces", ["hs visto alguna vez", "hs visto 2+ veces"], BASE),
                              ("+ hs log(1+min(n,3)) y × en la sesión", ["hs log(1+min(n,3))", "hs × en la sesión"], BASE),
                              ("+ hs log(1+min(n,3)) y × en el carrito", ["hs log(1+min(n,3))", "hs × en el carrito"], BASE),
                              ("sin hc_* (v24), para comparar", [], NO_HC),
                              ("sin hc_* + hs log(1+min(n,3))", ["hs log(1+min(n,3))"], NO_HC)):
        ll, auc, w, pb = cv(seg, extra, base)
        print(f"   {name:<52}{ll:>10.5f}{100 * (ll0 - ll) / ll0:>8.2f}%{auc:>8.4f}   {sum(a < b for a, b in zip(pb, pb0))}/5 | "
              + ", ".join(f"{x:+.2f}" for x in w))
print(f"\ntotal {time.time() - t0:.0f}s")
