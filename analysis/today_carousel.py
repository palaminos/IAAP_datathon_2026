"""¿Aporta lo que cada comprador ha hecho HOY con nuestros carruseles (enseñado y no clicado, clicado sin comprar)?

    python analysis/today_carousel.py      # usa cache/live_matrix.npz; guarda cache/live_extra.npz

Para cada producto enseñado se cuenta, con lo ocurrido ANTES de esa request en nuestro propio feedback:
  * veces que ya se le había enseñado a ese comprador hoy sin que lo clicara,
  * si ya lo había clicado en un carrusel nuestro (sin comprarlo).
Se mira la tasa de compra en bruto y cuánto mejora la predicción al añadirlo a las señales de v33
(validación cruzada en 5 bloques de tiempo). En producción se podría llevar la cuenta con nuestras propias respuestas.
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
import model  # noqa: E402
import recommender as rec  # noqa: E402
import train_live as tl  # noqa: E402

t0 = time.time()
d = np.load(rec.CACHE / "live_matrix.npz")
X, slot, buy, click, new, when, prod = d["X"].astype(np.float64), d["slot"], d["buy"], d["click"], d["new"].astype(bool), d["when"], d["prod"]
FEATS = list(d["features"]); FI = {n: i for i, n in enumerate(FEATS)}
extra_file = rec.CACHE / "live_extra.npz"
if extra_file.exists() and len(np.load(extra_file)["ign"]) == len(X):
    e = np.load(extra_file)
    ign, clk, gap, last = e["ign"], e["clk"], e["gap"], e["last"]
else:
    reqs = sorted((r for i in range(4) for r in rec.query(tl.LIVE.format(team=rec.TEAM, i=i), timeout=300)), key=lambda r: int(r["ms"]))
    known_products = set(d["names"].tolist())     # el catálogo con el que se hizo la tabla
    reqs = [r for r in reqs if len(r["shown"]) == 10 and all(p in known_products for p in r["shown"])]
    reqs = reqs[:len(X) // 10]
    assert len(reqs) * 10 == len(X), (len(reqs), len(X))
    assert all(int(r["ms"]) == int(w) for r, w in zip(reqs[::997], when[::9970])), "las filas no cuadran con la tabla guardada"
    ign, clk, gap, last = np.zeros(len(X), np.float32), np.zeros(len(X), np.float32), np.zeros(len(X), np.float32), np.zeros(len(X), np.float32)
    seen: dict[tuple, list] = {}       # (comprador, producto) -> [veces enseñado sin clic, veces clicado sin compra, ms de la última vez]
    nreq: dict[str, int] = {}
    for k, r in enumerate(reqs):
        ms, u = int(r["ms"]), r["u"]
        for j, p in enumerate(r["shown"]):
            s = seen.get((u, p))
            if s:
                ign[k * 10 + j], clk[k * 10 + j] = s[0], s[1]
                gap[k * 10 + j] = (ms - s[2]) / 60000.0
                last[k * 10 + j] = float(s[3] == nreq.get(u, 0))     # estaba en la lista de su request anterior
        n = nreq[u] = nreq.get(u, 0) + 1
        for p in r["shown"]:
            s = seen.setdefault((u, p), [0, 0, ms, n])
            if p == r["clicked"]:
                s[1] += p != r["purchased"]
            else:
                s[0] += 1
            s[2], s[3] = ms, n
    np.savez_compressed(extra_file, ign=ign, clk=clk, gap=gap, last=last)
    print(f"feedback de hoy recontado en {time.time() - t0:.0f}s")

top = slot < 3
print("tasa de compra en los puestos 1-3 según las veces que YA se le había enseñado hoy sin clic:")
for label, seg in (("conocidos", ~new), ("nuevos", new)):
    out = []
    for name, m in (("0", ign == 0), ("1", ign == 1), ("2", ign == 2), ("3-5", (ign >= 3) & (ign <= 5)), ("6+", ign >= 6)):
        k = seg & top & m & (clk == 0)
        out.append(f"{name}: {100 * buy[k].mean():.2f} % ({int(buy[k].sum())}/{int(k.sum())})")
    k = seg & top & (clk > 0)
    print(f"   {label:<10}" + " | ".join(out) + f" || ya clicado hoy sin comprar: {100 * buy[k].mean() if k.sum() else 0:.2f} % ({int(buy[k].sum())}/{int(k.sum())})")
cart = X[:, FI["p_cart"]] > 0
print("lo mismo, solo productos del carrito (compradores conocidos):")
out = []
for name, m in (("0", ign == 0), ("1", ign == 1), ("2", ign == 2), ("3-5", (ign >= 3) & (ign <= 5)), ("6+", ign >= 6)):
    k = ~new & top & m & cart & (clk == 0)
    out.append(f"{name}: {100 * buy[k].mean():.2f} % ({int(buy[k].sum())}/{int(k.sum())})")
print("   " + " | ".join(out))
print(f"% de los puestos 1-3 con un producto ya enseñado hoy a ese comprador: conocidos {100 * (ign[~new & top] > 0).mean():.0f} %, nuevos {100 * (ign[new & top] > 0).mean():.0f} %")

P = {"ign log(1+n)": np.log1p(ign), "ign log(1+min(n,5))": np.log1p(np.minimum(ign, 5)), "ign > 0": (ign > 0).astype(float), "clicado hoy": (clk > 0).astype(float),
     "ign × carrito": np.log1p(np.minimum(ign, 5)) * cart, "en la request anterior": last.astype(float)}
DROP = ("cart_no_buy", "viewed_no_cart", "organic", "on_sale", "cart_live", "view_live", "pop_log", "rate_buy")
BASE = [n for n in FEATS if n not in DROP]
onehot = np.eye(10)[slot][:, 1:]
block = np.minimum((np.argsort(np.argsort(when, kind="stable"), kind="stable") * 5 // len(when)), 4)


def cv(mask, extra, C=0.3):
    Z = np.stack([X[:, FI[n]] for n in BASE] + [P[e] for e in extra], 1)
    pr, yy = [], []
    for f in range(5):
        tr, te = mask & (block != f), mask & (block == f)
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        lr = LogisticRegression(C=C, max_iter=3000).fit(np.hstack([(Z[tr] - mu) / sd, onehot[tr]]), buy[tr])
        pr.append(lr.predict_proba(np.hstack([(Z[te] - mu) / sd, onehot[te]]))[:, 1]); yy.append(buy[te])
    p, y = np.concatenate(pr), np.concatenate(yy)
    mu, sd = Z[mask].mean(0), Z[mask].std(0) + 1e-9
    lr = LogisticRegression(C=C, max_iter=3000).fit(np.hstack([(Z[mask] - mu) / sd, onehot[mask]]), buy[mask])
    return log_loss(y, p), roc_auc_score(y, p), lr.coef_[0][len(BASE):len(BASE) + len(extra)] / sd[len(BASE):], [log_loss(b, a) for a, b in zip(pr, yy)]


for label, seg in (("COMPRADORES CONOCIDOS", ~new), ("COMPRADORES NUEVOS", new)):
    print(f"\n## {label}: {int(seg.sum()) // 10} respuestas, {int(buy[seg].sum())} compras")
    ll0, auc0, _, pb0 = cv(seg, [])
    print(f"   {'señales de v33':<46}{ll0:>10.5f}{0.0:>8.2f}%{auc0:>8.4f}")
    for name, extra in (("+ ign log(1+n)", ["ign log(1+n)"]), ("+ ign log(1+min(n,5))", ["ign log(1+min(n,5))"]), ("+ ign > 0", ["ign > 0"]),
                        ("+ clicado hoy", ["clicado hoy"]), ("+ ign log(1+min(n,5)) y clicado hoy", ["ign log(1+min(n,5))", "clicado hoy"]),
                        ("+ ign, clicado hoy e ign × carrito", ["ign log(1+min(n,5))", "clicado hoy", "ign × carrito"]),
                        ("+ ign, clicado hoy y en la request anterior", ["ign log(1+min(n,5))", "clicado hoy", "en la request anterior"])):
        ll, auc, w, pb = cv(seg, extra)
        print(f"   {name:<46}{ll:>10.5f}{100 * (ll0 - ll) / ll0:>8.2f}%{auc:>8.4f}   {sum(a < b for a, b in zip(pb, pb0))}/5 | " + ", ".join(f"{x:+.2f}" for x in w))
print(f"\ntotal {time.time() - t0:.0f}s")
