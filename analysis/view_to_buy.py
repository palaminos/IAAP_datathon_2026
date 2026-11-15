"""¿Aporta "quien vio X en una sesión acabó comprando Y en esa misma sesión"? (del histórico de eventos)

    python analysis/view_to_buy.py      # una agregación en RawTree (guardada en cache/view_to_buy.json) + cache/live_matrix.npz

Nuestro parecido entre productos (cf) sale de carritos y compras del mismo comprador en 60 días. Esto es otra cosa:
dentro de UNA sesión, qué se termina comprando después de ver X (incluido el propio X). Es la pregunta exacta de un
carrusel en la página de un producto, y sirve también para compradores nuevos (solo necesita lo que acaban de ver).
Señal candidata para cada producto Y de la lista: suma, sobre lo visto en la sesión, de P(comprar Y | vio X).
"""
import json
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
import train_live as tl  # noqa: E402

t0 = time.time()
path = rec.CACHE / "view_to_buy.json"
if path.exists():
    T = json.loads(path.read_text())
else:
    pairs = rec.query("""
SELECT x, y, n FROM (
    SELECT v.p AS x, b.p AS y, count() AS n
    FROM (SELECT DISTINCT session_id::String AS s, product_id::String AS p FROM shop.history_events WHERE event_type::String = 'view') AS v
    INNER JOIN (SELECT DISTINCT session_id::String AS s, product_id::String AS p FROM shop.history_events WHERE event_type::String = 'purchase') AS b ON v.s = b.s
    GROUP BY x, y HAVING n >= 3
) ORDER BY x, n DESC LIMIT 60 BY x""", timeout=900)
    views = rec.query("SELECT product_id::String AS p, uniqExact(session_id::String) AS n FROM shop.history_events WHERE event_type::String = 'view' GROUP BY p", timeout=300)
    T = {"pairs": [(r["x"], r["y"], int(r["n"])) for r in pairs], "views": {r["p"]: int(r["n"]) for r in views}}
    path.write_text(json.dumps(T))
print(f"vio X → compró Y: {len(T['pairs'])} parejas de {len(T['views'])} productos ({time.time() - t0:.0f}s)")
P = {}
for x, y, n in T["pairs"]:
    P.setdefault(x, {})[y] = n / max(T["views"].get(x, 0), 30)
self_rate = np.array([P.get(x, {}).get(x, 0.0) for x in list(P)[:5000]])
print(f"P(comprar el mismo producto que se vio, en la sesión): mediana {np.median(self_rate):.3f} | P(otro, el más frecuente): mediana {np.median([max([v for k, v in d.items() if k != x] or [0]) for x, d in list(P.items())[:5000]]):.3f}")

d = np.load(rec.CACHE / "live_matrix.npz")
X, slot, buy, new, when, prod, names = d["X"].astype(np.float64), d["slot"], d["buy"], d["new"].astype(bool), d["when"], d["prod"], d["names"].tolist()
FEATS = list(d["features"]); FI = {n: i for i, n in enumerate(FEATS)}
reqs = sorted((r for i in range(4) for r in rec.query(tl.LIVE.format(team=rec.TEAM, i=i), timeout=300)), key=lambda r: int(r["ms"]))
known_products = set(names)
reqs = [r for r in reqs if len(r["shown"]) == 10 and all(p in known_products for p in r["shown"])][:len(X) // 10]
assert len(reqs) * 10 == len(X) and all(int(r["ms"]) == int(w) for r, w in zip(reqs[::997], when[::9970])), "las filas no cuadran con la tabla guardada"
v2b, v2b_self, v2b_max = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
for k, r in enumerate(reqs):
    rv = list(dict.fromkeys(r["rv"]))
    if not rv:
        continue
    for j, y in enumerate(r["shown"]):
        vals = [P.get(x, {}).get(y, 0.0) for x in rv if x != y]
        if vals:
            v2b[k * 10 + j] = sum(vals); v2b_max[k * 10 + j] = max(vals)
        if y in rv:
            v2b_self[k * 10 + j] = P.get(y, {}).get(y, 0.0)
print(f"señales calculadas para {len(reqs)} respuestas ({time.time() - t0:.0f}s) | filas con señal: {100 * (v2b > 0).mean():.1f} % | del propio producto visto: {100 * (v2b_self > 0).mean():.1f} %")
C = {"v2b log": np.log1p(100 * v2b), "v2b > 0": (v2b > 0).astype(float), "v2b máx": np.log1p(100 * v2b_max), "propio": np.log1p(100 * v2b_self)}
top = slot < 3
ins = X[:, FI["in_session"]] > 0
for label, seg in (("conocidos", ~new), ("nuevos", new)):
    q = [0, 1e-9, 0.002, 0.005, 0.01, 0.02, 9]
    print(f"tasa de compra ({label}, puestos 1-3, producto que NO está entre lo visto) según P(comprar Y | lo visto en la sesión): " +
          " | ".join(f"{lo:g}-{hi:g}: {100 * buy[seg & top & ~ins & (v2b >= lo) & (v2b < hi)].mean():.2f} % ({int((seg & top & ~ins & (v2b >= lo) & (v2b < hi)).sum())})" for lo, hi in zip(q, q[1:])))
DROP = ("cart_no_buy", "viewed_no_cart", "organic", "on_sale", "cart_live", "view_live", "pop_log", "rate_buy")
onehot = np.eye(10)[slot][:, 1:]
block = np.minimum((np.argsort(np.argsort(when, kind="stable"), kind="stable") * 5 // len(when)), 4)


def cv(mask, extra, base, Cc=0.3):
    Z = np.stack([X[:, FI[n]] for n in base] + [C[e] for e in extra], 1)
    pr, yy = [], []
    for f in range(5):
        tr, te = mask & (block != f), mask & (block == f)
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        lr = LogisticRegression(C=Cc, max_iter=3000).fit(np.hstack([(Z[tr] - mu) / sd, onehot[tr]]), buy[tr])
        pr.append(lr.predict_proba(np.hstack([(Z[te] - mu) / sd, onehot[te]]))[:, 1]); yy.append(buy[te])
    p, y = np.concatenate(pr), np.concatenate(yy)
    mu, sd = Z[mask].mean(0), Z[mask].std(0) + 1e-9
    lr = LogisticRegression(C=Cc, max_iter=3000).fit(np.hstack([(Z[mask] - mu) / sd, onehot[mask]]), buy[mask])
    return log_loss(y, p), roc_auc_score(y, p), lr.coef_[0][len(base):len(base) + len(extra)] / sd[len(base):], [log_loss(b, a) for a, b in zip(pr, yy)]


for label, seg in (("COMPRADORES NUEVOS", new), ("COMPRADORES CONOCIDOS", ~new)):
    base = [n for n in FEATS if n not in DROP and (seg is not new or not n.startswith(("hc_", "hs_")))]
    print(f"\n## {label}: {int(seg.sum()) // 10} respuestas, {int(buy[seg].sum())} compras")
    ll0, auc0, _, pb0 = cv(seg, [], base)
    print(f"   {'señales actuales':<40}{ll0:>10.5f}{0.0:>8.2f}%{auc0:>8.4f}")
    for name, extra in (("+ v2b log", ["v2b log"]), ("+ v2b máx", ["v2b máx"]), ("+ propio", ["propio"]), ("+ v2b log y propio", ["v2b log", "propio"]), ("+ v2b log, v2b > 0 y propio", ["v2b log", "v2b > 0", "propio"])):
        ll, auc, w, pb = cv(seg, extra, base)
        print(f"   {name:<40}{ll:>10.5f}{100 * (ll0 - ll) / ll0:>8.2f}%{auc:>8.4f}   {sum(a < b for a, b in zip(pb, pb0))}/5 | " + ", ".join(f"{x:+.2f}" for x in w))
np.savez_compressed(rec.CACHE / "live_v2b.npz", v2b=v2b.astype(np.float32), v2b_self=v2b_self.astype(np.float32), v2b_max=v2b_max.astype(np.float32))
print(f"\ntotal {time.time() - t0:.0f}s")
