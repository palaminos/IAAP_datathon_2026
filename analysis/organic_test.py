"""¿Sirve la actividad ORGÁNICA de hoy (shop.events) para entrenar el modelo del carrusel? Solo lee.

    python analysis/organic_test.py     # desde la raíz del repo

Hipótesis: shop.events lo genera el mismo simulador, no depende de lo que enseñamos y tiene ~120 veces más
compras que nuestro feedback. Ejemplos orgánicos: cada vista (usuario, producto, instante) con sus señales
calculadas SOLO con lo anterior (se reproduce el día en orden, con 30 s de retraso para no ver la propia
sesión) y la etiqueta "¿lo metió en el carrito / lo compró en los 30 min siguientes?".
Medida que decide: puntuar con ese modelo las impresiones del último 25 % de nuestro feedback y medir AUC
y log-loss de compra, calibrando con la posición (como train_live.judge). Se compara con v4 y v6.
"""
import datetime as dt
import sys
import time
import warnings
from collections import deque
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.simplefilter("ignore")
import model  # noqa: E402
import recommender as rec  # noqa: E402
import train_live as tl  # noqa: E402

SAMPLE = 12            # se usa 1 de cada 12 vistas orgánicas
LAG, HALF_H = 30_000, 30 * 60_000
OUTCOMES = """
SELECT user_id::String AS u, product_id::String AS p, event_type::String AS t, toUnixTimestamp64Milli(ts::DateTime64(3)) AS ms
FROM shop.events WHERE event_type != 'view' AND cityHash64(user_id::String) % 4 = {i}"""

t0 = time.time()
mf = model.load_mf()
state = rec.build_state(rec.hist_popularity(), mf)
items = state.items
users = model.UserStore()
users.load_file(rec.CACHE / "users_history_v2.pkl")
users.init_mf(mf)
cov_idx = model.index_coviews(rec.load_coviews())
reqs = sorted((r for i in range(4) for r in rec.query(tl.LIVE.format(team=rec.TEAM, i=i), timeout=300)), key=lambda r: int(r["ms"]))
reqs = [r for r in reqs if len(r["shown"]) == 10 and all(p in model.PRODUCTS.idx for p in r["shown"])]
later: dict[tuple, dict] = {}      # (usuario, producto) -> {"add_to_cart": [ms...], "purchase": [ms...]}
for i in range(4):
    for r in rec.query(OUTCOMES.format(i=i), timeout=300):
        later.setdefault((r["u"], r["p"]), {}).setdefault(r["t"], []).append(int(r["ms"]))
print(f"{len(reqs)} respuestas nuestras, {len(later)} pares usuario-producto con carrito o compra orgánicos ({time.time() - t0:.0f}s)", flush=True)

hour = dt.datetime.fromisoformat(rec.query("SELECT toString(toStartOfHour(min(ts::DateTime64(3)))) AS d FROM shop.events")[0]["d"])
last = dt.datetime.fromisoformat(rec.query("SELECT toString(max(ts::DateTime64(3))) AS d FROM shop.events")[0]["d"][:19])
pending: deque = deque()
Xo, mso, yc, yb = [], [], [], []
Xf, slot, click, buy, when = [], [], [], [], []
n_views = 0


def flush(t: int) -> None:
    """Incorpora a la memoria los eventos anteriores a t - 30 s (así una vista no ve su propia sesión)."""
    while pending and pending[0][3] < t - LAG:
        u, p, ty, _ = pending.popleft()
        users.add_event(u, p, ty, items)


def events():
    global hour
    while hour <= last:
        yield from rec.query(tl.EVENTS.format(a=hour, b=hour + dt.timedelta(hours=1)), timeout=300)
        hour += dt.timedelta(hours=1)


ev_it = events()
ev = next(ev_it, None)
for r in reqs + [None]:
    limit = int(r["ms"]) if r else 10**18
    while ev is not None and int(ev["ms"]) < limit:
        ms = int(ev["ms"])
        flush(ms)
        if ev["t"] == "view":
            n_views += 1
            i = model.PRODUCTS.idx.get(ev["p"])
            if n_views % SAMPLE == 0 and i is not None and i < items.n:
                out = later.get((ev["u"], ev["p"]), {})
                Xo.append(model.featurize(ev["u"], [], items, users, cov_idx)[i].copy())   # .copy(): una fila, no la tabla entera
                mso.append(ms)
                yc.append(any(ms <= t <= ms + HALF_H for t in out.get("add_to_cart", ())))
                yb.append(any(ms <= t <= ms + HALF_H for t in out.get("purchase", ())))
        pending.append((ev["u"], ev["p"], ev["t"], ms))
        ev = next(ev_it, None)
    if r is None:
        break
    flush(limit)
    idx = [model.PRODUCTS.idx[p] for p in r["shown"]]
    Xf.append(model.featurize(r["u"], [{"product_id": p} for p in r["rv"]], items, users, cov_idx)[idx])
    slot.append(np.arange(10)), when.append(np.full(10, limit))
    click.append([p == r["clicked"] for p in r["shown"]]), buy.append([p == r["purchased"] for p in r["shown"]])
    if r["purchased"]:
        users.add_event(r["u"], r["purchased"], "purchase", items)
c = np.concatenate
Xo, mso, yc, yb = np.array(Xo), np.array(mso), np.array(yc), np.array(yb)
Xf, slot, click, buy, when = c(Xf), c(slot), c(click).astype(bool), c(buy).astype(bool), c(when)
np.savez_compressed(rec.CACHE / "exp_organic.npz", Xo=Xo, mso=mso, yc=yc, yb=yb, Xf=Xf, slot=slot, click=click, buy=buy, when=when)
print(f"orgánico: {len(Xo)} vistas muestreadas, {int(yc.sum())} con carrito ({100 * yc.mean():.1f} %), {int(yb.sum())} con compra ({100 * yb.mean():.1f} %) | "
      f"feedback: {len(Xf) // 10} respuestas, {int(buy.sum())} compras ({time.time() - t0:.0f}s)", flush=True)

# ------------------------------------------------------------------ modelos
q75 = np.quantile(when, 0.75)
tr, te = when < q75, when >= q75
otr = mso < q75                       # solo orgánico ANTERIOR al tramo de prueba
onehot = tl.onehot


def lr_weights(X, y, C=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    lr = LogisticRegression(C=C, max_iter=3000).fit((X - mu) / sd, y)
    return (lr.coef_[0] / sd).astype(np.float64)


def judge(score, target):
    """Puntuación + posición -> probabilidad; se calibra en el 75 % inicial y se mide en el 25 % final."""
    s = np.clip(score, -60, 60)
    Z = np.hstack([((s - s[tr].mean()) / (s[tr].std() + 1e-9))[:, None], onehot(slot)])
    lr = LogisticRegression(C=10.0, max_iter=2000).fit(Z[tr], target[tr])
    p = lr.predict_proba(Z[te])[:, 1]
    return roc_auc_score(target[te], p), log_loss(target[te], p)


def ranking(score):
    """Entre los 10 productos mostrados, ¿en qué puesto queda el comprado si se ordena con esta puntuación?"""
    S, Y = score[te].reshape(-1, 10), buy[te].reshape(-1, 10)
    has = Y.any(1)
    S, Y = S[has], Y[has]
    sj = S[np.arange(len(S)), Y.argmax(1)][:, None]
    rank = (S > sj).sum(1) + ((S == sj).sum(1) - 1) / 2
    return float((1 / (1 + rank)).mean()), float((rank < 0.5).mean())


scores = {}
v4 = model.load_weights(model.WEIGHTS_FILE.with_name("model_weights_live.json")).astype(np.float64)
v6 = model.load_weights(model.WEIGHTS_FILE.with_name("model_weights_v6.json")).astype(np.float64)
scores["tal cual se sirvió"] = -slot.astype(np.float64)
scores["v4 en producción (*)"] = Xf @ v4
scores["v6 en producción (*)"] = Xf @ v6
mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-9
fb = LogisticRegression(C=0.1, max_iter=3000).fit(np.hstack([(Xf[tr] - mu) / sd, onehot(slot[tr])]), buy[tr])
w_fb = fb.coef_[0][:model.K] / sd
scores["feedback: LR entrenada con el 75 % inicial"] = Xf @ w_fb
w_oc, w_ob = lr_weights(Xo[otr], yc[otr]), lr_weights(Xo[otr], yb[otr])
scores["orgánico: LR carrito"] = Xf @ w_oc
scores["orgánico: LR compra"] = Xf @ w_ob
gb = dict(max_iter=200, learning_rate=0.08, max_depth=4, min_samples_leaf=200, l2_regularization=1.0, random_state=0)
g_oc = HistGradientBoostingClassifier(**gb).fit(Xo[otr], yc[otr])
g_ob = HistGradientBoostingClassifier(**gb).fit(Xo[otr], yb[otr])
scores["orgánico: GBM carrito"] = g_oc.decision_function(Xf)
scores["orgánico: GBM compra"] = g_ob.decision_function(Xf)
# orgánico como SEÑAL añadida al modelo de feedback (se ajusta con el 75 % inicial del feedback)
for name, extra in (("feedback + señal orgánica (LR carrito)", scores["orgánico: LR carrito"]),
                    ("feedback + señal orgánica (GBM carrito)", scores["orgánico: GBM carrito"]),
                    ("feedback + señal orgánica (GBM compra)", scores["orgánico: GBM compra"])):
    e = (extra - extra[tr].mean()) / (extra[tr].std() + 1e-9)
    Z = np.hstack([(Xf - mu) / sd, e[:, None]])
    lr = LogisticRegression(C=0.1, max_iter=3000).fit(np.hstack([Z[tr], onehot(slot[tr])]), buy[tr])
    scores[name] = Z @ lr.coef_[0][:model.K + 1]

print(f"\nPRUEBA: último 25 % del feedback ({int(buy[te].sum())} compras, {int(click[te].sum())} clics). Orgánico entrenado con "
      f"{int(otr.sum())} vistas anteriores a ese tramo ({int(yc[otr].sum())} carritos, {int(yb[otr].sum())} compras)")
print("(*) v4 y v6 se entrenaron con parte de este tramo: salen favorecidos\n")
print(f"{'':<46}{'AUC compra':>11}{'log-loss':>10}{'AUC clic':>10}{'MRR':>8}{'hit@1':>8}")
for name, s in scores.items():
    (ab, lb), (ac, _), (mrr, h1) = judge(s, buy), judge(s, click), ranking(s)
    print(f"{name:<46}{ab:>11.4f}{lb:>10.5f}{ac:>10.4f}{mrr:>8.4f}{h1:>8.3f}")
print("\nPesos (por desviación típica de la señal en el feedback): feedback | orgánico carrito | orgánico compra")
sdf = Xf.std(0)
for i in np.argsort(-np.abs(w_fb * sdf)):
    print(f"  {model.FEATURES[i]:<16}{w_fb[i] * sdf[i]:>9.3f}{w_oc[i] * sdf[i]:>9.3f}{w_ob[i] * sdf[i]:>9.3f}")
print(f"\ntotal {time.time() - t0:.0f}s")
