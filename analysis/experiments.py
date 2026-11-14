"""Banco de pruebas offline: mide ideas contra una línea base con split y métrica fijos. Solo lee.

    python analysis/experiments.py     # desde la raíz del repo

SPLIT (fijo):  datos de apoyo < 2026-11-01 | entrenar 1-7 nov | probar 8-14 nov
MÉTRICA (fija), sobre los carruseles de la semana de prueba que acabaron en compra:
    NDCG@10   = 1/log2(1 + puesto del producto comprado en nuestro ranking), 0 si no está en el top 10
    Recall@10 = % de veces que el producto comprado está en nuestro top 10
"""
from __future__ import annotations

import datetime as dt
import math
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))  # raíz del repo
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import model
import recommender as rec

CUT = dt.date(2026, 11, 1)
UNTIL = f"{CUT} 00:00:00"
NEG_PCT = 30
BASE = list(model.FEATURES)
EXTRA = ["is_home", "aff_x_pop", "cart_abandoned", "pop_cat", "price_diff", "price_absdiff", "region_lift",
         "device_lift", "sub_share_recent", "sub_log_recent", "p_recent", "pop_recent", "trend", "cobuy"]
ALL = BASE + EXTRA
COL = {n: i for i, n in enumerate(ALL)}

SESSIONS = """
SELECT user_id::String AS u, context.page::String AS page, context.region::String AS region, context.device::String AS device,
       arrayMap(x -> x.product_id::String, recent_views::Array(JSON)) AS rv,
       shown::Array(String) AS shown, ifNull(clicked::String, '') AS clicked, ifNull(purchased::String, '') AS purchased
FROM shop.history_sessions
WHERE ts::DateTime64(3) >= '{a}' AND ts::DateTime64(3) < '{b}'
  AND (clicked != '' OR cityHash64(request_id::String) % 100 < {pct})"""
USER_PRICE = "SELECT user_id::String AS u, avg(log(price_eur::Float64)) AS m FROM shop.history_events WHERE ts::DateTime64(3) < '{until}' GROUP BY u"
SEGMENT_POP = """
SELECT {seg}::String AS seg, product_id::String AS p,
       3 * countIf(event_type = 'purchase') + countIf(event_type = 'add_to_cart') + 0.05 * countIf(event_type = 'view') AS s
FROM shop.history_events WHERE ts::DateTime64(3) < '{until}' GROUP BY seg, p"""
COBUY = """
SELECT a.p AS p, b.q AS q, count() AS n
FROM (SELECT DISTINCT session_id::String AS s, product_id::String AS p FROM shop.history_events
      WHERE event_type = 'view' AND ts::DateTime64(3) < '{until}') AS a
INNER JOIN (SELECT DISTINCT session_id::String AS s, product_id::String AS q FROM shop.history_events
            WHERE event_type = 'purchase' AND ts::DateTime64(3) < '{until}') AS b ON a.s = b.s
WHERE a.p != b.q GROUP BY p, q ORDER BY n DESC LIMIT 20 BY p"""


def cached(name, build):
    path = rec.CACHE / f"exp_{name}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    obj = build()
    path.write_bytes(pickle.dumps(obj, protocol=4))
    return obj


def segment_lift(seg: str, items) -> dict[str, np.ndarray]:
    """Por región o dispositivo: log(cuota del producto en ese segmento / cuota global), suavizado."""
    rows = cached(f"segpop_{seg}", lambda: rec.query(SEGMENT_POP.format(seg=seg, until=UNTIL), timeout=600))
    names = sorted({r["seg"] for r in rows})
    m = np.zeros((len(names), items.n))
    for r in rows:
        i = model.PRODUCTS.idx.get(r["p"])
        if i is not None:
            m[names.index(r["seg"]), i] = float(r["s"])
    glob = m.sum(0) / m.sum()
    prior = 50  # pseudo-eventos repartidos según la cuota global: evita ratios extremos con pocos datos
    share = (m + prior * glob) / (m.sum(1, keepdims=True) + prior)
    return {name: np.log((share[k] + 1e-9) / (glob + 1e-9)).astype(np.float32) for k, name in enumerate(names)}


class Ctx:
    def __init__(self):
        t0 = time.time()
        product = {r["product_id"]: r for r in cached("catalog", lambda: rec.query(rec.CATALOG.format(updates="")))}
        pop14 = cached("pop14", lambda: rec.hist_popularity(UNTIL))
        self.state = rec.make_state(product, set(product), pop14, {})
        self.items = it = self.state.items
        self.coviews = rec.load_coviews(UNTIL)
        self.cov_idx = model.index_coviews(self.coviews)
        self.users = self._users("until", f"AND e.ts::DateTime64(3) < '{UNTIL}'")
        self.users_recent = self._users("recent14", f"AND e.ts::DateTime64(3) < '{UNTIL}' AND e.ts::DateTime64(3) >= '{CUT - dt.timedelta(14)} 00:00:00'")
        self.user_price = {r["u"]: float(r["m"]) for r in cached("user_price", lambda: rec.query(USER_PRICE.format(until=UNTIL), timeout=600))}
        self.region, self.device = segment_lift("region", it), segment_lift("device", it)
        cobuy = cached("cobuy", lambda: rec.query(COBUY.format(until=UNTIL), timeout=600))
        cb: dict[str, list[str]] = {}
        for r in cobuy:
            cb.setdefault(r["p"], []).append(r["q"])
        self.cobuy_idx = model.index_coviews(cb)
        pop3 = cached("pop3", lambda: rec.score_counts(rec.query(rec.POPULARITY.format(table="shop.history_events", where=rec.window(3, UNTIL)), timeout=300)))
        p14 = np.array([pop14.get(p, 0.0) for p in model.PRODUCTS.names])
        p3 = np.array([pop3.get(p, 0.0) for p in model.PRODUCTS.names])
        rank3 = np.empty(it.n); rank3[np.argsort(-p3, kind="stable")] = np.arange(it.n)
        rank_cat = np.empty(it.n); seen = np.zeros(it.ncat, int)
        for i in np.argsort(-it.pop, kind="stable"):
            seen[it.cat[i]] += 1; rank_cat[i] = seen[it.cat[i]]
        self.static = np.stack([1 / np.sqrt(rank_cat), 1 - rank3 / it.n,
                                np.log(((p3 + 1) / (p3.sum() + it.n)) / ((p14 + 1) / (p14.sum() + it.n)))], 1).astype(np.float32)
        self.sessions = cached("sessions", self._sessions)
        print(f"contexto listo ({time.time() - t0:.0f}s): {len(self.sessions)} carruseles")

    def _users(self, name, cond):
        users = model.UserStore()
        path = rec.CACHE / f"exp_users_{name}.pkl"
        if path.exists():
            users.load_file(path)
        else:
            users.load(rec.query, "shop.history_events", cond=cond)
            users.save(path)
        return users

    def _sessions(self):
        spans = [(CUT + dt.timedelta(d), CUT + dt.timedelta(d + 1)) for d in range(14)]
        with ThreadPoolExecutor(4) as pool:
            parts = pool.map(lambda ab: rec.query(SESSIONS.format(a=ab[0], b=ab[1], pct=NEG_PCT), timeout=600), spans)
            out = [dict(r, week=i // 7) for i, part in enumerate(parts) for r in part]
        return [s for s in out if len(s["shown"]) == 10 and all(p in model.PRODUCTS.idx for p in s["shown"])]

    def features(self, s: dict) -> np.ndarray:
        """Todas las señales candidatas (BASE + EXTRA) para un carrusel."""
        it = self.items
        views = [{"product_id": p} for p in s["rv"]]
        F = np.zeros((it.n, len(ALL)), np.float32)
        F[:, :model.K] = B = model.featurize(s["u"], views, it, self.users, self.cov_idx)
        F[:, COL["is_home"]] = 0.0 if views else 1.0
        F[:, COL["aff_x_pop"]] = B[:, 0] * B[:, model.FI["pop_sub"]]
        F[:, COL["cart_abandoned"]] = (B[:, 5] > 0) & (B[:, 6] == 0)
        F[:, COL["pop_cat"]] = self.static[:, 0]
        m = self.user_price.get(s["u"])
        if m is not None:
            d = B[:, model.FI["log_price"]] - m
            F[:, COL["price_diff"]], F[:, COL["price_absdiff"]] = d, np.abs(d)
        if s["region"] in self.region:
            F[:, COL["region_lift"]] = self.region[s["region"]]
        if s["device"] in self.device:
            F[:, COL["device_lift"]] = self.device[s["device"]]
        r = self.users_recent.row.get(s["u"])
        if r is not None:
            sc = self.users_recent.sub[r].astype(np.float32)
            if sc.sum() > 0:
                F[:, COL["sub_share_recent"]] = sc[it.sub] / sc.sum()
                F[:, COL["sub_log_recent"]] = np.log1p(sc[it.sub])
            for i, packed in self.users_recent.prod.get(r, {}).items():
                F[i, COL["p_recent"]] = math.log1p((packed & model.MASK) + 3 * ((packed >> 20) & model.MASK))
        F[:, COL["pop_recent"]], F[:, COL["trend"]] = self.static[:, 1], self.static[:, 2]
        idx = [model.PRODUCTS.idx[p] for p in s["rv"] if p in model.PRODUCTS.idx]
        for k, p in enumerate(reversed(idx[-10:])):
            for rank, q in enumerate(self.cobuy_idx.get(p, ())):
                F[q, COL["cobuy"]] += 0.7 ** k / (rank + 1)
        return F


def build_train(ctx: Ctx):
    """Una fila por producto mostrado (semanas de entrenar y probar), con todas las señales."""
    def build():
        X, slot, buy, click, weight, week = [], [], [], [], [], []
        for s in ctx.sessions:
            idx = [model.PRODUCTS.idx[p] for p in s["shown"]]
            X.append(ctx.features(s)[idx])
            slot.append(np.arange(10)); week.append(np.full(10, s["week"]))
            buy.append([p == s["purchased"] for p in s["shown"]]); click.append([p == s["clicked"] for p in s["shown"]])
            weight.append(np.full(10, 1.0 if s["clicked"] else 100 / NEG_PCT))
        c = np.concatenate
        return c(X), c(slot), c(buy).astype(bool), c(click).astype(bool), c(weight), c(week)
    return cached("train_matrix", build)


def fit_lr(X, slot, y, w, cols):
    Xc = X[:, cols]
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-9
    Z = np.hstack([(Xc - mu) / sd, np.eye(10, dtype=np.float32)[slot][:, 1:]])
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(Z, y, sample_weight=w)
    coef = (lr.coef_[0][:len(cols)] / sd).astype(np.float32)
    return lambda F: F[:, cols] @ coef, coef


def fit_gbm(X, slot, y, w, cols, **kw):
    Z = np.hstack([X[:, cols], slot[:, None].astype(np.float32)])
    params = dict(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, random_state=0)
    params.update(kw)
    gbm = HistGradientBoostingClassifier(**params).fit(Z, y, sample_weight=w)
    def score(F):
        Zs = np.hstack([F[:, cols], np.zeros((len(F), 1), np.float32)])  # posición fija = la primera
        return gbm.decision_function(Zs)
    return score, gbm


def evaluate(ctx: Ctx, scorer, v1: bool = False) -> dict:
    """scorer(F, sesión) -> puntuación de cada producto. Métrica fija sobre la semana de prueba."""
    ndcg, hit, home, known = [], [], [], []
    for s in ctx.sessions:
        if s["week"] != 1 or not s["purchased"]:
            continue
        if v1:
            top = rec.recommend_v1({"recent_views": [{"product_id": p} for p in s["rv"]]}, ctx.state, ctx.coviews)
            rank = top.index(s["purchased"]) + 1 if s["purchased"] in top else 99
        else:
            score = scorer(ctx.features(s), s)
            rank = 1 + int((score > score[model.PRODUCTS.idx[s["purchased"]]]).sum())
        ndcg.append(1 / math.log2(1 + rank) if rank <= 10 else 0.0); hit.append(rank <= 10)
        home.append(s["page"] == "home"); known.append(s["u"] in ctx.users.row)
    ndcg, hit, home = np.array(ndcg), np.array(hit), np.array(home)
    return {"n": len(ndcg), "NDCG@10": float(ndcg.mean()), "Recall@10": float(hit.mean()),
            "NDCG home": float(ndcg[home].mean()), "NDCG resto": float(ndcg[~home].mean())}


def cols_of(names):
    return [COL[n] for n in names]


def show(name, r, base=None):
    delta = f"  ({100 * (r['NDCG@10'] / base - 1):+.1f} % vs base)" if base else ""
    print(f"{name:<44} NDCG@10 {r['NDCG@10']:.4f}  Recall@10 {r['Recall@10']:.4f}  home {r['NDCG home']:.3f}  resto {r['NDCG resto']:.3f}{delta}", flush=True)


if __name__ == "__main__":
    ctx = Ctx()
    X, slot, buy, click, weight, week = build_train(ctx)
    A = week == 0
    print(f"entrenamiento: {int(A.sum())} ejemplos, {int(buy[A].sum())} compras | prueba: {int(buy[~A].sum())} compras")
    pop = np.zeros(len(ALL), np.float32); pop[COL["pop_glob"]] = 1
    show("más vendidos (baseline del reto)", evaluate(ctx, lambda F, s: F @ pop))
    show("v1 reglas", evaluate(ctx, None, v1=True))
    w_prod = np.zeros(len(ALL), np.float32); w_prod[:model.K] = model.load_weights()
    show("v2 en producción (pesos guardados)", evaluate(ctx, lambda F, s: F @ w_prod))
    f, _ = fit_lr(X[A], slot[A], buy[A], weight[A], cols_of(BASE))
    base = evaluate(ctx, lambda F, s: f(F))
    show("LÍNEA BASE: v2 LR, 16 señales, semana A", base)
