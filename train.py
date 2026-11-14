"""Entrena y evalúa offline el modelo v2 con shop.history_sessions. Solo lee de RawTree.

    python train.py            # evalúa (entrena con una semana, prueba con la siguiente) y guarda pesos
    python train.py --no-save  # solo evalúa

Diseño, para no hacer trampa con el tiempo:

    |------- 46 días -------|--- semana A ---|--- semana B ---|
       memoria de usuarios      ENTRENAR          PROBAR
       popularidad, co-vistas

La memoria de usuarios, la popularidad y las co-vistas solo usan datos ANTERIORES a las dos semanas.
Cada producto mostrado en un carrusel antiguo es un ejemplo: sus señales (model.featurize, las mismas
que en vivo) y si se compró o no. Una regresión logística aprende cuánto pesa cada señal. La posición
en el carrusel entra como señal al entrenar (arriba se compra más, sea cual sea el producto) y se
descarta al servir.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import model
import recommender as rec

NEG_PCT = 30   # % de carruseles sin clic que se usan (los que tienen clic se usan todos)
SESSIONS = """
SELECT user_id::String AS u, context.page::String AS page,
       arrayMap(x -> x.product_id::String, recent_views::Array(JSON)) AS rv,
       shown::Array(String) AS shown, ifNull(clicked::String, '') AS clicked, ifNull(purchased::String, '') AS purchased,
       cityHash64(request_id::String) % 100 < {pct} AS sampled
FROM shop.history_sessions
WHERE ts::DateTime64(3) >= '{a}' AND ts::DateTime64(3) < '{b}'
  AND (clicked != '' OR cityHash64(request_id::String) % 100 < {pct})"""


def download_sessions(start: dt.date, days: int) -> list[dict]:
    spans = [(start + dt.timedelta(d), start + dt.timedelta(d + 1)) for d in range(days)]
    with ThreadPoolExecutor(4) as pool:
        parts = pool.map(lambda ab: rec.query(SESSIONS.format(a=ab[0], b=ab[1], pct=NEG_PCT), timeout=600), spans)
        return [dict(r, week=i // 7) for i, part in enumerate(parts) for r in part]


def build_examples(sessions, items, users, cov_idx):
    """Una fila por producto mostrado: señales, posición, clic, compra, peso y nº de carrusel."""
    X, slot, click, buy, weight, group, week = [], [], [], [], [], [], []
    kept = []
    for g, s in enumerate(sessions):
        idx = [model.PRODUCTS.idx.get(p) for p in s["shown"]]
        if len(idx) != 10 or None in idx:
            continue
        F = model.featurize(s["u"], [{"product_id": p} for p in s["rv"]], items, users, cov_idx)
        X.append(F[idx])
        slot.append(np.arange(10))
        click.append([p == s["clicked"] for p in s["shown"]])
        buy.append([p == s["purchased"] for p in s["shown"]])
        # los carruseles sin clic se muestrearon: cada uno representa a 100/NEG_PCT
        weight.append(np.full(10, 1.0 if s["sampled"] else 0.0) * (100 / NEG_PCT) if not s["clicked"] else np.ones(10))
        group.append(np.full(10, len(kept)))
        week.append(np.full(10, s["week"]))
        kept.append(s)
    cat = lambda a: np.concatenate(a)
    return (np.concatenate(X), cat(slot), cat(click).astype(bool), cat(buy).astype(bool), cat(weight), cat(group),
            cat(week), kept)


def fit(X, slot, y, weight) -> np.ndarray:
    """Regresión logística. Devuelve un peso por señal (en las unidades originales de cada señal)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([(X - mu) / sd, np.eye(10, dtype=np.float32)[slot][:, 1:]])
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(Z, y, sample_weight=weight)
    return (lr.coef_[0][:model.K] / sd).astype(np.float32)


def evaluate(sessions, items, users, cov_idx, state, coviews, policies: dict[str, np.ndarray | None]) -> dict:
    """Para los carruseles con compra: ¿está el producto comprado en el top 10 de cada política?
    Y entre los 10 que se mostraron, ¿en qué puesto lo habría colocado (1 = primero)?"""
    out = {name: {"hit": [], "rr": [], "home": [], "known": []} for name in policies}
    for s in sessions:
        if not s["purchased"]:
            continue
        target = model.PRODUCTS.idx[s["purchased"]]
        shown = [model.PRODUCTS.idx[p] for p in s["shown"]]
        req = {"user_id": s["u"], "recent_views": [{"product_id": p} for p in s["rv"]]}
        F = model.featurize(s["u"], req["recent_views"], items, users, cov_idx)
        for name, w in policies.items():
            if name == "v1 (reglas)":
                top = rec.recommend_v1(req, state, coviews)
                rr = np.nan
            else:
                score = F @ w
                top = model.top10(score, items)
                rr = 1 / (1 + int((score[shown] > score[target]).sum()))
            o = out[name]
            o["hit"].append(s["purchased"] in top), o["rr"].append(rr)
            o["home"].append(s["page"] == "home"), o["known"].append(s["u"] in users.row)
    res = {}
    for name, o in out.items():
        hit, home, known = np.array(o["hit"]), np.array(o["home"]), np.array(o["known"])
        res[name] = {"compras": len(hit), "hit@10 %": 100 * hit.mean(), "MRR entre mostrados": float(np.nanmean(o["rr"])),
                     "hit@10 home %": 100 * hit[home].mean(), "hit@10 resto %": 100 * hit[~home].mean()}
    return res


def history_examples():
    """Ejemplos del histórico con el corte temporal: 46 días de apoyo, luego semana A y semana B."""
    t0 = time.time()
    end = dt.datetime.fromisoformat(rec.query("SELECT toString(max(ts::DateTime64(0))) AS d FROM shop.history_sessions")[0]["d"]).date()
    cut = end - dt.timedelta(days=13)   # semana A = [cut, cut+7), semana B = [cut+7, cut+14)
    until = f"{cut} 00:00:00"
    print(f"memoria/popularidad/co-vistas: antes de {cut} | entrenar: {cut} → {cut + dt.timedelta(7)} | probar: → {cut + dt.timedelta(14)}")

    # Catálogo, popularidad y co-vistas tal como se conocían en la fecha de corte
    rows = rec.query(rec.CATALOG.format(updates=""))
    product = {r["product_id"]: r for r in rows}
    state = rec.make_state(product, set(product), rec.hist_popularity(until), {})
    items = state.items
    coviews = rec.load_coviews(until)
    cov_idx = model.index_coviews(coviews)

    users = model.UserStore()
    path = rec.CACHE / f"users_until_{cut}.pkl"
    if path.exists():
        users.load_file(path)
    else:
        print("cargando memoria de usuarios hasta la fecha de corte (solo la primera vez)...")
        users.load(rec.query, "shop.history_events", cond=f"AND e.ts::DateTime64(3) < '{until}'")
        users.save(path)
    print(f"{len(users.row)} usuarios en memoria ({time.time() - t0:.0f}s)")

    sessions = download_sessions(cut, 14)
    X, slot, click, buy, weight, group, week, kept = build_examples(sessions, items, users, cov_idx)
    print(f"{len(kept)} carruseles, {len(X)} ejemplos, {int(click.sum())} clics, {int(buy.sum())} compras ({time.time() - t0:.0f}s)")
    return dict(X=X, slot=slot, click=click, buy=buy, weight=weight, week=week, kept=kept, items=items, users=users,
                cov_idx=cov_idx, state=state, coviews=coviews, cut=cut)


def main() -> None:
    t0 = time.time()
    h = history_examples()
    X, slot, click, buy, weight, week, kept = (h[k] for k in ("X", "slot", "click", "buy", "weight", "week", "kept"))
    items, users, cov_idx, state, coviews, cut = (h[k] for k in ("items", "users", "cov_idx", "state", "coviews", "cut"))

    A = week == 0
    w_buy = fit(X[A], slot[A], buy[A], weight[A])
    w_click = fit(X[A], slot[A], click[A], weight[A])
    pop_only = np.zeros(model.K, np.float32)
    pop_only[model.FI["pop_glob"]] = 1
    test = [s for s in kept if s["week"] == 1]
    res = evaluate(test, items, users, cov_idx, state, coviews, {
        "más vendidos (baseline)": pop_only, "v1 (reglas)": None,
        "v2 entrenado con compras": w_buy, "v2 entrenado con clics": w_click})
    print(f"\nEvaluación en la semana de prueba ({time.time() - t0:.0f}s)")
    cols = list(next(iter(res.values())))
    print(f"{'':<28}" + "".join(f"{c:>22}" for c in cols))
    for name, r in res.items():
        print(f"{name:<28}" + "".join(f"{r[c]:>22.2f}" for c in cols))

    target = "compras" if res["v2 entrenado con compras"]["hit@10 %"] >= res["v2 entrenado con clics"]["hit@10 %"] else "clics"
    final = fit(X, slot, buy if target == "compras" else click, weight)   # pesos finales: las dos semanas
    print(f"\nPesos finales (entrenado con {target}, dos semanas):")
    for name, w in sorted(zip(model.FEATURES, final), key=lambda t: -abs(t[1])):
        print(f"  {name:<16}{w:>9.3f}")
    if "--no-save" not in sys.argv:
        model.WEIGHTS_FILE.write_text(json.dumps({
            "trained": dt.datetime.now().isoformat(timespec="seconds"), "target": target,
            "train_window": [str(cut), str(cut + dt.timedelta(14))],
            "offline_eval_test_week": res,
            "weights": {name: round(float(w), 5) for name, w in zip(model.FEATURES, final)}}, indent=2, ensure_ascii=False))
        print(f"\nguardado {model.WEIGHTS_FILE.name}")


if __name__ == "__main__":
    main()
