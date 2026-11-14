"""Reentrena los pesos de v2 con NUESTRO feedback de hoy (team_NN.feedback). Solo lee de RawTree.

    python train_live.py            # evalúa y, si mejora, guarda model_weights_live.json
    python train_live.py --no-save

Por qué: el simulador en vivo no se comporta igual que el histórico (p. ej. lo ya comprado no se
recompra). Cómo, sin hacer trampa:

  * Las señales de cada respuesta se calculan con lo que se sabía EN ESE MOMENTO: historial de 60 días
    + los eventos de hoy anteriores a la request (se "reproduce" el día en orden).
  * Hay pocas compras de hoy (cientos), así que no se tiran los pesos actuales (del histórico): se
    aprenden CORRECCIONES sobre ellos. Una corrección solo se aleja de cero si los datos de hoy la justifican
    (regularización), y la fuerza de esa regularización se elige con una parte de validación.
  * Split temporal de hoy: 60 % entrenar | 15 % validar | 25 % probar (lo último del día).
  * Métrica: sabiendo la posición en que se mostró cada producto, ¿qué pesos predicen mejor qué se
    compra y qué se clica? (log-loss y AUC en la parte de prueba; menos log-loss y más AUC es mejor).
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

import model
import recommender as rec

LAG_MS = 30_000  # en producción los eventos entran en la memoria con ~0-60 s de retraso
def _arg(flag: str) -> str | None:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None


# --v6            : aprende también las señales nuevas (mf, cart_no_buy, viewed_no_cart) -> model_weights_v6.json
# --out v7        : lo mismo, pero guarda en model_weights_v7.json
# --drop a,b      : señales que se dejan fuera (peso 0), p. ej. --drop cart_no_buy,viewed_no_cart
NAME = _arg("--out") or ("v6" if "--v6" in sys.argv else None)
V6 = NAME is not None
DROP = [model.FI[n] for n in (_arg("--drop") or "").split(",") if n]
# --cap is_new_prod=0.4 : tope al peso final de una señal. Para "producto nuevo" se usa 0,4 = log(1,5), el
# aumento de compra medido en vivo a igualdad de puesto. Sin tope, el peso aprendido (~1,2) llenaba de
# productos nuevos el 32 % de los huecos (su cuota real en ventas es ~5 %): el mismo exceso que hundió a v5.
CAPS = {n: float(v) for n, v in (kv.split("=") for kv in (_arg("--cap") or "").split(",") if kv)}
OUT = model.WEIGHTS_FILE.with_name(f"model_weights_{NAME}.json" if NAME else "model_weights_live.json")
LIVE = """
SELECT f.request_id::String AS rid, r.user_id::String AS u, toUnixTimestamp64Milli(r.ts::DateTime64(3)) AS ms,
       arrayMap(x -> x.product_id::String, r.recent_views::Array(JSON)) AS rv,
       f.shown::Array(String) AS shown, f.clicked::String AS clicked, f.purchased::String AS purchased
FROM {team}.feedback AS f INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
WHERE f.served_by::String = 'team' AND cityHash64(f.request_id::String) % 4 = {i}"""
EXPLORED = "SELECT request_id::String AS rid FROM {team}.scratch WHERE feature_set::String = 'explore_v1'"
"""Requests respondidas por la variante de exploración (listas sorteadas, no las 10 mejores)."""
EVENTS = """
SELECT user_id::String AS u, product_id::String AS p, event_type::String AS t, toUnixTimestamp64Milli(ts::DateTime64(3)) AS ms
FROM shop.events
WHERE ts::DateTime64(3) >= toDateTime64('{a}', 3) AND ts::DateTime64(3) < toDateTime64('{b}', 3)
ORDER BY ts"""


def live_examples():
    """Una fila por producto mostrado hoy, con las señales tal como eran en el momento de la request."""
    mf = model.load_mf()
    state = rec.build_state(rec.hist_popularity(), mf)
    items = state.items
    users = model.UserStore()
    users.load_file(rec.CACHE / "users_history_v2.pkl")
    users.init_mf(mf)
    cov_idx = model.index_coviews(rec.load_coviews())
    reqs = sorted((r for i in range(4) for r in rec.query(LIVE.format(team=rec.TEAM, i=i), timeout=300)), key=lambda r: int(r["ms"]))
    reqs = [r for r in reqs if len(r["shown"]) == 10 and all(p in model.PRODUCTS.idx for p in r["shown"])]

    start = dt.datetime.fromisoformat(rec.query("SELECT toString(toStartOfHour(min(ts::DateTime64(3)))) AS d FROM shop.events")[0]["d"])
    hour, events, e = start, [], 0
    X, slot, click, buy, when, explored = [], [], [], [], [], []
    try:
        explore_ids = {r["rid"] for r in rec.query(EXPLORED.format(team=rec.TEAM), timeout=120)}
    except Exception:   # aún no hay registros de exploración
        explore_ids = set()
    for r in reqs:
        limit = int(r["ms"]) - LAG_MS
        while True:  # incorporar a la memoria todos los eventos anteriores a esta request
            while e < len(events) and int(events[e]["ms"]) < limit:
                ev = events[e]
                users.add_event(ev["u"], ev["p"], ev["t"], items)
                e += 1
            if e < len(events) or hour.replace(tzinfo=dt.timezone.utc).timestamp() * 1000 > limit:
                break  # lo que queda (o la hora siguiente) es posterior a esta request
            events, e = rec.query(EVENTS.format(a=hour, b=hour + dt.timedelta(hours=1)), timeout=300), 0
            hour += dt.timedelta(hours=1)
        idx = [model.PRODUCTS.idx[p] for p in r["shown"]]
        X.append(model.featurize(r["u"], [{"product_id": p} for p in r["rv"]], items, users, cov_idx)[idx])
        slot.append(np.arange(10)), when.append(np.full(10, int(r["ms"])))
        explored.append(np.full(10, r.get("rid") in explore_ids))
        click.append([p == r["clicked"] for p in r["shown"]]), buy.append([p == r["purchased"] for p in r["shown"]])
        if r["purchased"]:  # lo comprado en nuestro carrusel también entra en la memoria
            users.add_event(r["u"], r["purchased"], "purchase", items)
    c = np.concatenate
    global EXPLORED_ROWS
    EXPLORED_ROWS = c(explored).astype(bool)   # filas que vienen de listas sorteadas (v7-explore)
    return c(X), c(slot), c(click).astype(bool), c(buy).astype(bool), c(when)


EXPLORED_ROWS = np.zeros(0, bool)


def onehot(slot):
    return np.eye(10, dtype=np.float32)[slot][:, 1:]


def fit_corrected(current, L, mask, C):
    """Pesos actuales (histórico) + corrección aprendida con los datos de hoy (filas `mask`).

    Modelo: compra ~ posición + b × (puntuación con los pesos actuales) + corrección por señal.
    La regularización (C pequeño = fuerte) empuja cada corrección hacia cero."""
    X = L["X"][mask]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    s = X @ current
    big = 20.0  # la columna de la puntuación actual va muy escalada para que la regularización casi no la toque
    Z = np.hstack([((s - s.mean()) / s.std() * big)[:, None], (X - mu) / sd, onehot(L["slot"][mask])])
    lr = LogisticRegression(C=C, max_iter=5000, tol=1e-6).fit(Z, L["buy"][mask])
    b = lr.coef_[0][0] * big / s.std()
    base = (b * current).astype(np.float32)
    return (base + lr.coef_[0][1:model.K + 1] / sd).astype(np.float32), base


def judge(w, L, fit_rows, test_rows, target):
    """Puntuación del producto + posición -> probabilidad. Se calibra en fit_rows y se mide en test_rows."""
    Z = np.hstack([(L["X"] @ w)[:, None], onehot(L["slot"])])
    lr = LogisticRegression(C=10.0, max_iter=2000).fit(Z[fit_rows], L[target][fit_rows])
    p = lr.predict_proba(Z[test_rows])[:, 1]
    return log_loss(L[target][test_rows], p), roc_auc_score(L[target][test_rows], p)


def main() -> None:
    t0 = time.time()
    X, slot, click, buy, when = live_examples()
    if not V6:  # la versión v4 solo usa las 16 señales originales
        X[:, model.FI["mf"]:] = 0
    X[:, DROP] = 0
    L = {"X": X, "slot": slot, "click": click, "buy": buy}
    q60, q75 = np.quantile(when, [0.60, 0.75])
    tr, va, te = when < q60, (when >= q60) & (when < q75), when >= q75
    print(f"feedback de hoy: {len(X) // 10} respuestas | compras: entrenar {int(buy[tr].sum())}, validar {int(buy[va].sum())}, "
          f"probar {int(buy[te].sum())} | clics en prueba: {int(click[te].sum())} ({time.time() - t0:.0f}s)")
    current = model.load_weights()
    # 1) elegir la fuerza de la regularización con la parte de validación
    best_C, best = None, 9.0
    for C in (0.001, 0.003, 0.01, 0.1, 1.0, 10.0):
        w, _ = fit_corrected(current, L, tr, C)
        ll, auc = judge(w, L, tr, va, "buy")
        print(f"  validación C={C:<7} log-loss compra {ll:.5f}  AUC {auc:.4f}")
        if ll < best:
            best_C, best = C, ll
    # 2) medir una vez en la parte de prueba, entrenando con entrenar+validar
    cands = {"pesos actuales (solo histórico)": current,
             f"corregidos con hoy (C={best_C})": fit_corrected(current, L, tr | va, best_C)[0]}
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(np.hstack([(X[tr | va] - X.mean(0)) / (X.std(0) + 1e-9), onehot(slot[tr | va])]), buy[tr | va])
    cands["solo con datos de hoy"] = (lr.coef_[0][:model.K] / (X.std(0) + 1e-9)).astype(np.float32)
    v4_file = model.WEIGHTS_FILE.with_name("model_weights_live.json")
    if V6 and v4_file.exists():  # ojo: los pesos ya en producción se entrenaron con parte de este tramo: salen favorecidos
        cands["v4 en producción"] = model.load_weights(v4_file)
        for f in sorted(model.WEIGHTS_FILE.parent.glob("model_weights_v*.json")):
            if f != OUT:
                cands[f"{f.stem.removeprefix('model_weights_')} en producción"] = model.load_weights(f)
    print(f"\nPRUEBA (último 25 % del día, {int(buy[te].sum())} compras, {int(click[te].sum())} clics):")
    res = {}
    for name, w in cands.items():
        (llb, aucb), (llc, aucc) = judge(w, L, tr | va, te, "buy"), judge(w, L, tr | va, te, "click")
        res[name] = {"logloss_compra": llb, "auc_compra": aucb, "logloss_clic": llc, "auc_clic": aucc}
        print(f"  {name:<34} compra: log-loss {llb:.5f} AUC {aucb:.4f} | clic: log-loss {llc:.5f} AUC {aucc:.4f}")
    # Solo las listas SORTEADAS: ahí el producto y su puesto no los eligió el modelo, así que es la medida
    # más limpia de si los pesos ordenan bien (sin el sesgo de "solo vemos lo que ya enseñamos").
    ex = EXPLORED_ROWS if len(EXPLORED_ROWS) == len(X) else np.zeros(len(X), bool)
    print(f"\nexploración: {int(ex.sum()) // 10} respuestas sorteadas en total, {int(buy[ex].sum())} compras, {int(click[ex].sum())} clics")
    if buy[te & ex].sum() >= 10:
        print(f"SOLO EXPLORACIÓN dentro de la prueba ({int(buy[te & ex].sum())} compras, {int(click[te & ex].sum())} clics):")
        for name, w in cands.items():
            (llb, aucb), (llc, aucc) = judge(w, L, tr | va, te & ex, "buy"), judge(w, L, tr | va, te & ex, "click")
            res[name].update(explore_auc_compra=aucb, explore_auc_clic=aucc)
            print(f"  {name:<34} compra: log-loss {llb:.5f} AUC {aucb:.4f} | clic: log-loss {llc:.5f} AUC {aucc:.4f}")
    if click[ex].sum() >= 100:   # efecto de la posición medido con listas sorteadas (clics: hay más que compras)
        Z = np.hstack([(X[ex] @ current)[:, None], onehot(slot[ex])])
        lr = LogisticRegression(C=10.0, max_iter=2000).fit(Z, click[ex])
        print("efecto de la posición en vivo (clics, puesto 1 = 1,00): "
              + "  ".join(f"{k + 1}:{np.exp(v):.2f}" for k, v in enumerate(np.concatenate([[0.0], lr.coef_[0][1:]]))))

    # 3) pesos finales: histórico + corrección con TODO el feedback de hoy
    final, base = fit_corrected(current, L, np.ones(len(X), bool), best_C)
    for name, cap in CAPS.items():
        if final[model.FI[name]] > cap:
            print(f"tope: {name} {final[model.FI[name]]:.3f} -> {cap}")
            final[model.FI[name]] = cap
    print("\nPesos (histórico → corregido con hoy):")
    for name, a, b in sorted(zip(model.FEATURES, base, final), key=lambda t: -abs(t[2] - t[1])):
        print(f"  {name:<16}{a:>9.3f} → {b:>8.3f}   ({b - a:+.3f})")
    name_new = f"corregidos con hoy (C={best_C})"
    better = (res[name_new]["logloss_compra"] < res["pesos actuales (solo histórico)"]["logloss_compra"]
              and res[name_new]["logloss_clic"] < res["pesos actuales (solo histórico)"]["logloss_clic"])
    print(f"\n¿Mejoran a los actuales en la prueba (compra y clic)? {'SÍ' if better else 'NO'}")
    if (better or V6) and "--no-save" not in sys.argv:
        OUT.write_text(json.dumps({"trained": dt.datetime.now().isoformat(timespec="seconds"), "C": best_C, "caps": CAPS,
                                   "live_responses": len(X) // 10, "live_purchases": int(buy.sum()), "test": res,
                                   "weights": {n: round(float(v), 5) for n, v in zip(model.FEATURES, final)}}, indent=2, ensure_ascii=False))
        print(f"guardado {OUT.name}")


if __name__ == "__main__":
    main()
