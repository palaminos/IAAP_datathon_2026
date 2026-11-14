"""Prueba las ideas una a una contra la línea base (ver experiments.py). Conserva solo lo que mejora."""
import json
import sys
import time

import numpy as np

from experiments import *  # noqa: F401,F403

ctx = Ctx()
X, slot, buy, click, weight, week = build_train(ctx)
A = week == 0
log = []


def run(name, scorer, base=None):
    t = time.time()
    r = evaluate(ctx, scorer)
    show(name, r, base)
    log.append({"idea": name, **r, "seg": round(time.time() - t)})
    return r["NDCG@10"]


def lr(names, y=buy, mask=A):
    f, _ = fit_lr(X[mask], slot[mask], y[mask], weight[mask], cols_of(names))
    return lambda F, s: f(F)


def gbm(names, y=buy, **kw):
    f, _ = fit_gbm(X[A], slot[A], y[A], weight[A], cols_of(names), **kw)
    return lambda F, s: f(F)


base = run("BASE  LR 16 señales", lr(BASE))
kept = list(BASE)

# 1. Modelos por tipo de página (home no tiene vistas: otras señales mandan)
home_rows = X[:, COL["is_home"]] == 1
f_home, f_rest = lr(BASE, mask=A & home_rows), lr(BASE, mask=A & ~home_rows)
run("1. LR separada home / resto", lambda F, s: f_home(F, s) if s["page"] == "home" else f_rest(F, s), base)

# 2. Entrenar con clics (más ejemplos positivos) en vez de compras
run("2. LR entrenada con clics", lr(BASE, y=click), base)

# 3. Modelo no lineal (árboles): capta combinaciones de señales
g16 = run("3. GBM 16 señales", gbm(BASE), base)

# 4-9. Grupos de señales nuevas, de una en una sobre lo conservado (se mide con LR y con GBM)
groups = {
    "4. interacciones (afinidad×popularidad, carrito abandonado, pop. en categoría)": ["aff_x_pop", "cart_abandoned", "pop_cat"],
    "5. afinidad de precio del usuario": ["price_diff", "price_absdiff"],
    "6. popularidad por región y dispositivo": ["region_lift", "device_lift"],
    "7. afinidad reciente del usuario (14 días)": ["sub_share_recent", "sub_log_recent", "p_recent"],
    "8. popularidad reciente y tendencia (3 días)": ["pop_recent", "trend"],
    "9. vista→compra en la misma sesión (cobuy)": ["cobuy"],
}
best_lr, best_gbm = base, g16
for name, cols in groups.items():
    a = run(f"{name} | LR", lr(kept + cols), base)
    b = run(f"{name} | GBM", gbm(kept + cols), base)
    if a > best_lr * 1.002 or b > best_gbm * 1.002:
        kept += cols
        best_lr, best_gbm = max(best_lr, a), max(best_gbm, b)
        print(f"   → CONSERVADA. Señales: {len(kept)}")
    else:
        print("   → descartada")

json.dump({"kept": kept, "log": log, "base": base}, open("cache/exp_ideas.json", "w"), indent=1, ensure_ascii=False)
print("\nseñales conservadas:", kept)
