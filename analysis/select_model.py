"""Elige señales y regularización con VALIDACIÓN (20 % de usuarios de la semana de entrenamiento),
y mide una sola vez en la semana de prueba. Evita elegir mirando la prueba."""
import json
import math
import warnings
import zlib

import numpy as np
from sklearn.linear_model import LogisticRegression

from experiments import *  # noqa: F401,F403

warnings.simplefilter("ignore")
ctx = Ctx()
X, slot, buy, click, weight, week = build_train(ctx)
is_val_s = np.array([zlib.crc32(s["u"].encode()) % 5 == 0 for s in ctx.sessions])
val_rows = np.repeat(is_val_s, 10)
A = week == 0
KEPT = BASE + ["aff_x_pop", "cart_abandoned", "pop_cat", "price_diff", "price_absdiff"]
SETS = {
    "base16": BASE,
    "+interacciones": BASE + ["aff_x_pop", "cart_abandoned", "pop_cat"],
    "+precio": BASE + ["price_diff", "price_absdiff"],
    "+inter+precio": KEPT,
    "+inter+precio+region": KEPT + ["region_lift", "device_lift"],
    "+inter+precio+tendencia": KEPT + ["pop_recent", "trend"],
    "+inter+precio+reciente": KEPT + ["sub_share_recent", "sub_log_recent", "p_recent"],
    "+inter+precio+cobuy": KEPT + ["cobuy"],
    "+inter+precio+region+tendencia": KEPT + ["region_lift", "device_lift", "pop_recent", "trend"],
    "todas": ALL[:COL["is_home"]] + ALL[COL["is_home"] + 1:],
}
CS = (0.003, 0.01, 0.03, 0.1, 1.0)


def fit(names, C, mask):
    cols = cols_of(names)
    Xc = X[mask][:, cols]
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-9
    Z = np.hstack([(Xc - mu) / sd, np.eye(10, dtype=np.float32)[slot[mask]][:, 1:]])
    lr = LogisticRegression(C=C, max_iter=5000, tol=1e-6).fit(Z, buy[mask], sample_weight=weight[mask])
    return cols, (lr.coef_[0][:len(cols)] / sd).astype(np.float32)


def ndcgs(cands, sessions):
    out = {k: [] for k in cands}
    for s in sessions:
        F = ctx.features(s)
        t = model.PRODUCTS.idx[s["purchased"]]
        for k, (cols, w) in cands.items():
            sc = F[:, cols] @ w
            r = 1 + int((sc > sc[t]).sum())
            out[k].append(1 / math.log2(1 + r) if r <= 10 else 0.0)
    return {k: np.array(v) for k, v in out.items()}


# 1) selección en validación
cands = {(n, C): fit(names, C, A & ~val_rows) for n, names in SETS.items() for C in CS}
val = [s for s, v in zip(ctx.sessions, is_val_s) if v and s["week"] == 0 and s["purchased"]]
res = ndcgs(cands, val)
print(f"VALIDACIÓN ({len(val)} compras). NDCG@10 por conjunto de señales y C:")
print(f"{'':<34}" + "".join(f"{C:>9}" for C in CS))
for n in SETS:
    print(f"{n:<34}" + "".join(f"{res[(n, C)].mean():>9.4f}" for C in CS))
best = max(cands, key=lambda k: res[k].mean())
best_base = max(((("base16", C)) for C in CS), key=lambda k: res[k].mean())
print(f"\nmejor en validación: {best}  | mejor base16: {best_base}")

# 2) una sola medida en la semana de prueba, reentrenando con toda la semana A
final = {"base (C=1, receta actual)": fit(BASE, 1.0, A), f"base C={best_base[1]}": fit(BASE, best_base[1], A),
         f"elegido: {best[0]} C={best[1]}": fit(SETS[best[0]], best[1], A)}
test = [s for s in ctx.sessions if s["week"] == 1 and s["purchased"]]
tr = ndcgs(final, test)
ref = tr["base (C=1, receta actual)"]
rng = np.random.default_rng(0)
print(f"\nPRUEBA ({len(test)} compras):")
out = {}
for k, v in tr.items():
    d = v - ref
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    print(f"{k:<46} NDCG@10 {v.mean():.4f}  Recall@10 {(v > 0).mean():.4f}  Δ {d.mean():+.4f}  IC95 [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]")
    out[k] = {"ndcg": float(v.mean()), "recall": float((v > 0).mean())}
json.dump({"best": list(best), "features": SETS[best[0]], "test": out,
           "val": {f"{n}|{C}": float(res[(n, C)].mean()) for n in SETS for C in CS}}, open("cache/exp_selection.json", "w"), indent=1, ensure_ascii=False)
