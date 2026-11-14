"""¿Qué aporta cada pieza de v6? Se entrena con el feedback de hoy quitando piezas y se mide en el último
25 % del día (mismo método que train_live.py). Solo lee.

    python analysis/v6_ablation.py     # desde la raíz del repo
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.simplefilter("ignore")
import model  # noqa: E402
import train_live as tl  # noqa: E402

X, slot, click, buy, when = tl.live_examples()
q60, q75 = np.quantile(when, [0.60, 0.75])
tr, va, te = when < q60, (when >= q60) & (when < q75), when >= q75
FI = model.FI
PIECES = {"usuarios parecidos": [FI["mf"], FI["mf_z"]], "carrito sin compra": [FI["cart_no_buy"]], "visto sin carrito": [FI["viewed_no_cart"]]}
NEW = [c for cols in PIECES.values() for c in cols]
current = model.load_weights()
print(f"{len(X) // 10} respuestas | prueba: {int(buy[te].sum())} compras, {int(click[te].sum())} clics\n")


def run(name, drop):
    Xd = X.copy()
    Xd[:, drop] = 0
    L = {"X": Xd, "slot": slot, "click": click, "buy": buy}
    best = min((0.001, 0.01, 0.1), key=lambda C: tl.judge(tl.fit_corrected(current, L, tr, C)[0], L, tr, va, "buy")[0])
    w = tl.fit_corrected(current, L, tr | va, best)[0]
    (llb, aucb), (llc, aucc) = tl.judge(w, L, tr | va, te, "buy"), tl.judge(w, L, tr | va, te, "click")
    print(f"  {name:<44} compra: AUC {aucb:.4f} log-loss {llb:.5f} | clic: AUC {aucc:.4f} log-loss {llc:.5f} | C={best}")
    return aucb, aucc


L0 = {"X": X, "slot": slot, "click": click, "buy": buy}
for name, f in (("v4 tal como está en producción", "model_weights_live.json"), ("v6 tal como está en producción", "model_weights_v6.json")):
    w = model.load_weights(model.WEIGHTS_FILE.with_name(f))
    (llb, aucb), (llc, aucc) = tl.judge(w, L0, tr | va, te, "buy"), tl.judge(w, L0, tr | va, te, "click")
    print(f"  {name:<44} compra: AUC {aucb:.4f} log-loss {llb:.5f} | clic: AUC {aucc:.4f} log-loss {llc:.5f}")
print()
run("16 señales antiguas, pesos reentrenados hoy", NEW)
for piece, cols in PIECES.items():
    run(f"  + solo '{piece}'", [c for c in NEW if c not in cols])
print()
run("TODO (v6)", [])
for piece, cols in PIECES.items():
    run(f"  v6 SIN '{piece}'", cols)
