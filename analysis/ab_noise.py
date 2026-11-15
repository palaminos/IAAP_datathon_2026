"""Ruido real de la conversión esperada del leaderboard, medido con las diferencias entre versiones que corrieron a la vez.

    python analysis/ab_noise.py

Para cada pareja de versiones y cada ventana de 15 minutos en que corrieron las dos:
    d = conv_a - conv_b ;  z = (d - diferencia media de la pareja) / sqrt(1/n_a + 1/n_b)
La desviación típica de z es el ruido POR REQUEST (en puntos). Al comparar dentro de la misma ventana se quitan los
cambios de hora a hora. Se calcula para el total, para cold y para compradores conocidos.
"""
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import monitor as mo  # noqa: E402

start = dt.datetime(2026, 11, 14, 20, 15).timestamp()
end = int(time.time()) // 900 * 900
wins = []
t = start
while t + 900 <= end:
    vs, _, _ = mo.since(t, t + 900)
    wins.append({k: x for k, x in vs.items() if x["n"] >= 200 and x["cold_n"] >= 60})
    t += 900
pairs = {}
for w in wins:
    names = sorted(w)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            xa, xb = w[a], w[b]
            ka, kb = xa["n"] - xa["cold_n"], xb["n"] - xb["cold_n"]
            pairs.setdefault((a, b), []).append(((xa["conv"] - xb["conv"]), 1 / xa["n"] + 1 / xb["n"],
                                                 (xa["cold"] - xb["cold"]), 1 / xa["cold_n"] + 1 / xb["cold_n"],
                                                 (xa["known"] - xb["known"]), 1 / ka + 1 / kb))
out = {"total": [], "cold": [], "conocidos": []}
for (a, b), rows in pairs.items():
    r = np.array(rows)
    if len(r) < 3:
        continue
    for key, c in (("total", 0), ("cold", 2), ("conocidos", 4)):
        w = 1 / r[:, c + 1]
        mean = (r[:, c] * w).sum() / w.sum()
        z = (r[:, c] - mean) / np.sqrt(r[:, c + 1])
        out[key].extend(z * np.sqrt(len(r) / (len(r) - 1)))
print(f"ventanas de 15 min: {len(wins)} | parejas de versiones con 3 ventanas o más: {sum(len(v) >= 3 for v in pairs.values())}")
for key, z in out.items():
    z = np.array(z)
    print(f"   ruido por request, {key:<10}: {z.std():.2f} puntos (n = {len(z)}; mediana robusta {1.4826 * np.median(np.abs(z - np.median(z))):.2f})")
