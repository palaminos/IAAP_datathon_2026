"""Ajusta el peso de "producto nuevo" (is_new_prod) de una versión para que enseñe productos nuevos tanto
como la versión de referencia (por defecto v8, ~6 % de los huecos). Solo lee de RawTree; escribe el JSON.

    python analysis/calibrate_new.py v15            # ajusta model_weights_v15.json y, si existe, _v15n.json
    python analysis/calibrate_new.py v15 --ref v8 --dry
    python analysis/calibrate_new.py v26 --factor 2      # el doble de productos nuevos que v8

Por qué: el peso que aprende el entrenamiento para is_new_prod sale muy alto (2-3) porque los productos
nuevos no tienen historial y el resto de señales los infravalora; usado tal cual llenaría de productos
nuevos el 30-50 % de los huecos, muy lejos de lo que se ha probado en vivo. Hasta que una prueba en vivo
diga otra cosa, todas las versiones enseñan la misma proporción de productos nuevos que v8.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model  # noqa: E402
import recommender as rec  # noqa: E402

name = sys.argv[1]
ref = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else "v8"
factor = float(sys.argv[sys.argv.index("--factor") + 1]) if "--factor" in sys.argv else 1.0   # 2.0 = el doble de exposición que la referencia
bg = rec.Background(rec.hist_popularity())
m = rec.load_models()
bg._load_users()
s, it = bg.state, bg.state.items
blocked = {p for p, last in rec.stock_changes().items() if last == "out_of_stock"}
reqs = rec.query("""SELECT request_id::String AS request_id, user_id::String AS user_id, recent_views, context, is_new_user FROM shop.requests
WHERE ts::DateTime64(3) > now() - INTERVAL 6 MINUTE LIMIT 1000""")
idx = model.PRODUCTS.idx
isnew = np.array([str(r.get("is_new_user")).lower() in ("true", "1") for r in reqs])
new = np.asarray(it.is_new).astype(bool)


def exposure(version: str, mask) -> float:
    lists = [rec.respond(r, version + "-x", s, blocked, bg.users, m)["items"] for r, k in zip(reqs, mask) if k]
    return 100 * float(np.mean([[new[idx[p]] for p in a] for a in lists]))


split = name + "n" in m.by_version
for key, mask, label in ([(name, ~isnew, "conocidos"), (name + "n", isnew, "nuevos")] if split else [(name, np.ones(len(reqs), bool), "todos")]):
    target = factor * exposure(ref, mask)
    lo, hi = -1.0, 3.0
    for _ in range(9):                      # bisección: la exposición crece con el peso
        mid = (lo + hi) / 2
        w = m.by_version[key].copy()
        w[model.FI["is_new_prod"]] = mid
        m.by_version[key] = w
        lo, hi = (mid, hi) if exposure(name, mask) < target else (lo, mid)
    val = round((lo + hi) / 2, 2)
    w = m.by_version[key].copy()
    w[model.FI["is_new_prod"]] = val
    m.by_version[key] = w
    print(f"{key} ({label}): {ref} enseña productos nuevos en el {target:.1f} % de los huecos -> is_new_prod = {val} ({exposure(name, mask):.1f} %)")
    if "--dry" not in sys.argv:
        path = model.WEIGHTS_FILE.with_name(f"model_weights_{key}.json")
        d = json.loads(path.read_text())
        d["learned_is_new_prod"] = d.get("learned_is_new_prod", d["weights"]["is_new_prod"])
        d["weights"]["is_new_prod"] = val
        d["caps"] = {"is_new_prod": val}
        path.write_text(json.dumps(d, indent=2, ensure_ascii=False))
