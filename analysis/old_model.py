"""¿De qué depende la puntuación del sistema antiguo? (columna `propensity` de shop.history_sessions)

    python analysis/old_model.py      # baja una muestra de carruseles antiguos con su propensity (dos trozos, de uno en uno)

La propensity es la probabilidad con que el sistema antiguo eligió cada producto: log(propensity) = puntuación del
sistema antiguo + una constante por carrusel. Y esa puntuación predice muchísimo la compra (un producto de la sesión
con propensity > 0,3 en los puestos 1-3 se compra el 11 % de las veces). Aquí se calculan nuestras señales para una
muestra de carruseles de las dos últimas semanas del histórico (con la memoria de compradores anterior al 1 de
noviembre) y se mira cuánto de log(propensity) explican, dentro de cada carrusel.
Guarda cache/old_model_sample.npz para seguir analizando sin repetir nada.
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model  # noqa: E402
import recommender as rec  # noqa: E402

CUT = "2026-11-01"
SQL = """
SELECT user_id::String AS u, context.page::String AS page,
       arrayMap(x -> x.product_id::String, recent_views::Array(JSON)) AS rv, shown::Array(String) AS shown,
       propensity::Array(Float64) AS prop, ifNull(clicked::String, '') AS clicked, ifNull(purchased::String, '') AS purchased
FROM shop.history_sessions
WHERE ts::DateTime64(3) >= '{cut} 00:00:00' AND cityHash64(user_id::String) % 16 = {i}"""

t0 = time.time()
raw = rec.CACHE / "old_model_sessions.pkl"
if raw.exists():
    sessions = pickle.loads(raw.read_bytes())
else:
    sessions = [r for i in (0, 1) for r in rec.query(SQL.format(cut=CUT, i=i), timeout=600)]
    raw.write_bytes(pickle.dumps(sessions, protocol=4))
print(f"{len(sessions)} carruseles antiguos con propensity ({time.time() - t0:.0f}s)")

until = f"{CUT} 00:00:00"
product = {r["product_id"]: r for r in rec.query(rec.CATALOG.format(updates=""))}
mf = model.load_mf()
state = rec.make_state(product, set(product), rec.hist_popularity(until), {}, mf=mf, cf=model.load_cf(rec.CACHE / f"cf_sim_until_{CUT}.npz"))
items = state.items
users = model.UserStore()
users.load_file(rec.CACHE / f"users_until_{CUT}.pkl")
users.init_mf(mf)
cov_idx = model.index_coviews(rec.load_coviews(until))
print(f"estado a fecha {CUT}: {items.n} productos, {len(users.row)} compradores ({time.time() - t0:.0f}s)")

X, y, slot, sid, click, buy, known, page = [], [], [], [], [], [], [], []
idx = model.PRODUCTS.idx
k = 0
for s in sessions:
    if len(s["shown"]) != 10 or len(s["prop"]) != 10 or not all(p in idx for p in s["shown"]):
        continue
    F = model.featurize(s["u"], [{"product_id": p} for p in s["rv"]], items, users, cov_idx, True)
    ii = [idx[p] for p in s["shown"]]
    X.append(F[ii]); y.append(np.log(np.maximum(np.array(s["prop"], float), 1e-9)))
    slot.append(np.arange(10)); sid.append(np.full(10, k)); k += 1
    click.append([p == s["clicked"] for p in s["shown"]]); buy.append([p == s["purchased"] for p in s["shown"]])
    known.append(np.full(10, s["u"] in users.row)); page.append(np.full(10, s["page"]))
c = np.concatenate
np.savez_compressed(rec.CACHE / "old_model_sample.npz", X=c(X).astype(np.float32), y=c(y), slot=c(slot), sid=c(sid), click=c(click).astype(bool), buy=c(buy).astype(bool),
                    known=c(known), page=c(page), features=np.array(model.FEATURES))
print(f"{k} carruseles, {int(c(buy).sum())} compras, {int(c(click).sum())} clics -> cache/old_model_sample.npz ({time.time() - t0:.0f}s)")
