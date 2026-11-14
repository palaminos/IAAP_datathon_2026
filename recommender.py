"""Recomendador del equipo (bucle pull → predict → push).

    python recommender.py              # producción: responde y envía (usad ./run_forever.sh)
    python recommender.py --dry-run    # prueba: hace todo igual pero NO envía nada
    python recommender.py --preflight  # comprueba que cada variante responde bien a 100 requests reales y sale

Dos modelos conviven y se reparten los compradores (test A/B, ver VARIANTS):
  * v1: reglas a mano: co-vistas + populares de la subcategoría/categoría de lo que acaba de ver.
  * v2: puntúa TODOS los productos con pesos aprendidos del histórico (train.py), usando además
        lo que cada usuario ha visto y comprado en 60 días y hoy (model.py).

En segundo plano:
  * cada REFRESH_S: catálogo actual (precios, productos nuevos) y popularidad de la última hora;
  * cada REFRESH_S: se incorporan a la memoria de usuarios los eventos nuevos de shop.events;
  * cada STOCK_S: productos agotados (recomendar uno invalida la respuesta entera).
Si algo falla, se responde con los más populares: nunca se deja una request sin respuesta.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import queue
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Un solo hilo para numpy. Las operaciones de este programa son pequeñas: repartirlas en varios hilos no
# acelera nada y, si el Mac está ocupado (p. ej. reentrenando a la vez), las frena unas 50 veces
# (medido: de 0,6 ms a 30 ms por request). Tiene que ir ANTES de importar numpy.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import requests  # noqa: E402

import model
from model import Items, UserStore

for line in Path(".env").read_text().splitlines() if Path(".env").exists() else []:
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())
API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ["TEAM_ID"]
http = requests.Session()
http.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"

REFRESH_S = 60          # catálogo, popularidad en vivo y memoria de usuarios
STOCK_S = 2             # productos agotados
POLL_S = 0.2            # pausa de cada lector entre dos lecturas de requests nuevas
POLLERS = 2             # lectores en paralelo (cada uno con su conexión): si una lectura se cuelga, el otro sigue
PUSHERS = 2             # hilos que envían las respuestas: un envío lento no frena la lectura siguiente
COVIEW_DAYS = 21        # días de histórico usados para co-vistas
CACHE = Path("cache")
DRY_RUN = "--dry-run" in sys.argv

# Test A/B: % de compradores que recibe cada versión. El reparto es por user_id (siempre la misma
# para el mismo comprador). La clave es la etiqueta model_version que muestra `monitor.py conv`.
# Para la final, dejad solo la ganadora al 100.
VARIANTS: dict[str, int] = {
    # Noche del sábado (ver NIGHT_RUNBOOK.md). Solo puntúa el domingo de 10 a 12: esta noche el tráfico se
    # reparte para APRENDER qué versión es la mejor.
    "v8-new-products": 28,  # CAMPEONA desde las 21:05 (decisión de tarre): v6 reentrenado + señal de producto nuevo
    "v12-cf": 20,           # v8 + parecido con lo que el comprador metió en el carrito o compró
    "v13-cf-session": 20,   # v12 + parecido con lo que acaba de ver en la sesión y con lo que vio otros días
    "v14-cf-split": 20,     # v13 + pesos distintos para compradores nuevos y conocidos + parecido estimado para productos nuevos
    "v6-similar-users": 12, # campeona anterior, como control de v8 (de momento empatadas)
    # Retiradas (conversión esperada del leaderboard, misma ventana):
    #   v9-organic      segunda oportunidad (20:35-21:15): 1,59 % frente a 1,96 % de v6 (z = -2,2). La primera vez,
    #                   1,30 % frente a 1,53 % (z = -2,0). Dos veces por debajo -> no usar
    #   v10-sale        v6 + empujón de 0,8 a los rebajados: con los datos de hoy el peso aprendido de "rebajado"
    #                   sale -0,15 -> el empujón no tiene base
    #   v5-cart-first   1,24 % frente a 1,53 % de v4 (z = -3,0)  -> no usar
    #   v4-live-weights 1,53 % frente a 1,66 % de v6 (z = -2,5)
    #   v7-explore / v7b-explore-v6: sorteos para aprender; bajan la conversión. Solo si se decide a propósito
}

# Exploración: variante -> versión de cuyas puntuaciones parte el sorteo.
# v7-explore sigue sobre v9 a propósito: v9 como modelo pierde (1,20 % frente a 1,46 % de v6), pero SORTEAR
# entre sus 40 mejores da 1,54 %, lo mejor que tenemos. Se compara con el mismo sorteo sobre v6 para saber
# si lo bueno es el sorteo en sí o los candidatos de v9.
EXPLORE = {"v7-explore": "v9", "v7b-explore-v6": "v6"}
EXPLORE_TOP = 40         # candidatos entre los que se sortea
EXPLORE_T = 1.0          # temperatura: 1 = cada candidato sale en proporción a su probabilidad estimada de compra
EXPLORE_DRAWS = 300      # sorteos extra para estimar la probabilidad de cada producto en cada puesto
RNG = np.random.default_rng()

# Pesos de v1 (puestos a mano)
V1 = {"coview": 1.0, "subcat": 0.6, "cat": 0.25, "viewed": 0.3, "decay": 0.7}

Record = dict[str, Any]


def pick_variant(req: Record) -> str:
    key = str(req.get("user_id") or req["request_id"])
    bucket = int(hashlib.md5(key.encode()).hexdigest(), 16) % sum(VARIANTS.values())
    for name, share in VARIANTS.items():
        bucket -= share
        if bucket < 0:
            return name
    return next(iter(VARIANTS))


def new_session() -> requests.Session:
    """Conexión propia para un hilo (misma clave). Así un hilo colgado no bloquea a los demás."""
    session = requests.Session()
    session.headers["Authorization"] = http.headers["Authorization"]
    return session


def query(sql: str, timeout: float = 30, session: requests.Session | None = None) -> list[Record]:
    r = (session or http).post(f"{API}/v1/query", json={"sql": sql}, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
    return r.json()["data"]


def push(answers: list[Record], session: requests.Session | None = None) -> None:
    """Inserción por lotes en team_NN.submissions."""
    if DRY_RUN:
        return
    # Hasta 3 intentos cortos: una respuesta que tarda más de 3 s ya no vale, así que no tiene sentido
    # esperar 10 s a un envío colgado. Si un intento se escribió pero no nos llegó la confirmación, el
    # reintento duplica la fila con la misma lista: no pasa nada, solo cuenta una.
    for attempt in (1, 2, 3):
        try:
            (session or http).post(f"{API}/v1/tables/submissions", params={"database": TEAM}, json=answers, timeout=4).raise_for_status()
            return
        except Exception as e:
            print(f"envío fallido (intento {attempt} de 3, {len(answers)} respuestas): {e!r}")
    raise RuntimeError(f"{len(answers)} respuestas perdidas: RawTree no aceptó el envío")


# --------------------------------------------------------------------------- SQL
CATALOG = """
SELECT product_id,
       argMax(category, v)    AS category,
       argMax(subcategory, v) AS subcategory,
       argMax(brand, v)       AS brand,
       argMax(price, v)       AS price_eur,
       argMin(price, v)       AS base_price,
       argMax(is_active, v)   AS is_active,
       max(is_new)            AS is_new
FROM (
    SELECT product_id::String AS product_id, category::String AS category, subcategory::String AS subcategory,
           brand::String AS brand, price_eur::Float64 AS price, is_active::Bool AS is_active, '' AS v, 0 AS is_new
    FROM shop.catalog
    {updates}
)
GROUP BY product_id"""

CATALOG_UPDATES = """
    UNION ALL
    SELECT product_id::String, category::String, subcategory::String, brand::String, price_eur::Float64,
           is_active::Bool, toString(ts::DateTime64(3)), toUInt8(change::String = 'new_product')
    FROM shop.catalog_updates"""

STOCK = """
SELECT product_id::String AS product_id, argMax(change::String, ts::DateTime64(3)) AS last
FROM shop.catalog_updates
WHERE change::String IN ('out_of_stock', 'back_in_stock')
GROUP BY product_id"""
"""Último cambio de stock de cada producto."""

NEW_PRODUCTS = """
SELECT product_id::String AS product_id, dateDiff('minute', min(ts::DateTime64(3)), now()) AS age_min
FROM shop.catalog_updates
WHERE change::String = 'new_product'
GROUP BY product_id"""
"""Productos añadidos durante el evento y cuántos minutos llevan en la tienda."""

POPULARITY = """
SELECT product_id::String AS product_id,
       countIf(event_type = 'purchase')    AS buys,
       countIf(event_type = 'add_to_cart') AS carts,
       countIf(event_type = 'view')        AS views
FROM {table}
WHERE {where}
GROUP BY product_id"""

COVIEWS = """
SELECT a.product_id AS p, b.product_id AS q, count() AS n
FROM (SELECT DISTINCT session_id::String AS session_id, product_id::String AS product_id
      FROM shop.history_events WHERE event_type = 'view' AND {where}) AS a
INNER JOIN (SELECT DISTINCT session_id::String AS session_id, product_id::String AS product_id
            FROM shop.history_events WHERE event_type = 'view' AND {where}) AS b
    ON a.session_id = b.session_id
WHERE a.product_id != b.product_id
GROUP BY p, q
ORDER BY n DESC
LIMIT 20 BY p"""

ALREADY_ANSWERED = """
SELECT request_id::String AS request_id
FROM {team}.submissions
WHERE request_id::String IN (SELECT request_id::String FROM shop.requests ORDER BY ts DESC LIMIT 500)"""

NEW_REQUESTS = """
SELECT request_id, ts, user_id, is_new_user, context, recent_views
FROM shop.requests
ORDER BY ts DESC
LIMIT 100"""

CAROUSEL_PURCHASES = """
SELECT r.user_id::String AS u, f.purchased::String AS p, toString(f.ts::DateTime64(3)) AS ts
FROM {team}.feedback AS f INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
WHERE f.purchased != '' AND f.ts::DateTime64(3) > '{since}'
ORDER BY ts"""
"""Lo que los usuarios nos han comprado a través del carrusel (para no volver a recomendárselo)."""

DUPLICATES = """
SELECT countIf(c > 1) AS dup, count() AS total FROM (
    SELECT request_id::String AS rid, count() AS c FROM {team}.submissions
    WHERE request_id::String IN (SELECT request_id::String FROM shop.requests WHERE ts::DateTime64(3) > now() - INTERVAL 2 MINUTE)
    GROUP BY rid)"""
"""Requests de los últimos 2 minutos con más de una respuesta: señal de que hay otro recomendador enviando."""

NEW_EVENTS = """
SELECT user_id::String AS u, product_id::String AS p, event_type::String AS t, toString(ts::DateTime64(3)) AS ts
FROM shop.events
WHERE ts::DateTime64(3) > '{since}'
ORDER BY ts
LIMIT 300000"""


def window(days: int, until: str | None) -> str:
    """Condición SQL: los `days` días anteriores a `until` (o al final del histórico)."""
    end = f"toDateTime64('{until}', 3)" if until else "(SELECT max(ts::DateTime64(3)) FROM shop.history_events) + INTERVAL 1 SECOND"
    return f"ts::DateTime64(3) < {end} AND ts::DateTime64(3) >= {end} - INTERVAL {days} DAY"


# --------------------------------------------------------------------------- estado
@dataclass
class State:
    """Todo lo que los modelos necesitan saber del catálogo. Se reemplaza entero en cada refresco."""
    product: dict[str, Record] = field(default_factory=dict)      # product_id -> {category, subcategory, ...}
    available: set[str] = field(default_factory=set)               # activos y en stock
    top_global: list[str] = field(default_factory=list)
    top_by_cat: dict[str, list[str]] = field(default_factory=dict)
    top_by_subcat: dict[str, list[str]] = field(default_factory=dict)
    items: Items | None = None                                     # lo mismo en columnas numéricas (v2)


def score_counts(rows: list[Record]) -> dict[str, float]:
    """Compras pesan más que carritos, y carritos más que vistas."""
    return {r["product_id"]: 3 * float(r["buys"]) + 1 * float(r["carts"]) + 0.05 * float(r["views"]) for r in rows}


def hist_popularity(until: str | None = None) -> dict[str, float]:
    """Popularidad de cada producto en los últimos 14 días del histórico."""
    return score_counts(query(POPULARITY.format(table="shop.history_events", where=window(14, until)), timeout=300))


def load_coviews(until: str | None = None) -> dict[str, list[str]]:
    """Para cada producto, los 20 que más se ven en la misma sesión. Se calcula en RawTree una vez."""
    path = CACHE / f"coviews_{COVIEW_DAYS}d{'_' + until[:10] if until else ''}.json"
    if path.exists():
        return json.loads(path.read_text())
    print(f"calculando co-vistas de {COVIEW_DAYS} días en RawTree (solo la primera vez)...")
    rows = query(COVIEWS.format(where=window(COVIEW_DAYS, until)), timeout=600)
    cov: dict[str, list[str]] = defaultdict(list)
    for r in rows:  # ya vienen ordenadas por n desc
        cov[r["p"]].append(r["q"])
    CACHE.mkdir(exist_ok=True)
    path.write_text(json.dumps(cov))
    return dict(cov)


def fair_new_popularity(pop: dict[str, float], hist_pop: dict[str, float], live_pop: dict[str, float],
                        new_age: dict[str, float]) -> int:
    """Corrige la popularidad de los productos nuevos (cold start). Devuelve cuántos ha corregido.

    Un producto nuevo no tiene 14 días de historia, así que al sumar "histórico + última hora" siempre
    quedaba hundido aunque hoy se venda como un superventas. Aquí se le estima la parte histórica que
    le falta: la que tienen los productos antiguos que hoy venden a un ritmo parecido (recta ajustada
    sobre los antiguos). Con menos de 5 minutos de vida no hay datos fiables: se deja sin popularidad
    y model.Items le pone la mediana de su subcategoría."""
    old = [(math.log1p(live_pop[p]), math.log1p(h)) for p, h in hist_pop.items() if h > 0 and live_pop.get(p, 0) > 0]
    if len(old) < 50:
        return 0
    x, y = np.array(old).T
    slope, intercept = np.polyfit(x, y, 1)
    fixed = 0
    for p, age_min in new_age.items():
        live = live_pop.get(p, 0)
        if p not in pop or hist_pop.get(p, 0) > 0 or age_min < 5 or live <= 0:
            continue
        rate = math.log1p(live * 60 / min(60.0, age_min))   # ventas por hora, aunque lleve menos de una hora
        pop[p] = min(max(intercept + slope * rate, 0.0), float(y.max())) + 2 * rate
        fixed += 1
    return fixed


def make_state(product: dict[str, Record], available: set[str], hist_pop: dict[str, float],
               live_pop: dict[str, float], new_age: dict[str, float] | None = None, mf: dict | None = None,
               cf: dict | None = None) -> State:
    """Ordena el catálogo por popularidad: histórico + última hora (que pesa x2: tendencias)."""
    # log1p para que los superventas no lo aplasten todo
    pop = {p: math.log1p(hist_pop.get(p, 0)) + 2 * math.log1p(live_pop.get(p, 0)) for p in product}
    if new_age:
        fair_new_popularity(pop, hist_pop, live_pop, new_age)
    s = State(product=product, available=available)
    ranked = sorted(available, key=lambda p: pop[p], reverse=True)
    s.top_global = ranked[:200]
    by_cat, by_sub = defaultdict(list), defaultdict(list)
    for p in ranked:
        info = product[p]
        if len(by_cat[info["category"]]) < 50:
            by_cat[info["category"]].append(p)
        if len(by_sub[info["subcategory"]]) < 30:
            by_sub[info["subcategory"]].append(p)
    s.top_by_cat, s.top_by_subcat = dict(by_cat), dict(by_sub)
    s.items = Items(product, pop, available, mf, cf)
    return s


def stock_changes() -> dict[str, str]:
    try:
        return {r["product_id"]: r["last"] for r in query(STOCK, timeout=10)}
    except Exception as e:
        if "Unknown expression" in str(e):  # antes de abrir el stream la tabla no tiene columnas
            return {}
        raise


def build_state(hist_pop: dict[str, float], mf: dict | None = None, cf: dict | None = None) -> State:
    try:
        rows = query(CATALOG.format(updates=CATALOG_UPDATES))
    except Exception:  # antes de abrir el stream catalog_updates no tiene columnas
        rows = query(CATALOG.format(updates=""))
    product = {r["product_id"]: r for r in rows}
    stock = stock_changes()
    # Disponible = su último cambio de stock es una reposición o, si nunca ha cambiado, está activo
    available = {p for p, r in product.items()
                 if (stock[p] == "back_in_stock" if p in stock else r["is_active"] in (True, 1, "true"))}
    try:
        live = score_counts(query(POPULARITY.format(table="shop.events", where="ts::DateTime64(3) > now() - INTERVAL 60 MINUTE")))
    except Exception as e:
        print(f"popularidad en vivo no disponible: {e}")
        live = {}
    try:
        new_age = {r["product_id"]: float(r["age_min"]) for r in query(NEW_PRODUCTS)}
    except Exception:
        new_age = {}
    return make_state(product, available, hist_pop, live, new_age, mf, cf)


# --------------------------------------------------------------------------- modelos
def recommend_v1(req: Record, s: State, coviews: dict[str, list[str]], w: dict = V1) -> list[str]:
    """v1: candidatos a partir de lo que acaba de ver; se rellena con lo más popular."""
    views = [v["product_id"] for v in reversed(req.get("recent_views") or [])]  # la más reciente primero
    score: Counter[str] = Counter()
    for k, p in enumerate(views[:10]):
        d = w["decay"] ** k
        score[p] += w["viewed"] * d
        for rank, q in enumerate(coviews.get(p, [])):
            score[q] += w["coview"] * d / (rank + 1)
        info = s.product.get(p)
        if info:
            for rank, q in enumerate(s.top_by_subcat.get(info["subcategory"], [])[:15]):
                score[q] += w["subcat"] * d / (rank + 1)
            for rank, q in enumerate(s.top_by_cat.get(info["category"], [])[:15]):
                score[q] += w["cat"] * d / (rank + 1)

    items = [p for p, _ in score.most_common() if p in s.available][:10]
    for p in s.top_global:  # relleno: quien llega sin vistas acaba aquí
        if len(items) == 10:
            break
        if p not in items:
            items.append(p)
    return items


def recommend_v2(req: Record, s: State, users: UserStore, coviews_idx: dict[int, list[int]], weights,
                 blocked: set[str] = frozenset(), skip_bought: bool = False, cart_first: bool = False,
                 cf_fill: bool = False) -> list[str]:
    """v2: puntúa todos los productos (señales × pesos aprendidos) y se queda con los 10 mejores.

    skip_bought (v3): descarta lo que el usuario ya compró. En el feedback real del 14-nov, un producto
    ya comprado se volvió a comprar 8 veces menos que el resto (analysis/live_lift.py)."""
    F = model.featurize(req.get("user_id") or "", req.get("recent_views"), s.items, users, coviews_idx, cf_fill)
    if weights[model.FI["organic"]] != 0:   # solo las versiones que usan la señal orgánica pagan sus ~6 ms
        model.add_organic(F)
    score = F @ weights
    if cart_first:
        # v5 (sobre los pesos de v4): lo que dejó en el carrito sin comprar va primero (ordenado entre sí por su puntuación). En el
        # feedback del 14-nov se compra 4-7 veces más que un producto normal en la misma posición, y solo
        # enseñábamos el 15 % de esos productos (analysis/cart_check.py).
        score[(F[:, model.FI["p_cart"]] > 0) & (F[:, model.FI["p_buy"]] == 0)] += 1000.0
    if skip_bought:
        score[F[:, model.FI["p_buy"]] > 0] = -np.inf
    return model.top10(score, s.items, blocked)


def recommend_explore(req: Record, s: State, users: UserStore, coviews_idx: dict[int, list[int]], weights,
                      blocked: set[str] = frozenset(), rng=RNG, T: float = EXPLORE_T,
                      top: int = EXPLORE_TOP) -> tuple[list[str], Record | None]:
    """Exploración: en vez de los 10 mejores, sortea 10 entre los `top` mejores (Plackett-Luce).

    Para qué: nuestro feedback solo dice qué pasa con lo que el modelo ya decide enseñar, y en el puesto en
    que lo pone. Aquí cada candidato sale con probabilidad proporcional a exp(puntuación / T), sin reemplazo,
    y el orden del sorteo es el orden de la lista: se aleatoriza también la posición. Devuelve la lista y un
    registro con las probabilidades, para corregir ese sesgo al reentrenar. Las reglas duras no cambian:
    nada ya comprado, nada agotado."""
    F = model.featurize(req.get("user_id") or "", req.get("recent_views"), s.items, users, coviews_idx)
    if weights[model.FI["organic"]] != 0:
        model.add_organic(F)
    score = (F @ weights).astype(np.float64)
    score[F[:, model.FI["p_buy"]] > 0] = -np.inf
    score[~s.items.avail] = -np.inf
    for p in blocked:
        i = model.PRODUCTS.idx.get(p)
        if i is not None and i < len(score):
            score[i] = -np.inf
    k = min(top, len(score) - 1)
    cand = np.argpartition(-score, k)[:k]
    cand = cand[np.isfinite(score[cand])]
    cand = cand[np.argsort(-score[cand], kind="stable")]
    if len(cand) < 10:
        return model.top10(score, s.items, blocked), None
    z = score[cand] / T
    z -= z.max()
    # Plackett-Luce sin reemplazo = ordenar por puntuación + ruido de Gumbel. La fila 0 es la lista que se
    # envía; las demás son sorteos de más para estimar con qué probabilidad sale cada producto en cada puesto.
    order = np.argsort(-(z + rng.gumbel(size=(EXPLORE_DRAWS + 1, len(cand)))), axis=1)[:, :10]
    pick, others = order[0], order[1:]
    w = np.exp(z)
    left, log_prob = float(w.sum()), 0.0
    for j in pick:  # probabilidad exacta de la lista: producto de las probabilidades de cada extracción
        log_prob += math.log(max(w[j] / left, 1e-300))
        left = max(left - w[j], 1e-300)
    names = model.PRODUCTS.names
    log = {"feature_set": "explore_v1", "kind": "explore", "T": T,
           "candidates": [names[i] for i in cand], "scores": [round(float(x), 4) for x in score[cand]],
           "shown": [names[cand[j]] for j in pick], "log_prob_list": round(log_prob, 4),
           # estimadas con EXPLORE_DRAWS sorteos: al usarlas como 1/p, acotar por abajo con 1/(2·EXPLORE_DRAWS)
           "p_slot": [round(float(x), 4) for x in (others == pick[None, :]).mean(0)],
           "p_any": [round(float(x), 4) for x in (others[:, :, None] == pick[None, None, :]).any(1).mean(0)]}
    return log["shown"], log


def valid(items: list[str], s: State, blocked: set[str]) -> list[str]:
    """Garantía final: 10 productos distintos, disponibles y no agotados; se rellena con populares."""
    out = list(dict.fromkeys(p for p in items if p in s.available and p not in blocked))[:10]
    for p in s.top_global:
        if len(out) == 10:
            break
        if p not in out and p not in blocked:
            out.append(p)
    return out


# --------------------------------------------------------------------------- segundo plano
class Background:
    """Mantiene al día, sin frenar el bucle principal: estado del catálogo, agotados y memoria de usuarios."""

    def __init__(self, hist_pop: dict[str, float]):
        self.hist_pop = hist_pop
        self.mf = model.load_mf()   # factores de "usuarios parecidos"; None si no está mf_factors.npz
        self.cf = model.load_cf()   # similitud producto-producto; None si no está cf_sim.npz
        self.state = build_state(hist_pop, self.mf, self.cf)
        self.blocked: set[str] = set()   # agotados según la última comprobación rápida
        self.users = UserStore()
        self.users_ready = False
        self.feedback_watermark = "2000-01-01 00:00:00"   # último feedback con compra ya incorporado
        self.stock_ok = time.time()                        # última comprobación de stock que funcionó
        self.started = time.time()

    def start(self) -> None:
        threading.Thread(target=self._stock_loop, daemon=True).start()
        threading.Thread(target=self._slow_loop, daemon=True).start()

    def _stock_loop(self) -> None:
        while True:
            try:
                self.blocked = {p for p, last in stock_changes().items() if last == "out_of_stock"}
                self.stock_ok = time.time()
            except Exception as e:
                stale = time.time() - self.stock_ok
                print(f"comprobación de stock fallida: {e}" + (
                    f" | AVISO: la lista de agotados tiene {stale:.0f} s; riesgo de respuestas inválidas" if stale > 30 else ""))
            time.sleep(STOCK_S)

    def _load_users(self) -> None:
        t = time.time()
        path = CACHE / "users_history_v2.pkl"   # v2: incluye todos los productos comprados
        if path.exists():
            self.users.load_file(path)
        else:
            print("cargando historial de usuarios desde RawTree (solo la primera vez, ~1 min)...")
            self.users.load(query, "shop.history_events")
            self.users.save(path)
        n_hist = len(self.users.row)
        self.users.init_mf(self.mf)
        try:
            wm = query("SELECT toString(max(ts::DateTime64(3))) AS wm FROM shop.events")[0]["wm"]
            self.users.load(query, "shop.events", cond=f"AND e.ts::DateTime64(3) <= '{wm}'", mfV=self.state.items.mfV)
            self.users.watermark = wm
        except Exception as e:
            print(f"eventos en vivo no disponibles todavía: {e}")
        self.users_ready = True
        print(f"memoria de usuarios lista en {time.time() - t:.0f}s: {n_hist} con historial, "
              f"{len(self.users.row)} contando los de hoy")

    def _update_users(self) -> int:
        if not self.users.watermark:
            return 0
        rows = query(NEW_EVENTS.format(since=self.users.watermark), timeout=60)
        items = self.state.items
        for r in rows:
            self.users.add_event(r["u"], r["p"], r["t"], items)
        if rows:
            self.users.watermark = rows[-1]["ts"]
        bought = query(CAROUSEL_PURCHASES.format(team=TEAM, since=self.feedback_watermark), timeout=60)
        for r in bought:
            self.users.add_event(r["u"], r["p"], "purchase", items)
        if bought:
            self.feedback_watermark = bought[-1]["ts"]
        return len(rows)

    def _slow_loop(self) -> None:
        try:
            self._load_users()
        except Exception as e:
            print(f"no se pudo cargar la memoria de usuarios ({e}); v2 funcionará sin historial")
        while True:
            time.sleep(REFRESH_S)
            try:
                t = time.time()
                self.state = build_state(self.hist_pop, self.mf, self.cf)  # reemplazo atómico de la referencia
                self.users.ensure_cols()
                n = self._update_users()
                print(f"refresco en {time.time() - t:.1f}s: {len(self.state.available)} productos disponibles, "
                      f"{n} eventos nuevos, {len(self.users.row)} usuarios en memoria")
                if not DRY_RUN and time.time() - self.started > 180:  # tras un relevo, dejar pasar la ventana
                    d = query(DUPLICATES.format(team=TEAM), timeout=20)[0]
                    if int(d["dup"]) > 5:
                        print(f"!!! AVISO: {d['dup']} de {d['total']} requests recientes tienen DOS respuestas: hay OTRO "
                              f"recomendador enviando (¿otro portátil?). Solo debe correr uno: parad el otro.")
            except Exception as e:
                print(f"refresco fallido, sigo con el estado anterior: {e}")


@dataclass
class Models:
    """Lo que no cambia mientras corre el proceso: co-vistas y pesos."""
    coviews: dict
    coviews_idx: dict
    weights: Any = None        # model_weights.json (histórico)
    weights_live: Any = None   # model_weights_live.json (corregidos con el feedback en vivo)
    by_version: dict = field(default_factory=dict)   # "v6", "v7"... -> pesos de model_weights_v6.json, _v7.json...
    options: dict = field(default_factory=dict)      # "v14" -> opciones con las que se entrenó (p. ej. cf_fill)
    explore_q: Any = None      # cola de registros de exploración pendientes de enviar a scratch


def load_models() -> Models:
    coviews = load_coviews()
    live_file = model.WEIGHTS_FILE.with_name("model_weights_live.json")
    m = Models(coviews, model.index_coviews(coviews),
               model.load_weights() if model.WEIGHTS_FILE.exists() else None,
               model.load_weights(live_file) if live_file.exists() else None)
    # Versiones posteriores (v6, v7...): cada una con su fichero de pesos model_weights_<versión>.json
    for f in sorted(model.WEIGHTS_FILE.parent.glob("model_weights_v*.json")):
        m.by_version[f.stem.removeprefix("model_weights_")] = model.load_weights(f)
        m.options[f.stem.removeprefix("model_weights_")] = model.load_options(f)
    m.explore_q = queue.Queue(maxsize=20_000)
    for v in VARIANTS:
        prefix = EXPLORE.get(v, v.split("-")[0])
        if prefix not in ("v1", "v2", "v3", "v4", "v5") and prefix not in m.by_version:
            print(f"AVISO: falta model_weights_{prefix}.json; {v} no puede funcionar")
    if not model.MF_FILE.exists():
        print("AVISO: falta mf_factors.npz; la señal de usuarios parecidos valdrá 0")
    if m.weights is None:
        print("AVISO: falta model_weights.json (ejecuta train.py); se usará v1")
    if m.weights_live is None and any(v.startswith(("v4", "v5")) for v in VARIANTS):
        print("AVISO: falta model_weights_live.json (ejecuta train_live.py); v4/v5 usarán los pesos del histórico")
    return m


def respond(req: Record, version: str, s: State, blocked: set[str], users: UserStore, m: Models) -> Record:
    """La respuesta a una request con la variante indicada. Nunca falla: si algo va mal, los más populares."""
    try:
        prefix = version.split("-")[0]
        if version in EXPLORE:
            w = m.by_version.get(EXPLORE[version])
            if w is None:
                raise RuntimeError(f"faltan los pesos de {EXPLORE[version]} para explorar")
            items, log = recommend_explore(req, s, users, m.coviews_idx, w, blocked)
            if log is not None and m.explore_q is not None:
                log.update(request_id=req["request_id"], user_id=req.get("user_id"), model_version=version,
                           base=EXPLORE[version])
                try:
                    m.explore_q.put_nowait(log)
                except queue.Full:   # el registro nunca frena ni rompe la respuesta
                    pass
        elif prefix != "v1" and m.weights is not None:   # ojo: comparar el prefijo exacto ("v10" empieza por "v1")
            w = m.weights_live if prefix in ("v4", "v5") and m.weights_live is not None else m.weights
            if prefix not in ("v2", "v3", "v4", "v5"):
                if prefix not in m.by_version:
                    raise RuntimeError(f"faltan los pesos de {prefix}")
                w = m.by_version[prefix]
                # Pesos propios para compradores nuevos, si la versión los trae (model_weights_<versión>n.json)
                if prefix + "n" in m.by_version and str(req.get("is_new_user")).lower() in ("true", "1"):
                    w = m.by_version[prefix + "n"]
            items = recommend_v2(req, s, users, m.coviews_idx, w, blocked,
                                 skip_bought=prefix != "v2", cart_first=prefix == "v5",
                                 cf_fill=bool(m.options.get(prefix, {}).get("cf_fill")))
        else:
            version = version if prefix == "v1" else "v1-coview-trend"
            items = recommend_v1(req, s, m.coviews)
    except Exception as e:
        print(f"recommend falló ({e!r}), uso los más populares")
        version, items = "fallback-propio", []
    return {"request_id": req["request_id"], "items": valid(items, s, blocked), "model_version": version}


def explore_logger(q: "queue.Queue") -> None:
    """Hilo aparte: envía a team_NN.scratch los registros de exploración, por lotes. Si un envío falla, ese
    lote se descarta y se avisa: las respuestas a los compradores nunca dependen de esto."""
    session = requests.Session()
    session.headers["Authorization"] = http.headers["Authorization"]
    while True:
        time.sleep(2)
        batch: list[Record] = []
        while len(batch) < 400:
            try:
                batch.append(q.get_nowait())
            except queue.Empty:
                break
        if batch:
            try:
                session.post(f"{API}/v1/tables/scratch", params={"database": TEAM}, json=batch, timeout=10).raise_for_status()
            except Exception as e:
                print(f"registro de exploración fallido ({len(batch)} filas descartadas): {e!r}")


def age_s(req: Record) -> float:
    """Segundos desde el ts de la request hasta ahora (según el reloj de este portátil). Solo para el log."""
    try:
        ts = dt.datetime.fromisoformat(str(req["ts"])[:23]).replace(tzinfo=dt.timezone.utc)
        return time.time() - ts.timestamp()
    except Exception:
        return 0.0


def poller(inbox: "queue.Queue", delay: float) -> None:
    """Hilo lector: pide las últimas requests una y otra vez y las deja en `inbox`. Nunca termina."""
    session = new_session()
    time.sleep(delay)   # los lectores van escalonados
    while True:
        try:
            inbox.put(query(NEW_REQUESTS, timeout=3, session=session))
        except Exception as e:
            print(f"error: {e}")
            session = new_session()   # conexión nueva: la anterior puede haberse quedado colgada
            time.sleep(0.3)
        time.sleep(POLL_S)


def pusher(outbox: "queue.Queue") -> None:
    """Hilo emisor: envía los lotes de respuestas. Nunca termina."""
    session = new_session()
    while True:
        answers = outbox.get()
        try:
            push(answers, session)
        except Exception as e:
            print(f"error: {e}")
            session = new_session()


def preflight() -> int:
    """Comprobación antes de un despliegue: carga todo y responde a 100 requests reales con CADA variante,
    sin enviar nada. Devuelve 0 si todo es válido (lo usa deploy.sh)."""
    t0 = time.time()
    bg = Background(hist_popularity())
    m = load_models()
    bg._load_users()
    s, blocked = bg.state, {p for p, last in stock_changes().items() if last == "out_of_stock"}
    reqs = query(NEW_REQUESTS, timeout=10)
    ok = len(reqs) > 0 and len(s.available) >= 10
    if any(w[model.FI["organic"]] != 0 for w in m.by_version.values()) and model.load_organic() is None:
        print("  FALTA organic_gbm.pkl (o scikit-learn): hay versiones que usan la señal orgánica")
        ok = False
    for version in VARIANTS:
        t = time.time()
        out = [respond(r, version, s, blocked, bg.users, m) for r in reqs]
        ms = (time.time() - t) * 1000 / max(len(out), 1)
        bad = sum(len(a["items"]) != 10 or len(set(a["items"])) != 10 or not set(a["items"]) <= s.available
                  or bool(set(a["items"]) & blocked) for a in out)
        fell_back = sum(a["model_version"] != version for a in out)
        print(f"  {version:<18} {len(out)} requests | inválidas: {bad} | no usaron esta variante: {fell_back} | {ms:.2f} ms/request")
        ok = ok and bad == 0 and fell_back == 0 and ms < 200
    print(f"PREFLIGHT {'OK' if ok else 'FALLIDO'} ({time.time() - t0:.0f}s) | variantes: {VARIANTS}")
    return 0 if ok else 1


def main() -> None:
    if "--preflight" in sys.argv:
        sys.exit(preflight())
    print("MODO PRUEBA: no se envía nada" if DRY_RUN else "PRODUCCIÓN: enviando respuestas")
    hist_pop = hist_popularity()
    bg = Background(hist_pop)
    m = load_models()
    bg.start()
    if not DRY_RUN and any(v in EXPLORE for v in VARIANTS):
        threading.Thread(target=explore_logger, args=(m.explore_q,), daemon=True).start()
    print(f"listo: {len(bg.state.available)} productos disponibles. Variantes: {VARIANTS}")

    seen: dict[str, None] = {}  # dict = conjunto que recuerda el orden de llegada
    # Al (re)arrancar, no volver a responder lo que ya respondió el proceso anterior: sería una
    # respuesta duplicada y además tardía.
    try:
        seen = dict.fromkeys(r["request_id"] for r in query(ALREADY_ANSWERED.format(team=TEAM)))
        print(f"{len(seen)} requests recientes ya tenían respuesta: no se repiten")
    except Exception as e:
        print(f"no se pudo leer lo ya respondido ({e}); sigo igualmente")
    # Tres tipos de hilo, para que nada espere a nada:
    #   lectores  -> leen las requests nuevas y las dejan en `inbox`
    #   este hilo -> calcula las respuestas (es el único que decide qué está ya respondido)
    #   emisores  -> envían cada lote a RawTree
    inbox: "queue.Queue[list[Record]]" = queue.Queue()
    outbox: "queue.Queue[list[Record]]" = queue.Queue()

    def start_threads() -> list[threading.Thread]:
        threads = [threading.Thread(target=poller, args=(inbox, k * POLL_S / POLLERS), daemon=True) for k in range(POLLERS)]
        threads += [threading.Thread(target=pusher, args=(outbox,), daemon=True) for _ in range(PUSHERS)]
        for t in threads:
            t.start()
        return threads

    threads = start_threads()
    answered = 0
    ages: list[float] = []   # segundos entre el ts de cada request y el momento de responderla
    while True:
        try:
            try:
                rows = inbox.get(timeout=10)
            except queue.Empty:   # 10 s sin ninguna lectura: no debería pasar nunca
                dead = sum(not t.is_alive() for t in threads)
                print(f"AVISO: 10 s sin lecturas de requests ({dead} hilos caídos de {len(threads)})")
                if dead:
                    threads = start_threads()
                continue
            fresh = [r for r in rows if r["request_id"] not in seen]
            if fresh:
                s, blocked = bg.state, bg.blocked
                answers = []
                for req in fresh:
                    seen[req["request_id"]] = None
                    answers.append(respond(req, pick_variant(req), s, blocked, bg.users, m))
                outbox.put(answers)
                answered += len(answers)
                ages += [age_s(req) for req in fresh]
                if answered % 300 < len(answers):
                    ages.sort()
                    print(f"{answered} respondidas | edad de la request al responder: mediana {ages[len(ages) // 2]:.2f} s, "
                          f"p99 {ages[int(len(ages) * 0.99)]:.2f} s | lotes en cola de envío: {outbox.qsize()}"
                          + (f" | ejemplo {answers[-1]}" if DRY_RUN else ""))
                    ages = []
            if len(seen) > 50_000:  # no crecer sin límite durante toda la noche
                seen = dict.fromkeys(list(seen)[-10_000:])  # conserva las más recientes
        except Exception as e:  # sigue vivo: los compradores no paran
            print(f"error: {e}")
            time.sleep(0.2)


if __name__ == "__main__":
    main()
