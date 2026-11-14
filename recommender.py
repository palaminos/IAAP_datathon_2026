"""Recomendador del equipo (bucle pull → predict → push).

    python recommender.py              # producción: responde y envía (usad ./run_forever.sh)
    python recommender.py --dry-run    # prueba: hace todo igual pero NO envía nada

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

import hashlib
import json
import math
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests

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
POLL_S = 0.3            # cada cuánto se leen requests nuevas
COVIEW_DAYS = 21        # días de histórico usados para co-vistas
CACHE = Path("cache")
DRY_RUN = "--dry-run" in sys.argv

# Test A/B: % de compradores que recibe cada versión. El reparto es por user_id (siempre la misma
# para el mismo comprador). La clave es la etiqueta model_version que muestra `monitor.py conv`.
# Para la final, dejad solo la ganadora al 100.
VARIANTS: dict[str, int] = {
    "v5-cart-first": 50,    # v4 + los productos que el usuario dejó en el carrito sin comprar, arriba del todo
    "v4-live-weights": 50,  # v3 con los pesos corregidos con nuestro feedback en vivo (train_live.py)
    # "v3-no-rebuy": 0,     # v2 + no recomendar lo ya comprado. Empate con v4 en vivo (1,43 % vs 1,57 %, 14-nov)
    # "v2-user-lr": 0,      # iba por detrás de v3 en vivo (14-nov)
    # "v1-coview-trend": 0, # perdió contra v2 en vivo (0,53 % vs 1,13 % de conversión, 14-nov)
}

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


def query(sql: str, timeout: float = 30) -> list[Record]:
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
    return r.json()["data"]


def push(answers: list[Record]) -> None:
    """Inserción por lotes en team_NN.submissions."""
    if DRY_RUN:
        return
    http.post(f"{API}/v1/tables/submissions", params={"database": TEAM}, json=answers, timeout=10).raise_for_status()


# --------------------------------------------------------------------------- SQL
CATALOG = """
SELECT product_id,
       argMax(category, v)    AS category,
       argMax(subcategory, v) AS subcategory,
       argMax(brand, v)       AS brand,
       argMax(price_eur, v)   AS price_eur,
       argMax(is_active, v)   AS is_active,
       max(is_new)            AS is_new
FROM (
    SELECT product_id::String AS product_id, category::String AS category, subcategory::String AS subcategory,
           brand::String AS brand, price_eur::Float64 AS price_eur, is_active::Bool AS is_active, '' AS v, 0 AS is_new
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
               live_pop: dict[str, float], new_age: dict[str, float] | None = None) -> State:
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
    s.items = Items(product, pop, available)
    return s


def stock_changes() -> dict[str, str]:
    try:
        return {r["product_id"]: r["last"] for r in query(STOCK, timeout=10)}
    except Exception as e:
        if "Unknown expression" in str(e):  # antes de abrir el stream la tabla no tiene columnas
            return {}
        raise


def build_state(hist_pop: dict[str, float]) -> State:
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
    return make_state(product, available, hist_pop, live, new_age)


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
                 blocked: set[str] = frozenset(), skip_bought: bool = False, cart_first: bool = False) -> list[str]:
    """v2: puntúa todos los productos (señales × pesos aprendidos) y se queda con los 10 mejores.

    skip_bought (v3): descarta lo que el usuario ya compró. En el feedback real del 14-nov, un producto
    ya comprado se volvió a comprar 8 veces menos que el resto (analysis/live_lift.py)."""
    F = model.featurize(req.get("user_id") or "", req.get("recent_views"), s.items, users, coviews_idx)
    score = F @ weights
    if cart_first:
        # v5 (sobre los pesos de v4): lo que dejó en el carrito sin comprar va primero (ordenado entre sí por su puntuación). En el
        # feedback del 14-nov se compra 4-7 veces más que un producto normal en la misma posición, y solo
        # enseñábamos el 15 % de esos productos (analysis/cart_check.py).
        score[(F[:, model.FI["p_cart"]] > 0) & (F[:, model.FI["p_buy"]] == 0)] += 1000.0
    if skip_bought:
        score[F[:, model.FI["p_buy"]] > 0] = -np.inf
    return model.top10(score, s.items, blocked)


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
        self.state = build_state(hist_pop)
        self.blocked: set[str] = set()   # agotados según la última comprobación rápida
        self.users = UserStore()
        self.users_ready = False
        self.feedback_watermark = "2000-01-01 00:00:00"   # último feedback con compra ya incorporado

    def start(self) -> None:
        threading.Thread(target=self._stock_loop, daemon=True).start()
        threading.Thread(target=self._slow_loop, daemon=True).start()

    def _stock_loop(self) -> None:
        while True:
            try:
                self.blocked = {p for p, last in stock_changes().items() if last == "out_of_stock"}
            except Exception as e:
                print(f"comprobación de stock fallida: {e}")
            time.sleep(STOCK_S)

    def _load_users(self) -> None:
        t = time.time()
        path = CACHE / "users_history.pkl"
        if path.exists():
            self.users.load_file(path)
        else:
            print("cargando historial de usuarios desde RawTree (solo la primera vez, ~1 min)...")
            self.users.load(query, "shop.history_events")
            self.users.save(path)
        n_hist = len(self.users.row)
        try:
            wm = query("SELECT toString(max(ts::DateTime64(3))) AS wm FROM shop.events")[0]["wm"]
            self.users.load(query, "shop.events", cond=f"AND e.ts::DateTime64(3) <= '{wm}'")
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
                self.state = build_state(self.hist_pop)  # reemplazo atómico de la referencia
                self.users.ensure_cols()
                n = self._update_users()
                print(f"refresco en {time.time() - t:.1f}s: {len(self.state.available)} productos disponibles, "
                      f"{n} eventos nuevos, {len(self.users.row)} usuarios en memoria")
            except Exception as e:
                print(f"refresco fallido, sigo con el estado anterior: {e}")


def main() -> None:
    print("MODO PRUEBA: no se envía nada" if DRY_RUN else "PRODUCCIÓN: enviando respuestas")
    hist_pop = hist_popularity()
    coviews = load_coviews()
    bg = Background(hist_pop)
    coviews_idx = model.index_coviews(coviews)
    weights = model.load_weights() if model.WEIGHTS_FILE.exists() else None
    live_file = model.WEIGHTS_FILE.with_name("model_weights_live.json")
    weights_live = model.load_weights(live_file) if live_file.exists() else None
    if weights_live is None and any(v.startswith(("v4", "v5")) for v in VARIANTS):
        print("AVISO: falta model_weights_live.json (ejecuta train_live.py); v4 usará los pesos del histórico")
    if weights is None:
        print("AVISO: falta model_weights.json (ejecuta train.py); los compradores de v2 recibirán v1")
    bg.start()
    print(f"listo: {len(bg.state.available)} productos disponibles. Variantes: {VARIANTS}")

    seen: dict[str, None] = {}  # dict = conjunto que recuerda el orden de llegada
    # Al (re)arrancar, no volver a responder lo que ya respondió el proceso anterior: sería una
    # respuesta duplicada y además tardía.
    try:
        seen = dict.fromkeys(r["request_id"] for r in query(ALREADY_ANSWERED.format(team=TEAM)))
        print(f"{len(seen)} requests recientes ya tenían respuesta: no se repiten")
    except Exception as e:
        print(f"no se pudo leer lo ya respondido ({e}); sigo igualmente")
    answered = 0
    while True:
        try:
            fresh = [r for r in query(NEW_REQUESTS, timeout=5) if r["request_id"] not in seen]
            if fresh:
                s, blocked = bg.state, bg.blocked
                answers = []
                for req in fresh:
                    seen[req["request_id"]] = None
                    try:
                        version = pick_variant(req)
                        if version.startswith(("v2", "v3", "v4", "v5")) and weights is not None:
                            w = weights_live if version.startswith(("v4", "v5")) and weights_live is not None else weights
                            items = recommend_v2(req, s, bg.users, coviews_idx, w, blocked,
                                                 skip_bought=not version.startswith("v2"),
                                                 cart_first=version.startswith("v5"))
                        else:
                            version = version if version.startswith("v1") else "v1-coview-trend"
                            items = recommend_v1(req, s, coviews)
                    except Exception as e:
                        print(f"recommend falló ({e!r}), uso los más populares")
                        version, items = "fallback-propio", []
                    answers.append({"request_id": req["request_id"], "items": valid(items, s, blocked),
                                    "model_version": version})
                push(answers)
                answered += len(answers)
                if answered % 300 < len(answers):
                    print(f"{answered} respondidas" + (f" | ejemplo {answers[-1]}" if DRY_RUN else ""))
            if len(seen) > 50_000:  # no crecer sin límite durante toda la noche
                seen = dict.fromkeys(list(seen)[-10_000:])  # conserva las más recientes
        except Exception as e:  # sigue vivo: los compradores no paran
            print(f"error: {e}")
            time.sleep(1)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
