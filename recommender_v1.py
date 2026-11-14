"""Recomendador v1: co-vistas + popularidad por subcategoría/categoría + tendencias en vivo.

    python recommender_v1.py

Mismo bucle pull → predict → push que starter/recommender.py, pero:
  * Un hilo en segundo plano refresca cada REFRESH_S segundos el catálogo actual (stock,
    precios, productos nuevos) y la popularidad en vivo de shop.events (las tendencias).
  * Las co-vistas del histórico (qué productos se ven en la misma sesión) se calculan una
    vez en RawTree y se guardan en cache/coviews.json para que reiniciar sea instantáneo.
  * recommend() puntúa candidatos a partir de lo que el usuario acaba de ver y rellena con
    lo más popular. Si algo falla, responde con la lista global: nunca se queda sin responder.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

for line in Path(".env").read_text().splitlines() if Path(".env").exists() else []:
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())
API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ["TEAM_ID"]
http = requests.Session()
http.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"

MODEL_VERSION = "v1-coview-trend"
REFRESH_S = 60          # cada cuánto se recalcula catálogo + popularidad en vivo
POLL_S = 0.3            # cada cuánto se leen requests nuevas
COVIEW_DAYS = 21        # días de histórico usados para co-vistas
CACHE = Path("cache")

# Pesos de la puntuación: ajústalos con la evaluación offline y el feedback.
#   coview : productos co-vistos con lo que acaba de ver
#   subcat : populares de la misma subcategoría
#   cat    : populares de la misma categoría
#   viewed : el propio producto visto (mucha gente compra lo que estaba mirando)
#   decay  : peso de cada vista anterior respecto a la siguiente (la más reciente pesa 1)
DEFAULT = {"coview": 1.0, "subcat": 0.6, "cat": 0.25, "viewed": 0.3, "decay": 0.7}

# Test A/B: cada comprador se asigna SIEMPRE a la misma variante (por user_id), y todas compiten a la
# vez con los mismos compradores. La clave es la etiqueta model_version que verá monitor.py.
# Con una sola variante no hay test: todos reciben esa. Para la final, dejad solo la ganadora.
VARIANTS: dict[str, dict] = {
    MODEL_VERSION: DEFAULT,
    # "v1b-sin-vistos": {**DEFAULT, "viewed": 0.0},   # ejemplo: ¿ayuda recomendar lo que ya vio?
}


def pick_variant(req: Record) -> str:
    """Reparto estable: el mismo user_id cae siempre en la misma variante, aunque se reinicie el script."""
    names = list(VARIANTS)
    key = str(req.get("user_id") or req["request_id"])
    return names[int(hashlib.md5(key.encode()).hexdigest(), 16) % len(names)]

Record = dict[str, Any]


def query(sql: str, timeout: float = 30) -> list[Record]:
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
    return r.json()["data"]


def push(answers: list[Record]) -> None:
    """Inserción por lotes en team_NN.submissions."""
    http.post(f"{API}/v1/tables/submissions", params={"database": TEAM}, json=answers, timeout=10).raise_for_status()


# --------------------------------------------------------------------------- SQL
CATALOG = """
SELECT product_id,
       argMax(category, v)    AS category,
       argMax(subcategory, v) AS subcategory,
       argMax(price_eur, v)   AS price_eur,
       argMax(is_active, v)   AS is_active,
       max(is_new)            AS is_new
FROM (
    SELECT product_id::String AS product_id, category::String AS category, subcategory::String AS subcategory,
           price_eur::Float64 AS price_eur, is_active::Bool AS is_active, '' AS v, 0 AS is_new
    FROM shop.catalog
    {updates}
)
GROUP BY product_id"""

CATALOG_UPDATES = """
    UNION ALL
    SELECT product_id::String, category::String, subcategory::String, price_eur::Float64, is_active::Bool,
           toString(ts::DateTime64(3)), toUInt8(change::String = 'new_product')
    FROM shop.catalog_updates"""

OUT_OF_STOCK = """
SELECT product_id::String AS product_id
FROM shop.catalog_updates
WHERE change::String IN ('out_of_stock', 'back_in_stock')
GROUP BY product_id
HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'"""

HIST_POPULARITY = """
SELECT product_id::String AS product_id,
       countIf(event_type = 'purchase')    AS buys,
       countIf(event_type = 'add_to_cart') AS carts,
       countIf(event_type = 'view')        AS views
FROM shop.history_events
WHERE ts::DateTime64(3) > (SELECT max(ts::DateTime64(3)) FROM shop.history_events) - INTERVAL 14 DAY
GROUP BY product_id"""

LIVE_POPULARITY = """
SELECT product_id::String AS product_id,
       countIf(event_type = 'purchase')    AS buys,
       countIf(event_type = 'add_to_cart') AS carts,
       countIf(event_type = 'view')        AS views
FROM shop.events
WHERE ts::DateTime64(3) > now() - INTERVAL 60 MINUTE
GROUP BY product_id"""

COVIEWS = """
WITH (SELECT max(ts::DateTime64(3)) FROM shop.history_events) - INTERVAL {days} DAY AS cut
SELECT a.product_id AS p, b.product_id AS q, count() AS n
FROM (SELECT DISTINCT session_id::String AS session_id, product_id::String AS product_id
      FROM shop.history_events WHERE event_type = 'view' AND ts::DateTime64(3) > cut) AS a
INNER JOIN (SELECT DISTINCT session_id::String AS session_id, product_id::String AS product_id
            FROM shop.history_events WHERE event_type = 'view' AND ts::DateTime64(3) > cut) AS b
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


# --------------------------------------------------------------------------- estado
@dataclass
class State:
    """Todo lo que recommend() necesita, en memoria. Se reemplaza entero en cada refresco."""
    product: dict[str, Record] = field(default_factory=dict)      # product_id -> {category, subcategory, ...}
    available: set[str] = field(default_factory=set)               # activos y en stock
    popularity: dict[str, float] = field(default_factory=dict)     # score histórico + en vivo
    top_global: list[str] = field(default_factory=list)
    top_by_cat: dict[str, list[str]] = field(default_factory=dict)
    top_by_subcat: dict[str, list[str]] = field(default_factory=dict)


def load_coviews() -> dict[str, list[str]]:
    path = CACHE / f"coviews_{COVIEW_DAYS}d.json"
    if path.exists():
        return json.loads(path.read_text())
    print(f"calculando co-vistas de {COVIEW_DAYS} días en RawTree (solo la primera vez)...")
    rows = query(COVIEWS.format(days=COVIEW_DAYS), timeout=600)
    cov: dict[str, list[str]] = defaultdict(list)
    for r in rows:  # ya vienen ordenadas por n desc
        cov[r["p"]].append(r["q"])
    CACHE.mkdir(exist_ok=True)
    path.write_text(json.dumps(cov))
    print(f"co-vistas: {len(cov)} productos")
    return dict(cov)


def score_counts(rows: list[Record]) -> dict[str, float]:
    """Compras pesan más que carritos, y carritos más que vistas."""
    return {r["product_id"]: 3 * float(r["buys"]) + 1 * float(r["carts"]) + 0.05 * float(r["views"]) for r in rows}


def build_state(hist_pop: dict[str, float]) -> State:
    s = State()
    try:
        rows = query(CATALOG.format(updates=CATALOG_UPDATES))
    except Exception:  # antes de abrir el stream catalog_updates no tiene columnas
        rows = query(CATALOG.format(updates=""))
    s.product = {r["product_id"]: r for r in rows}
    try:
        oos = {r["product_id"] for r in query(OUT_OF_STOCK)}
    except Exception:
        oos = set()
    s.available = {p for p, r in s.product.items() if r["is_active"] in (True, 1, "true") and p not in oos}

    try:
        live = score_counts(query(LIVE_POPULARITY))
    except Exception as e:
        print(f"popularidad en vivo no disponible: {e}")
        live = {}
    # log1p para que los superventas no lo aplasten todo; lo de la última hora pesa x2 (tendencias)
    s.popularity = {p: math.log1p(hist_pop.get(p, 0)) + 2 * math.log1p(live.get(p, 0)) for p in s.available}

    ranked = sorted(s.available, key=lambda p: s.popularity[p], reverse=True)
    s.top_global = ranked[:200]
    by_cat, by_sub = defaultdict(list), defaultdict(list)
    for p in ranked:
        info = s.product[p]
        if len(by_cat[info["category"]]) < 50:
            by_cat[info["category"]].append(p)
        if len(by_sub[info["subcategory"]]) < 30:
            by_sub[info["subcategory"]].append(p)
    s.top_by_cat, s.top_by_subcat = dict(by_cat), dict(by_sub)
    return s


# --------------------------------------------------------------------------- modelo
def recommend(req: Record, s: State, coviews: dict[str, list[str]], w: dict = DEFAULT) -> list[str]:
    """10 product IDs para una request. Esta es la función a mejorar."""
    views = [v["product_id"] for v in reversed(req.get("recent_views") or [])]  # la más reciente primero
    score: Counter[str] = Counter()
    for k, p in enumerate(views[:10]):
        d = w["decay"] ** k
        score[p] += w["viewed"] * d
        for rank, q in enumerate(coviews.get(p, [])):
            score[q] += w["coview"] * d / (rank + 1)
        info = s.product.get(p)
        if info:  # funciona también con productos nuevos sin co-vistas (cold start)
            for rank, q in enumerate(s.top_by_subcat.get(info["subcategory"], [])[:15]):
                score[q] += w["subcat"] * d / (rank + 1)
            for rank, q in enumerate(s.top_by_cat.get(info["category"], [])[:15]):
                score[q] += w["cat"] * d / (rank + 1)

    items = [p for p, _ in score.most_common() if p in s.available][:10]
    for p in s.top_global:  # relleno: usuarios nuevos sin vistas acaban aquí (tendencias globales)
        if len(items) == 10:
            break
        if p not in items:
            items.append(p)
    return items


# --------------------------------------------------------------------------- bucle
class Refresher(threading.Thread):
    """Recalcula State en segundo plano para que el bucle principal nunca espere a SQL pesado."""

    def __init__(self, hist_pop: dict[str, float]):
        super().__init__(daemon=True)
        self.hist_pop = hist_pop
        self.state = build_state(hist_pop)

    def run(self) -> None:
        while True:
            time.sleep(REFRESH_S)
            try:
                t = time.time()
                self.state = build_state(self.hist_pop)  # reemplazo atómico de la referencia
                print(f"estado refrescado en {time.time() - t:.1f}s: {len(self.state.available)} productos disponibles")
            except Exception as e:
                print(f"refresco fallido, sigo con el estado anterior: {e}")


def main() -> None:
    print("cargando popularidad histórica...")
    hist_pop = score_counts(query(HIST_POPULARITY, timeout=300))
    coviews = load_coviews()
    refresher = Refresher(hist_pop)
    refresher.start()
    print(f"listo: {len(refresher.state.available)} productos disponibles. Esperando requests...")

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
                s = refresher.state
                answers = []
                for req in fresh:
                    seen[req["request_id"]] = None
                    try:
                        version = pick_variant(req)
                        items = recommend(req, s, coviews, VARIANTS[version])
                    except Exception as e:
                        print(f"recommend falló ({e}), uso top global")
                        version, items = "fallback-propio", s.top_global[:10]
                    answers.append({"request_id": req["request_id"], "items": items, "model_version": version})
                push(answers)
                answered += len(answers)
                if answered % 300 < len(answers):
                    print(f"{answered} respondidas")
            if len(seen) > 50_000:  # no crecer sin límite durante toda la noche
                seen = dict.fromkeys(list(seen)[-10_000:])  # conserva las más recientes
        except Exception as e:  # sigue vivo: los compradores no paran
            print(f"error: {e}")
            time.sleep(1)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
