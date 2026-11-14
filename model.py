"""Modelo v2: señales por (usuario, producto) y memoria de usuarios.

La misma función `featurize()` se usa para ENTRENAR (train.py, sobre shop.history_sessions) y para
SERVIR (recommender.py, en vivo): así las señales significan exactamente lo mismo en los dos sitios.

Para cada request se calcula una tabla de N productos × K señales; la puntuación de cada producto es
la suma de señal × peso, con los pesos aprendidos por una regresión logística (model_weights.json).
"""
from __future__ import annotations

import json
import math
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np

FEATURES = [
    # --- lo que sabemos del USUARIO (60 días de histórico + lo que hace hoy) ---
    "sub_share",       # % de la actividad del usuario que cae en la subcategoría del producto
    "sub_log",         # log(1 + nº de eventos del usuario en esa subcategoría)
    "cat_share",       # % de su actividad en la categoría del producto
    "brand_share",     # % de su actividad en la marca del producto
    "p_view",          # log(1 + veces que el usuario vio ESE producto)
    "p_cart",          # log(1 + veces que lo metió en el carrito)
    "p_buy",           # log(1 + veces que lo compró)
    # --- lo que está haciendo AHORA (recent_views de la request) ---
    "in_session",      # el producto está entre los vistos en esta sesión
    "is_last",         # es el último visto
    "same_sub",        # misma subcategoría que el último visto
    "same_cat",        # misma categoría que el último visto
    "sess_sub_share",  # % de las vistas de la sesión que son de su subcategoría
    "coview",          # co-vistas: se suele ver en la misma sesión que lo que acaba de ver
    # --- el PRODUCTO en sí ---
    "pop_sub",         # 1/sqrt(posición en ventas dentro de su subcategoría)
    "pop_glob",        # posición en ventas global (1 = el más vendido, 0 = el menos)
    "log_price",       # log(precio)
    # --- añadidas en v6 (las versiones anteriores las tienen con peso 0) ---
    "mf",              # afinidad por "usuarios parecidos" (factorización usuario × producto, train_mf.py)
    "mf_z",            # la misma, normalizada dentro del usuario (cuánto destaca este producto para él)
    "cart_no_buy",     # 1 si lo metió en el carrito y no lo ha comprado
    "viewed_no_cart",  # 1 si lo ha visto pero nunca lo metió en el carrito ni lo compró
    # --- añadida en v8 ---
    "is_new_prod",     # 1 si el producto se lanzó durante el evento (new_product en shop.catalog_updates)
    # --- añadida en v9 ---
    "organic",         # salida de un GBM entrenado con la actividad orgánica de la tienda (train_organic.py)
    # --- añadida en v10 ---
    "on_sale",         # 1 si el producto está rebajado ahora (precio actual < 95 % de su precio base)
    # --- añadidas en v12 ---
    "cf",              # filtrado colaborativo: parecido a lo que el usuario metió en el carrito o compró (train_cf.py)
    "cf_any",          # 1 si tiene algún parecido con eso (los que no tienen ninguno casi nunca se compran)
]
K = len(FEATURES)
FI = {name: i for i, name in enumerate(FEATURES)}
STATIC = slice(FI["pop_sub"], FI["pop_sub"] + 3)  # columnas que no dependen de la request
WEIGHTS_FILE = Path(__file__).with_name("model_weights.json")
MF_FILE = Path(__file__).with_name("mf_factors.npz")
EVENT_MF = {"view": math.log1p(1), "add_to_cart": math.log1p(3), "purchase": math.log1p(5)}


CTX_FILE = Path(__file__).with_name("ctx_tables.json")
CF_FILE = Path(__file__).with_name("cf_sim.npz")
_CF: dict = {"n": 0, "S": None}   # matriz de similitud ya traducida a los números de este proceso


def load_cf(path: Path = CF_FILE) -> dict | None:
    """Similitud producto-producto (train_cf.py). None si no existe el fichero."""
    if not path.exists():
        return None
    d = np.load(path)
    return {"products": d["products"].tolist(), "i": d["i"], "j": d["j"], "v": d["v"], "scale": float(d["scale"])}


def cf_matrix(cf: dict, n: int):
    """Matriz dispersa n x n (fila = producto que le gustó, columna = candidato). Solo se rehace si el
    catálogo ha crecido: los números de los productos ya conocidos no cambian nunca."""
    if _CF["S"] is None or _CF["n"] != n:
        from scipy.sparse import csr_matrix
        lookup = np.array([PRODUCTS.idx.get(p, -1) for p in cf["products"]], np.int64)
        i, j = lookup[cf["i"]], lookup[cf["j"]]
        ok = (i >= 0) & (j >= 0)
        _CF["S"], _CF["n"] = csr_matrix((cf["v"][ok], (i[ok], j[ok])), shape=(n, n)), n
    return _CF["S"]
ORGANIC_FILE = Path(__file__).with_name("organic_gbm.pkl")
_ORGANIC: list = [False, None]   # [ya se intentó cargar, (modelo, columnas) o None]


def load_organic(path: Path = ORGANIC_FILE):
    """El GBM de la señal orgánica y las columnas que usa. None si no hay fichero o no se puede cargar."""
    if not _ORGANIC[0]:
        _ORGANIC[0] = True
        try:
            with path.open("rb") as f:
                d = pickle.load(f)
            _ORGANIC[1] = (d["model"], [FI[n] for n in d["features"]])
        except Exception as e:
            print(f"señal orgánica no disponible ({e!r}): valdrá 0")
    return _ORGANIC[1]


def add_organic(F: np.ndarray) -> np.ndarray:
    """Rellena la señal 'organic' de las filas de F (productos × señales) con la salida del GBM orgánico."""
    loaded = load_organic()
    if loaded is not None and len(F):
        gbm, cols = loaded
        F[:, FI["organic"]] = gbm.decision_function(F[:, cols])
    return F


def load_mf(path: Path = MF_FILE) -> dict | None:
    """Factores de usuarios y productos (train_mf.py). None si no existe el fichero."""
    if not path.exists():
        return None
    d = np.load(path)
    return {"V": dict(zip(d["products"].tolist(), d["V"])), "users": d["users"].tolist(), "U": d["U"].astype(np.float32),
            "k": d["V"].shape[1]}
MASK = (1 << 20) - 1  # contadores vistas/carritos/compras empaquetados en un entero


class Index:
    """Texto -> número, sin reordenar nunca: los números asignados son estables mientras viva el proceso."""

    def __init__(self) -> None:
        self.idx: dict[str, int] = {}
        self.names: list[str] = []

    def add(self, name: str) -> int:
        i = self.idx.get(name)
        if i is None:
            i = self.idx[name] = len(self.names)
            self.names.append(name)
        return i

    def __len__(self) -> int:
        return len(self.names)


PRODUCTS, SUBS, CATS, BRANDS = Index(), Index(), Index(), Index()


class Items:
    """El catálogo como columnas numéricas alineadas con PRODUCTS. Se reconstruye en cada refresco."""

    def __init__(self, product: dict[str, dict], popularity: dict[str, float], available: set[str],
                 mf: dict | None = None, cf: dict | None = None):
        for pid, info in product.items():
            PRODUCTS.add(pid), SUBS.add(info["subcategory"]), CATS.add(info["category"]), BRANDS.add(info.get("brand") or "")
        n = self.n = len(PRODUCTS)
        self.sub = np.zeros(n, np.int32)
        self.cat = np.zeros(n, np.int32)
        self.brand = np.zeros(n, np.int32)
        self.avail = np.zeros(n, bool)
        self.is_new = np.zeros(n, np.float32)
        self.on_sale = np.zeros(n, np.float32)
        price = np.ones(n)
        pop = np.zeros(n)
        for pid, info in product.items():
            i = PRODUCTS.idx[pid]
            self.sub[i], self.cat[i] = SUBS.idx[info["subcategory"]], CATS.idx[info["category"]]
            self.brand[i] = BRANDS.idx[info.get("brand") or ""]
            price[i] = max(float(info.get("price_eur") or 1.0), 0.5)
            pop[i] = popularity.get(pid, 0.0)
            self.avail[i] = pid in available
            self.is_new[i] = float(info.get("is_new") in (1, True, "1", "true"))
            base = float(info.get("base_price") or 0.0)
            self.on_sale[i] = float(base > 0 and price[i] < 0.95 * base)
        self.nsub, self.ncat = len(SUBS), len(CATS)
        self.sub2cat = np.zeros(self.nsub, np.int32)
        self.sub2cat[self.sub] = self.cat
        # Productos sin ventas todavía (nuevos): se les supone la popularidad mediana de su subcategoría,
        # para no enterrarlos antes de tener datos (cold start).
        for s in np.unique(self.sub[pop == 0]):
            known = pop[(self.sub == s) & (pop > 0)]
            if len(known):
                pop[(self.sub == s) & (pop == 0)] = np.median(known)
        order = np.argsort(-pop, kind="stable")
        rank_glob = np.empty(n)
        rank_glob[order] = np.arange(n)
        rank_sub = np.empty(n)
        seen_in_sub = np.zeros(self.nsub, np.int32)
        for i in order:
            seen_in_sub[self.sub[i]] += 1
            rank_sub[i] = seen_in_sub[self.sub[i]]
        self.static = np.stack([1 / np.sqrt(rank_sub), 1 - rank_glob / n, np.log(price)], axis=1).astype(np.float32)
        self.pop = pop
        self.cf = cf_matrix(cf, n) if cf is not None else None
        self.cf_scale = cf["scale"] if cf is not None else 1.0
        # Factores de cada producto para la señal de "usuarios parecidos" (ceros si es nuevo: aún no los tiene)
        self.mfV = None
        if mf is not None:
            self.mfV = np.zeros((n, mf["k"]), np.float32)
            self.mf_known = np.zeros(n, bool)   # productos con factores (los lanzados durante el evento no tienen)
            for pid, v in mf["V"].items():
                i = PRODUCTS.idx.get(pid)
                if i is not None:
                    self.mfV[i], self.mf_known[i] = v, True
            self.mf_sub_n = np.bincount(self.sub[self.mf_known], minlength=self.nsub).astype(np.float64)


class UserStore:
    """Memoria de usuarios: cuánto ha interactuado cada uno con cada subcategoría, marca y producto."""

    def __init__(self, capacity: int = 1 << 17, cols: int = 96) -> None:
        self.row: dict[str, int] = {}
        self.sub = np.zeros((capacity, cols), np.uint16)
        self.brand = np.zeros((capacity, cols), np.uint16)
        self.prod: dict[int, dict[int, int]] = {}   # fila de usuario -> {producto: contadores empaquetados}
        self.watermark = ""                          # último ts de shop.events ya incorporado
        self.mf: np.ndarray | None = None            # factores de "usuarios parecidos" por usuario

    def init_mf(self, mf: dict | None) -> None:
        """Carga los factores del histórico. Llamar después de cargar el historial y antes de los eventos de hoy."""
        if mf is None:
            return
        self.mf = np.zeros((len(self.sub), mf["k"]), np.float32)
        for uid, vec in zip(mf["users"], mf["U"]):
            r = self.row.get(uid)
            if r is not None:
                self.mf[r] = vec

    def _row(self, uid: str) -> int:
        r = self.row.get(uid)
        if r is None:
            r = len(self.row)
            if r >= len(self.sub):  # doblar capacidad
                self.sub = np.concatenate([self.sub, np.zeros_like(self.sub)])
                self.brand = np.concatenate([self.brand, np.zeros_like(self.brand)])
                if self.mf is not None:
                    self.mf = np.concatenate([self.mf, np.zeros_like(self.mf)])
            self.row[uid] = r
        return r

    def ensure_cols(self) -> None:
        need = max(len(SUBS), len(BRANDS))
        if need > self.sub.shape[1]:
            pad = ((0, 0), (0, need + 32 - self.sub.shape[1]))
            self.sub, self.brand = np.pad(self.sub, pad), np.pad(self.brand, pad)

    @staticmethod
    def _bump(m: np.ndarray, r: int, c: int, n: int) -> None:
        m[r, c] = min(int(m[r, c]) + n, 65535)

    def add_event(self, uid: str, pid: str, etype: str, items: Items) -> None:
        """Un evento en vivo."""
        i = PRODUCTS.idx.get(pid)
        if i is None or i >= items.n:
            return
        r = self._row(uid)
        self._bump(self.sub, r, items.sub[i], 1)
        self._bump(self.brand, r, items.brand[i], 1)
        d = self.prod.setdefault(r, {})
        shift = 0 if etype == "view" else 20 if etype == "add_to_cart" else 40
        if i in d or shift or len(d) < 80:  # tope por usuario para no gastar memoria en vistas sueltas
            d[i] = d.get(i, 0) + (1 << shift)
        if self.mf is not None and items.mfV is not None:  # lo de hoy también mueve sus "gustos"
            self.mf[r] += EVENT_MF.get(etype, 0.0) * items.mfV[i]

    # ------------------------------------------------------------------ carga desde RawTree
    def load(self, query: Callable[..., list[dict]], table: str, cond: str = "", chunks: int = 8,
             mfV: np.ndarray | None = None) -> None:
        """Agrega en RawTree (SQL) la actividad por usuario y la suma a la memoria.
        Con mfV (eventos de hoy), además actualiza los factores de "usuarios parecidos"."""
        self.ensure_cols()
        jobs = [(kind, i) for kind in ("sub", "brand", "prod") for i in range(chunks)]

        def run(job):
            kind, i = job
            sql = (USER_PROD if kind == "prod" else USER_DIM).format(
                table=table, cat=CAT_NOW, dim=kind, cond=cond, chunks=chunks, i=i)
            return kind, query(sql, timeout=600)

        with ThreadPoolExecutor(4) as pool:
            for kind, rows in pool.map(run, jobs):
                for row in rows:
                    r = self._row(row["u"])
                    if kind == "prod":
                        d = self.prod.setdefault(r, {})
                        for p, v, a, b in zip(row["p"], row["v"], row["a"], row["b"]):
                            i = PRODUCTS.idx.get(p)
                            if i is not None:
                                d[i] = d.get(i, 0) + int(v) + (int(a) << 20) + (int(b) << 40)
                                if mfV is not None and self.mf is not None and i < len(mfV):
                                    self.mf[r] += math.log1p(int(v) + 3 * int(a) + 5 * int(b)) * mfV[i]
                    else:
                        index, m = (SUBS, self.sub) if kind == "sub" else (BRANDS, self.brand)
                        for name, n in zip(row["k"], row["n"]):
                            c = index.idx.get(name)
                            if c is not None:
                                self._bump(m, r, c, int(n))

    def save(self, path: Path) -> None:
        n = len(self.row)
        path.parent.mkdir(exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"row": self.row, "sub": self.sub[:n], "brand": self.brand[:n], "prod": self.prod,
                         "products": PRODUCTS.names, "subs": SUBS.names, "brands": BRANDS.names}, f, protocol=4)

    def load_file(self, path: Path) -> None:
        """Carga una memoria guardada (en un almacén vacío), traduciendo sus números a los de este proceso."""
        with path.open("rb") as f:
            d = pickle.load(f)
        n = len(d["row"])
        while n > len(self.sub):
            self.sub = np.concatenate([self.sub, np.zeros_like(self.sub)])
            self.brand = np.concatenate([self.brand, np.zeros_like(self.brand)])
        self.ensure_cols()
        self.row = d["row"]
        for index, names, saved, m in ((SUBS, d["subs"], d["sub"], self.sub), (BRANDS, d["brands"], d["brand"], self.brand)):
            for j, name in enumerate(names):
                c = index.idx.get(name)
                if c is not None and j < saved.shape[1]:
                    m[:n, c] = saved[:, j]
        pmap = [PRODUCTS.idx.get(name, -1) for name in d["products"]]
        self.prod = {r: {pmap[i]: v for i, v in mine.items() if pmap[i] >= 0} for r, mine in d["prod"].items()}


# Catálogo actual (shop.catalog + novedades) para saber subcategoría y marca de cada producto
CAT_NOW = """(SELECT p, argMax(sub, v) AS sub, argMax(brand, v) AS brand FROM (
    SELECT product_id::String AS p, subcategory::String AS sub, brand::String AS brand, '' AS v FROM shop.catalog
    UNION ALL
    SELECT product_id::String, subcategory::String, brand::String, toString(ts::DateTime64(3)) FROM shop.catalog_updates
  ) GROUP BY p)"""

# Por usuario: nº de eventos en cada subcategoría (dim=sub) o marca (dim=brand)
USER_DIM = """
SELECT u, groupArray(k) AS k, groupArray(n) AS n FROM (
    SELECT e.user_id::String AS u, c.{dim} AS k, count() AS n
    FROM {table} AS e INNER JOIN {cat} AS c ON c.p = e.product_id::String
    WHERE cityHash64(e.user_id::String) % {chunks} = {i} {cond}
    GROUP BY u, k
) GROUP BY u"""

# Por usuario: sus productos "fuertes" (visto 2+ veces, o en carrito, o comprado). Primero TODOS los
# comprados (para poder excluirlos sin que se escape ninguno) y luego los demás, hasta 150 en total.
USER_PROD = """
SELECT u, groupArray(p) AS p, groupArray(v) AS v, groupArray(a) AS a, groupArray(b) AS b FROM (
    SELECT e.user_id::String AS u, e.product_id::String AS p,
           countIf(e.event_type = 'view') AS v, countIf(e.event_type = 'add_to_cart') AS a,
           countIf(e.event_type = 'purchase') AS b
    FROM {table} AS e
    WHERE cityHash64(e.user_id::String) % {chunks} = {i} {cond}
    GROUP BY u, p
    HAVING v >= 2 OR a >= 1 OR b >= 1
    ORDER BY b > 0 DESC, v + 3 * a + 5 * b DESC
    LIMIT 150 BY u
) GROUP BY u"""


def featurize(user_id: str, recent_views: list[dict] | None, items: Items, users: UserStore | None,
              coviews: dict[int, list[int]]) -> np.ndarray:
    """Tabla N productos × K señales para una request."""
    n = items.n
    F = np.zeros((n, K), np.float32)
    F[:, STATIC] = items.static
    F[:, FI["is_new_prod"]] = items.is_new
    F[:, FI["on_sale"]] = items.on_sale

    row = users.row.get(user_id) if users is not None else None
    if row is not None:
        sc = users.sub[row].astype(np.float32)
        total = float(sc.sum())
        if total > 0:
            per_item = sc[items.sub]
            F[:, 0] = per_item / total
            F[:, 1] = np.log1p(per_item)
            F[:, 2] = np.bincount(items.sub2cat, weights=sc[:items.nsub], minlength=items.ncat)[items.cat] / total
            F[:, 3] = users.brand[row][items.brand] / total
        mine = users.prod.get(row)
        if mine:
            for i, packed in list(mine.items()):
                if i < n:
                    v, a, b = packed & MASK, (packed >> 20) & MASK, packed >> 40
                    F[i, 4], F[i, 5], F[i, 6] = math.log1p(v), math.log1p(a), math.log1p(b)
                    if not b:
                        F[i, FI["cart_no_buy" if a else "viewed_no_cart"]] = 1.0
            if items.cf is not None:
                liked = [i for i, packed in list(mine.items()) if i < n and ((packed >> 20) & MASK or packed >> 40)]
                if liked:   # suma de parecidos con todo lo que metió en el carrito o compró
                    sim = np.asarray(items.cf[liked].sum(0)).ravel()
                    F[:, FI["cf"]] = np.log1p(items.cf_scale * sim)
                    F[:, FI["cf_any"]] = sim > 0
        if users.mf is not None and items.mfV is not None and row < len(users.mf):
            vec = users.mf[row]
            known = items.mf_known
            if vec.any() and known.any():
                aff = items.mfV @ vec
                mu, sd = float(aff[known].mean()), float(aff[known].std()) + 1e-9
                if not known.all():
                    # Un producto nuevo no tiene factores: su afinidad saldría 0, por debajo de casi todo, y el
                    # modelo lo enterraba (v6 enseñaba productos nuevos 4 veces menos de lo que se venden).
                    # Se le supone la afinidad media que este usuario tiene por su subcategoría.
                    sub_sum = np.bincount(items.sub[known], weights=aff[known], minlength=items.nsub)
                    sub_mean = np.where(items.mf_sub_n > 0, sub_sum / np.maximum(items.mf_sub_n, 1), mu)
                    aff = np.where(known, aff, sub_mean[items.sub])
                F[:, FI["mf"]] = aff
                F[:, FI["mf_z"]] = (aff - mu) / sd

    idx = [PRODUCTS.idx.get(v["product_id"]) for v in recent_views or []]
    idx = [i for i in idx if i is not None and i < n]
    if idx:
        last = idx[-1]
        F[idx, 7] = 1
        F[last, 8] = 1
        F[:, 9] = items.sub == items.sub[last]
        F[:, 10] = items.cat == items.cat[last]
        F[:, 11] = np.bincount(items.sub[idx], minlength=items.nsub)[items.sub] / len(idx)
        for k, p in enumerate(reversed(idx[-10:])):  # la vista más reciente pesa más
            decay = 0.7 ** k
            for rank, q in enumerate(coviews.get(p, ())):
                if q < n:
                    F[q, 12] += decay / (rank + 1)
    return F


def index_coviews(coviews: dict[str, list[str]]) -> dict[int, list[int]]:
    """Co-vistas con números en vez de IDs de producto."""
    out = {}
    for p, qs in coviews.items():
        if p in PRODUCTS.idx:
            out[PRODUCTS.idx[p]] = [PRODUCTS.idx[q] for q in qs if q in PRODUCTS.idx]
    return out


def load_weights(path: Path = WEIGHTS_FILE) -> np.ndarray:
    saved: dict[str, Any] = json.loads(path.read_text())["weights"]
    return np.array([saved.get(name, 0.0) for name in FEATURES], np.float32)


def top10(score: np.ndarray, items: Items, blocked: set[str] = frozenset()) -> list[str]:
    """Los 10 productos disponibles con más puntuación."""
    score = np.where(items.avail, score, -np.inf)
    cand = np.argpartition(-score, 30)[:30]
    cand = cand[np.argsort(-score[cand], kind="stable")]
    out = [PRODUCTS.names[i] for i in cand if np.isfinite(score[i]) and PRODUCTS.names[i] not in blocked]
    return out[:10]
