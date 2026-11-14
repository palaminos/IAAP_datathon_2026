"""Sensibilidad al precio y gustos por categoría según el CONTEXTO de la request (dispositivo, región, página).

    python train_context.py        # solo lee de RawTree; guarda ctx_tables.json (va en el repo)

Idea (la sugiere el propio material de los organizadores): lo que se compra depende de dónde y desde qué
se mira. En el histórico del carrusel (2 M de carruseles, 20 M de impresiones), en Android un producto de
más de 80 EUR se compra 0,32 veces lo que uno de menos de 10; en iOS y web, 0,77. Por regiones va de 0,37
(Andalucía) a 0,88 (Galicia). El modelo no usaba ni dispositivo ni región.

Qué hace: agrega en RawTree las impresiones del histórico por (dispositivo, región, página, puesto, tramo
de precio, categoría) y ajusta una regresión logística con las interacciones contexto x precio y
contexto x categoría. De ahí sale, para cada contexto, cuánto sumar (en logit) a cada tramo de precio y a
cada categoría. Las tablas se centran para que solo REORDENEN: no cambian la media, solo qué va antes.

Comprobación incluida: ¿se cumple lo mismo en NUESTRO feedback de hoy? (coeficiente ~1 = se cumple igual).
"""
import json
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression

import model
from monitor import TEAM, query

warnings.simplefilter("ignore")
EDGES = [8, 12, 18, 27, 40, 60, 90, 150]     # tramos de precio en EUR (9 tramos)
PB = "multiIf(" + ", ".join(f"price < {e}, {i}" for i, e in enumerate(EDGES)) + f", {len(EDGES)})"
HIST = f"""
WITH cat AS (SELECT product_id::String p, price_eur::Float64 price, category::String c FROM shop.catalog)
SELECT device, region, page, slot, {PB} pb, c, count() n, sum(buy) b
FROM (SELECT context.device::String device, context.region::String region, context.page::String page, item, slot,
             ifNull(purchased::String = item, 0) buy
      FROM shop.history_sessions ARRAY JOIN shown::Array(String) AS item, arrayEnumerate(shown::Array(String)) AS slot) imp
INNER JOIN cat ON cat.p = imp.item
GROUP BY device, region, page, slot, pb, c"""
LIVE = f"""
WITH cat AS (SELECT p, argMax(price, v) price, argMax(c, v) c FROM (
    SELECT product_id::String p, price_eur::Float64 price, category::String c, '' v FROM shop.catalog
    UNION ALL SELECT product_id::String, price_eur::Float64, category::String, toString(ts::DateTime64(3)) FROM shop.catalog_updates) GROUP BY p)
SELECT device, region, page, slot, {PB} pb, c, count() n, sum(buy) b
FROM (SELECT r.context.device::String device, r.context.region::String region, r.context.page::String page, item, slot,
             f.purchased::String = item AS buy
      FROM {TEAM}.feedback f INNER JOIN shop.requests r ON r.request_id::String = f.request_id::String
      ARRAY JOIN f.shown::Array(String) AS item, arrayEnumerate(f.shown::Array(String)) AS slot
      WHERE f.request_id != '__init__') imp
INNER JOIN cat ON cat.p = imp.item
GROUP BY device, region, page, slot, pb, c"""


def design(rows, levels):
    """Matriz de diseño con efectos principales e interacciones contexto x precio y contexto x categoría."""
    idx = {k: {v: i for i, v in enumerate(vals)} for k, vals in levels.items()}
    blocks = [("slot", None), ("device", None), ("region", None), ("page", None), ("pb", None), ("c", None),
              ("device", "pb"), ("region", "pb"), ("page", "pb"), ("device", "c"), ("region", "c")]
    offs, k = {}, 0
    for a, b in blocks:
        offs[(a, b)] = k
        k += len(levels[a]) * (len(levels[b]) if b else 1)
    X = np.zeros((len(rows), k), np.float32)
    for i, r in enumerate(rows):
        for a, b in blocks:
            j = idx[a][r[a]] * (len(levels[b]) if b else 1) + (idx[b][r[b]] if b else 0)
            X[i, offs[(a, b)] + j] = 1
    return X, offs


def fit(rows, levels, C=1.0):
    X, offs = design(rows, levels)
    n, b = np.array([float(r["n"]) for r in rows]), np.array([float(r["b"]) for r in rows])
    lr = LogisticRegression(C=C, max_iter=3000).fit(np.vstack([X, X]), np.r_[np.ones(len(X)), np.zeros(len(X))],
                                                   sample_weight=np.r_[b, n - b] + 1e-9)
    return lr.coef_[0], offs


def centered(coef, offs, levels, rows, a, b):
    """Tabla contexto x (precio|categoría) centrada: para cada columna, media ponderada por tráfico = 0."""
    A, B = levels[a], levels[b]
    T = coef[offs[(a, b)]:offs[(a, b)] + len(A) * len(B)].reshape(len(A), len(B)).copy()
    w = np.array([sum(float(r["n"]) for r in rows if r[a] == v) for v in A])
    T -= (w[:, None] * T).sum(0) / w.sum()          # quita lo común a todos los contextos (lo ya recoge el modelo)
    T -= T.mean(1, keepdims=True)                   # y el nivel de cada contexto (no cambia el orden dentro de una request)
    return {v: [round(float(x), 4) for x in T[i]] for i, v in enumerate(A)}


if __name__ == "__main__":
    hist = query(HIST, timeout=900)
    for r in hist:
        r["pb"], r["slot"] = int(r["pb"]), int(r["slot"])
    levels = {k: sorted({r[k] for r in hist}) for k in ("slot", "device", "region", "page", "pb", "c")}
    print(f"histórico: {len(hist)} celdas, {sum(float(r['n']) for r in hist) / 1e6:.1f} M impresiones, {int(sum(float(r['b']) for r in hist))} compras")
    coef, offs = fit(hist, levels)
    tables = {"edges": EDGES, "categories": levels["c"],
              "device_price": centered(coef, offs, levels, hist, "device", "pb"),
              "region_price": centered(coef, offs, levels, hist, "region", "pb"),
              "page_price": centered(coef, offs, levels, hist, "page", "pb"),
              "device_cat": centered(coef, offs, levels, hist, "device", "c"),
              "region_cat": centered(coef, offs, levels, hist, "region", "c")}
    head = "".join(f"{('<' + str(e)):>7}" for e in EDGES) + f"{'>=' + str(EDGES[-1]):>7}"
    for name in ("device_price", "region_price", "page_price"):
        print(f"\n{name} (logit que se suma a cada tramo de precio):\n  {'':<22}{head}")
        for k, v in tables[name].items():
            print(f"  {k:<22}" + "".join(f"{x:>7.2f}" for x in v))
    for name in ("device_cat", "region_cat"):
        t = np.array(list(tables[name].values()))
        print(f"\n{name}: rango de los ajustes por categoría: de {t.min():+.2f} a {t.max():+.2f} (desviación típica {t.std():.2f})")

    # ---- ¿se cumple en nuestro feedback de hoy? ----
    live = query(LIVE, timeout=900)
    for r in live:
        r["pb"], r["slot"] = int(r["pb"]), int(r["slot"])
    live = [r for r in live if all(r[k] in levels[k] for k in ("slot", "device", "region", "page", "pb", "c"))]
    ci = {c: i for i, c in enumerate(levels["c"])}

    def adj(r, parts):
        s = 0.0
        if "price" in parts:
            s += tables["device_price"][r["device"]][r["pb"]] + tables["region_price"][r["region"]][r["pb"]] + tables["page_price"][r["page"]][r["pb"]]
        if "cat" in parts:
            s += tables["device_cat"][r["device"]][ci[r["c"]]] + tables["region_cat"][r["region"]][ci[r["c"]]]
        return s

    n, b = np.array([float(r["n"]) for r in live]), np.array([float(r["b"]) for r in live])
    print(f"\nfeedback de hoy: {int(n.sum())} impresiones, {int(b.sum())} compras")
    # efectos principales (puesto, dispositivo, región, página, precio, categoría) + el ajuste del histórico como una señal más
    X, offs_l = design(live, levels)
    main = X[:, :offs_l[("device", "pb")]]
    for parts in (("price",), ("cat",), ("price", "cat")):
        a = np.array([adj(r, parts) for r in live])[:, None]
        Z = np.hstack([main, a])
        lr = LogisticRegression(C=10.0, max_iter=3000).fit(np.vstack([Z, Z]), np.r_[np.ones(len(Z)), np.zeros(len(Z))], sample_weight=np.r_[b, n - b] + 1e-9)
        # error típico aproximado del coeficiente (información de Fisher de esa columna, con el resto fijo)
        p = lr.predict_proba(Z)[:, 1]
        se = 1 / np.sqrt((n * p * (1 - p) * (a[:, 0] - np.average(a[:, 0], weights=n)) ** 2).sum())
        print(f"  ajuste por {' + '.join(parts):<12}: coeficiente en vivo {lr.coef_[0][-1]:+.2f} (± {se:.2f})   [1 = se cumple igual que en el histórico; 0 = no se cumple]")
    if "--no-save" not in sys.argv:
        model.CTX_FILE.write_text(json.dumps(tables, indent=1, ensure_ascii=False))
        print(f"\nguardado {model.CTX_FILE.name}")
