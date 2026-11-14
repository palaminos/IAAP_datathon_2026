"""¿Aporta el TEXTO de los productos algo más que la subcategoría y la marca? Solo lee.

    python analysis/text_similarity.py     # desde la raíz del repo

Convierte cada descripción en un vector (TF-IDF: qué palabras tiene) y mira si los productos de texto
parecido se ven juntos en las mismas sesiones (co-vistas del histórico) más que otros productos de su
MISMA subcategoría. Si no hay diferencia, un embedding del texto no añadiría información.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import recommender as rec  # noqa: E402

rows = rec.query("SELECT product_id::String AS p, subcategory::String AS sub, brand::String AS brand, description::String AS d, price_eur::Float64 AS price FROM shop.catalog", timeout=60)
ids = [r["p"] for r in rows]
idx = {p: i for i, p in enumerate(ids)}
sub = np.array([r["sub"] for r in rows]); brand = np.array([r["brand"] for r in rows])
print("Ejemplos de descripción:")
for r in rows[:6]:
    print(f"  [{r['sub']} | {r['brand']}] {r['d']}")

# Quitar del texto la marca y la subcategoría: lo que queda son los atributos
def attributes(r):
    t = r["d"].lower()
    for w in (r["brand"].lower(), r["sub"].lower()):
        t = t.replace(w, " ")
    return re.sub(r"[^a-z\- ]", " ", t)

texts = [attributes(r) for r in rows]
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, stop_words="english")
X = vec.fit_transform(texts)
print(f"\n{len(vec.vocabulary_)} términos distintos. Los más frecuentes:")
freq = Counter({t: int(n) for t, n in zip(vec.get_feature_names_out(), (X > 0).sum(0).A1)})
print("  " + ", ".join(f"{t} ({n})" for t, n in freq.most_common(25)))

cov = json.loads((rec.CACHE / "coviews_21d.json").read_text())
S = (X @ X.T).toarray()
rng = np.random.default_rng(0)
hit = {"texto más parecido (misma subcat.)": [], "al azar (misma subcat.)": [], "misma marca (misma subcat.)": [], "precio más parecido (misma subcat.)": []}
price = np.array([r["price"] for r in rows])
for p, qs in cov.items():
    i = idx.get(p)
    if i is None:
        continue
    same = np.where((sub == sub[i]) & (np.arange(len(ids)) != i))[0]
    if len(same) < 10:
        continue
    viewed = {idx[q] for q in qs if q in idx}
    k = 5
    top_text = same[np.argsort(-S[i, same])[:k]]
    hit["texto más parecido (misma subcat.)"].append(len(viewed & set(top_text)) / k)
    hit["al azar (misma subcat.)"].append(len(viewed & set(rng.choice(same, k, replace=False))) / k)
    sb = same[brand[same] == brand[i]]
    if len(sb) >= 2:
        hit["misma marca (misma subcat.)"].append(len(viewed & set(sb[:k])) / min(k, len(sb)))
    hit["precio más parecido (misma subcat.)"].append(len(viewed & set(same[np.argsort(np.abs(np.log(price[same] / price[i])))[:k]])) / k)
print("\n% de vecinos que están entre las 20 co-vistas del producto:")
base = np.mean(hit["al azar (misma subcat.)"])
for name, v in hit.items():
    print(f"  {name:<38} {100 * np.mean(v):5.1f} %   ({np.mean(v) / base:.2f}x el azar, n={len(v)})")
