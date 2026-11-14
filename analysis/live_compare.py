"""Compara dos tablas de señales (cache/*.npz de analysis/live_matrix.py) sobre las MISMAS respuestas.

    python analysis/live_compare.py live_matrix.npz live_matrix_s15.npz --after "2026-11-14 14:00:00"

Sirve para probar otra forma de calcular una señal (p. ej. otra tabla de parecidos entre productos): se
reproduce el día con cada variante y aquí se mide cuál predice mejor la compra. Solo se usan las respuestas
posteriores a --after (hora UTC); de ellas, el primer 70 % entrena y el último 30 % prueba. Pesos propios
para compradores nuevos y conocidos, como v14.
"""
import datetime as dt
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.simplefilter("ignore")
import recommender as rec  # noqa: E402

files = [a for a in sys.argv[1:] if a.endswith(".npz")]
after = sys.argv[sys.argv.index("--after") + 1] if "--after" in sys.argv else "2026-11-14 00:00:00"
cut_ms = dt.datetime.fromisoformat(after).replace(tzinfo=dt.timezone.utc).timestamp() * 1000
before = sys.argv[sys.argv.index("--before") + 1] if "--before" in sys.argv else "2099-01-01 00:00:00"
end_ms = dt.datetime.fromisoformat(before).replace(tzinfo=dt.timezone.utc).timestamp() * 1000
DROP = ["cart_no_buy", "viewed_no_cart", "organic", "on_sale"]
onehot = lambda s: np.eye(10)[s][:, 1:]


def evaluate(path):
    d = np.load(rec.CACHE / path)
    keep = (d["when"] >= cut_ms) & (d["when"] < end_ms)
    X, slot, buy, when, new = d["X"][keep].astype(np.float64), d["slot"][keep], d["buy"][keep], d["when"][keep], d["new"][keep].astype(bool)
    cols = [i for i, n in enumerate(d["features"]) if n not in DROP]
    q = np.quantile(when, 0.70)
    tr, te = when < q, when >= q
    out = {}
    score = np.zeros(len(X))
    for name, m in (("nuevos", new), ("conocidos", ~new)):
        Xm = X[tr & m][:, cols]
        mu, sd = Xm.mean(0), Xm.std(0) + 1e-9
        lr = LogisticRegression(C=1.0, max_iter=4000).fit(np.hstack([(Xm - mu) / sd, onehot(slot[tr & m])]), buy[tr & m])
        s = ((X[:, cols] - mu) / sd) @ lr.coef_[0][:len(cols)]
        score[m] = s[m]
        Z = np.hstack([s[:, None], onehot(slot)])
        cal = LogisticRegression(C=10.0, max_iter=2000).fit(Z[tr & m], buy[tr & m])
        p = cal.predict_proba(Z[te & m])[:, 1]
        S, Y = s.reshape(-1, 10), buy.reshape(-1, 10)
        R = (te & m).reshape(-1, 10)[:, 0] & Y.any(1)
        sj = S[R][np.arange(R.sum()), Y[R].argmax(1)][:, None]
        rank = ((S[R] > sj).sum(1) + ((S[R] == sj).sum(1) - 1) / 2 + 1).mean()
        out[name] = (roc_auc_score(buy[te & m], p), log_loss(buy[te & m], p), rank, int(buy[tr & m].sum()), int(buy[te & m].sum()))
    return out, len(X) // 10


print(f"respuestas entre {after} y {before} UTC; entrenar 70 % / probar 30 %\n")
print(f"{'tabla':<28}{'respuestas':>11} | {'NUEVOS: AUC':>12}{'log-loss':>10}{'puesto':>8}{'compras':>10} | {'CONOCIDOS: AUC':>15}{'log-loss':>10}{'puesto':>8}{'compras':>10}")
for f in files:
    r, n = evaluate(f)
    a, b = r["nuevos"], r["conocidos"]
    print(f"{f:<28}{n:>11} | {a[0]:>12.4f}{a[1]:>10.5f}{a[2]:>8.2f}{f'{a[3]}/{a[4]}':>10} | {b[0]:>15.4f}{b[1]:>10.5f}{b[2]:>8.2f}{f'{b[3]}/{b[4]}':>10}")
print("\npuesto = puesto medio (1-10) del producto comprado entre los 10 mostrados; menos es mejor")
