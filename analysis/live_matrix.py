"""Reproduce el día (train_live.live_examples) y guarda la tabla de señales en cache/live_matrix.npz,
para poder probar ideas en segundos sin repetir la reproducción (2-3 min). Solo lee de RawTree.

    python analysis/live_matrix.py [--cf-fill] [--cf-file cache/otra.npz] [--save-as nombre.npz]
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model  # noqa: E402
import recommender as rec  # noqa: E402
import train_live as tl  # noqa: E402

t0 = time.time()
X, slot, click, buy, when = tl.live_examples()
out = rec.CACHE / (sys.argv[sys.argv.index("--save-as") + 1] if "--save-as" in sys.argv else "live_matrix.npz")
np.savez_compressed(out, X=X, slot=slot, click=click, buy=buy, when=when,
                    explored=tl.EXPLORED_ROWS, features=np.array(model.FEATURES), names=np.array(model.PRODUCTS.names), cf_fill=tl.CF_FILL,
                    cf_file=Path(tl._arg("--cf-file")).name if tl._arg("--cf-file") else "", **tl.META)
print(f"{len(X) // 10} respuestas, {int(buy.sum())} compras, {int(click.sum())} clics, {X.shape[1]} señales -> {out} ({time.time() - t0:.0f}s)")
