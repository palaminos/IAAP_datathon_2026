"""Reproduce el día (train_live.live_examples) y guarda la tabla de señales en cache/live_matrix.npz,
para poder probar ideas en segundos sin repetir la reproducción (2-3 min). Solo lee de RawTree.

    python analysis/live_matrix.py
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
np.savez_compressed(rec.CACHE / "live_matrix.npz", X=X, slot=slot, click=click, buy=buy, when=when,
                    explored=tl.EXPLORED_ROWS, features=np.array(model.FEATURES), **tl.META)
print(f"{len(X) // 10} respuestas, {int(buy.sum())} compras, {int(click.sum())} clics, {X.shape[1]} señales -> cache/live_matrix.npz ({time.time() - t0:.0f}s)")
