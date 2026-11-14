"""Tests sin credenciales ni red: lo esencial que no puede romperse.

    python -m unittest discover -s tests -v      # desde la raíz del repo
"""
import os
import sys
import unittest
from collections import Counter
from pathlib import Path

os.environ.setdefault("TEAM_ID", "team_test")
os.environ.setdefault("RAWTREE_API_KEY", "test")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

import model  # noqa: E402
import recommender as rec  # noqa: E402

PRODUCT = {f"p{i:03d}": {"category": f"C{i % 3}", "subcategory": f"S{i % 6}", "brand": f"B{i % 4}", "price_eur": 5 + i}
           for i in range(60)}
AVAILABLE = set(PRODUCT) - {"p000", "p001"}
STATE = rec.make_state(PRODUCT, AVAILABLE, {p: 100 - i for i, p in enumerate(PRODUCT)}, {})
WEIGHTS = np.ones(model.K, np.float32)


def is_valid(items, blocked=frozenset()):
    return len(items) == 10 and len(set(items)) == 10 and set(items) <= AVAILABLE and not set(items) & set(blocked)


class Basics(unittest.TestCase):
    def test_valid_fills_and_filters(self):
        self.assertTrue(is_valid(rec.valid([], STATE, set())))                              # vacío -> populares
        self.assertTrue(is_valid(rec.valid(["p000", "p005", "p005", "nope"], STATE, set())))  # agotado, repetido, inexistente
        blocked = set(STATE.top_global[:5])
        self.assertTrue(is_valid(rec.valid(STATE.top_global[:10], STATE, blocked), blocked))

    def test_every_variant_answers_validly(self):
        users = model.UserStore(capacity=8)
        for _ in range(3):
            users.add_event("u1", "p010", "view", STATE.items)
        users.add_event("u1", "p011", "add_to_cart", STATE.items)
        users.add_event("u1", "p012", "purchase", STATE.items)
        m = rec.Models({}, {}, WEIGHTS, WEIGHTS, {"v6": WEIGHTS})
        reqs = [{"request_id": "r1", "user_id": "u1", "recent_views": [{"product_id": "p010"}]},
                {"request_id": "r2", "user_id": "desconocido", "recent_views": []},
                {"request_id": "r3", "user_id": "u1", "recent_views": [{"product_id": "no-existe"}]}]
        for version in ("v1-coview-trend", "v2-user-lr", "v3-no-rebuy", "v4-live-weights", "v5-cart-first", "v6-similar-users"):
            for req in reqs:
                out = rec.respond(req, version, STATE, {"p020"}, users, m)
                self.assertTrue(is_valid(out["items"], {"p020"}), (version, req["request_id"], out))
                self.assertEqual(out["model_version"], version)
        self.assertNotIn("p012", rec.respond(reqs[0], "v3-no-rebuy", STATE, set(), users, m)["items"])   # ya comprado
        self.assertEqual(rec.respond(reqs[0], "v5-cart-first", STATE, set(), users, m)["items"][0], "p011")  # carrito primero

    def test_split_is_stable_and_balanced(self):
        self.assertEqual(rec.pick_variant({"user_id": "u_1", "request_id": "a"}), rec.pick_variant({"user_id": "u_1", "request_id": "b"}))
        counts = Counter(rec.pick_variant({"user_id": f"u_{i}", "request_id": "x"}) for i in range(4000))
        total = sum(rec.VARIANTS.values())
        for name, share in rec.VARIANTS.items():
            self.assertAlmostEqual(counts[name] / 4000, share / total, delta=0.04)


if __name__ == "__main__":
    unittest.main()
