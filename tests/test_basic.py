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
        m = rec.Models({}, {}, WEIGHTS, WEIGHTS, {"v6": WEIGHTS, "v8": WEIGHTS, "v9": WEIGHTS, "v10": WEIGHTS})
        reqs = [{"request_id": "r1", "user_id": "u1", "recent_views": [{"product_id": "p010"}]},
                {"request_id": "r2", "user_id": "desconocido", "recent_views": []},
                {"request_id": "r3", "user_id": "u1", "recent_views": [{"product_id": "no-existe"}]}]
        for version in ("v1-coview-trend", "v2-user-lr", "v3-no-rebuy", "v4-live-weights", "v5-cart-first", "v6-similar-users", "v8-new-products", "v9-organic", "v10-sale"):
            for req in reqs:
                out = rec.respond(req, version, STATE, {"p020"}, users, m)
                self.assertTrue(is_valid(out["items"], {"p020"}), (version, req["request_id"], out))
                self.assertEqual(out["model_version"], version)
        self.assertNotIn("p012", rec.respond(reqs[0], "v3-no-rebuy", STATE, set(), users, m)["items"])   # ya comprado
        # "v10" no debe confundirse con "v1": tiene que usar sus propios pesos (aquí, premiar solo la rebaja)
        only_sale = np.zeros(model.K, np.float32)
        only_sale[model.FI["on_sale"]] = 1.0
        STATE.items.on_sale[model.PRODUCTS.idx["p033"]] = 1.0
        m10 = rec.Models({}, {}, WEIGHTS, WEIGHTS, {"v10": only_sale})
        self.assertEqual(rec.respond(reqs[1], "v10-sale", STATE, set(), users, m10)["items"][0], "p033")
        STATE.items.on_sale[model.PRODUCTS.idx["p033"]] = 0.0
        self.assertEqual(rec.respond(reqs[0], "v5-cart-first", STATE, set(), users, m)["items"][0], "p011")  # carrito primero

    def test_explore_is_valid_random_and_logged(self):
        users = model.UserStore(capacity=8)
        users.add_event("u1", "p012", "purchase", STATE.items)
        m = rec.Models({}, {}, WEIGHTS, WEIGHTS, {base: WEIGHTS for base in rec.EXPLORE.values()})
        m.explore_q = rec.queue.Queue()
        req = {"request_id": "r1", "user_id": "u1", "recent_views": [{"product_id": "p010"}]}
        outs = [rec.respond(req, "v7-explore", STATE, {"p020"}, users, m) for _ in range(5)]
        for out in outs:
            self.assertTrue(is_valid(out["items"], {"p020"}))
            self.assertEqual(out["model_version"], "v7-explore")
            self.assertNotIn("p012", out["items"])                                   # nunca lo ya comprado
        self.assertGreater(len({tuple(o["items"]) for o in outs}), 1)                # el sorteo cambia la lista
        log = m.explore_q.get_nowait()
        self.assertEqual(log["kind"], "explore")
        self.assertEqual(log["request_id"], "r1")
        self.assertEqual(log["shown"], outs[0]["items"])
        self.assertEqual(len(log["candidates"]), rec.EXPLORE_TOP)
        self.assertEqual(len(log["scores"]), len(log["candidates"]))
        self.assertTrue(set(log["shown"]) <= set(log["candidates"]))
        self.assertNotIn("p012", log["candidates"])
        self.assertNotIn("p020", log["candidates"])
        self.assertEqual((len(log["p_slot"]), len(log["p_any"])), (10, 10))
        self.assertTrue(all(0 <= x <= 1 for x in log["p_slot"] + log["p_any"]))
        self.assertLess(log["log_prob_list"], 0)
        self.assertEqual(m.explore_q.qsize(), 4)

    def test_cart_top_and_split_weights(self):
        """v22: con compradores conocidos, el mejor producto del carrito va primero; con nuevos no se toca.
        Y una versión con pesos propios para nuevos (v22n) los usa solo con is_new_user."""
        users = model.UserStore(capacity=8)
        users.add_event("u1", "p040", "add_to_cart", STATE.items)      # en el carrito, sin comprar
        users.add_event("u1", "p041", "add_to_cart", STATE.items)
        users.add_event("u1", "p041", "purchase", STATE.items)         # en el carrito pero ya comprado
        pop = np.zeros(model.K, np.float32)
        pop[model.FI["pop_glob"]] = 1.0                                # solo popularidad: p040 nunca saldría primero
        price = np.zeros(model.K, np.float32)
        price[model.FI["log_price"]] = 1.0                             # pesos "de nuevos": lo más caro primero
        m = rec.Models({}, {}, WEIGHTS, WEIGHTS, {"v14": pop, "v22": pop, "v22n": price})
        m.options = {"v22": {"cart_top": "known"}}
        known = {"request_id": "r1", "user_id": "u1", "recent_views": [], "is_new_user": False}
        new = dict(known, is_new_user=True)
        base = rec.respond(known, "v14-cf-split", STATE, set(), users, m)["items"]
        out = rec.respond(known, "v22-cart-top", STATE, set(), users, m)["items"]
        self.assertNotEqual(base[0], "p040")
        self.assertEqual(out[0], "p040")                               # el carrito sin comprar, primero
        self.assertNotIn("p041", out)                                  # lo ya comprado, nunca
        self.assertEqual(out[1:], [p for p in base if p != "p040"][:9])  # el resto de la lista no cambia
        self.assertTrue(is_valid(out))
        self.assertEqual(rec.respond(known, "v22-cart-top", STATE, {"p040"}, users, m)["items"][0], base[0])  # agotado: como v14
        new_out = rec.respond(new, "v22-cart-top", STATE, set(), users, m)["items"]
        self.assertEqual(new_out[0], "p059")                           # comprador nuevo: pesos v22n y sin regla del carrito
        self.assertTrue(is_valid(new_out))

    def test_old_carousel_history_signals(self):
        """v32: lo que el comprador hizo con cada producto en los carruseles antiguos entra como señales, y una compra
        hecha desde un carrusel antiguo NO cuenta como 'ya comprado' (al contrario: va la primera si el peso lo dice)."""
        users = model.UserStore(capacity=8)
        users.add_event("u1", "p030", "view", STATE.items)
        row, P = users.row["u1"], model.PRODUCTS.idx
        pack = lambda shown, clicks, buys: shown | (clicks << 12) | (buys << 24)
        pairs = sorted([(P["p040"], pack(3, 1, 1)), (P["p041"], pack(2, 1, 0)), (P["p042"], pack(5, 0, 0))])
        indptr = np.zeros(row + 2, np.int64)
        indptr[row + 1:] = len(pairs)
        users.hc = (indptr, np.array([i for i, _ in pairs], np.int32), np.array([v for _, v in pairs], np.uint32))
        F = model.featurize("u1", [], STATE.items, users, {})
        FI = model.FI
        self.assertEqual([F[P["p040"], FI[k]] for k in ("hc_buy", "hc_click", "hc_any", "hc_n")], [1, 0, 0, 0])
        self.assertEqual([F[P["p041"], FI[k]] for k in ("hc_buy", "hc_click", "hc_any")], [0, 1, 0])
        self.assertEqual([F[P["p042"], FI[k]] for k in ("hc_buy", "hc_click", "hc_any")], [0, 0, 1])
        self.assertAlmostEqual(float(F[P["p042"], FI["hc_n"]]), float(np.log1p(5)), places=5)
        self.assertEqual(float(F[:, FI["hc_buy"]].sum() + F[:, FI["hc_click"]].sum() + F[:, FI["hc_any"]].sum()), 3.0)
        self.assertEqual(float(model.featurize("otro", [], STATE.items, users, {})[:, FI["hc_buy"]].sum()), 0.0)   # sin historial: ceros
        w = np.zeros(model.K, np.float32)
        w[FI["hc_buy"]] = 5.0
        m = rec.Models({}, {}, WEIGHTS, WEIGHTS, {"v32": w})
        out = rec.respond({"request_id": "r1", "user_id": "u1", "recent_views": []}, "v32-x", STATE, set(), users, m)["items"]
        self.assertEqual(out[0], "p040")
        self.assertTrue(is_valid(out))

    def test_old_session_views_signal(self):
        """v33: en cuántas sesiones antiguas vio el comprador cada producto (tope en 3), solo para quien tiene historial."""
        users = model.UserStore(capacity=8)
        users.add_event("u1", "p030", "view", STATE.items)
        row, P, FI = users.row["u1"], model.PRODUCTS.idx, model.FI
        pairs = sorted([(P["p040"], 1), (P["p041"], 2), (P["p042"], 9)])
        indptr = np.zeros(row + 2, np.int64)
        indptr[row + 1:] = len(pairs)
        users.hs = (indptr, np.array([i for i, _ in pairs], np.int32), np.array([v for _, v in pairs], np.uint32))
        F = model.featurize("u1", [], STATE.items, users, {})
        got = [float(F[P[p], FI["hs_views"]]) for p in ("p040", "p041", "p042", "p043")]
        for x, y in zip(got, [np.log1p(1), np.log1p(2), np.log1p(3), 0.0]):
            self.assertAlmostEqual(x, float(y), places=5)
        self.assertEqual(float(model.featurize("otro", [], STATE.items, users, {})[:, FI["hs_views"]].sum()), 0.0)
        w = np.zeros(model.K, np.float32)
        w[FI["hs_views"]] = 5.0
        m = rec.Models({}, {}, WEIGHTS, WEIGHTS, {"v33": w})
        out = rec.respond({"request_id": "r1", "user_id": "u1", "recent_views": []}, "v33-x", STATE, set(), users, m)["items"]
        self.assertEqual(out[0], "p042")
        self.assertTrue(is_valid(out))

    def test_alarm_only_during_the_final_and_after_30s(self):
        import datetime as dt
        inside = dt.datetime.strptime(rec.ALARM_WINDOW[0], "%Y-%m-%d %H:%M").timestamp() + 600
        before = dt.datetime.strptime(rec.ALARM_WINDOW[0], "%Y-%m-%d %H:%M").timestamp() - 600
        old = rec.LAST_OK[0]
        try:
            rec.LAST_OK[0] = inside - 10
            self.assertEqual(rec.alarm_due(inside), 0.0)               # 10 s sin conexión: aún no
            rec.LAST_OK[0] = inside - 45
            self.assertAlmostEqual(rec.alarm_due(inside), 45.0)        # 45 s sin conexión durante la final: alarma
            rec.LAST_OK[0] = before - 300
            self.assertEqual(rec.alarm_due(before), 0.0)               # fuera de la final: nunca
        finally:
            rec.LAST_OK[0] = old

    def test_split_is_stable_and_balanced(self):
        self.assertEqual(rec.pick_variant({"user_id": "u_1", "request_id": "a"}), rec.pick_variant({"user_id": "u_1", "request_id": "b"}))
        counts = Counter(rec.pick_variant({"user_id": f"u_{i}", "request_id": "x"}) for i in range(4000))
        total = sum(rec.VARIANTS.values())
        for name, share in rec.VARIANTS.items():
            self.assertAlmostEqual(counts[name] / 4000, share / total, delta=0.04)


if __name__ == "__main__":
    unittest.main()
