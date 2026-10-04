import numpy as np
import pandas as pd
import pytest

from data.loaders import COLUMNS, load_synthetic
from lob import BUY, SELL, OrderBook
from lob import queue_experiment as qx


# ---------- matching engine -----------------------------------------------------------------
def test_price_time_priority_same_price():
    b = OrderBook()
    first = b.limit(SELL, 101, 5, ts=1)
    second = b.limit(SELL, 101, 5, ts=2)
    trades = b.market(BUY, 7, ts=3)
    assert [(t.maker_id, t.qty) for t in trades] == [(first, 5), (second, 2)]
    assert b.depth_at(SELL, 101) == 3 and b.queue_ahead(second) == 0


def test_better_price_beats_earlier_time():
    b = OrderBook()
    early_worse = b.limit(BUY, 99, 5, ts=1)
    late_better = b.limit(BUY, 100, 5, ts=2)
    t = b.market(SELL, 1, ts=3)[0]
    assert t.maker_id == late_better and t.price == 100 and b.queue_ahead(early_worse) == 0


def test_market_sweep_vwap_and_remaining_book():
    b = OrderBook()
    b.limit(SELL, 100, 2)
    b.limit(SELL, 101, 3)
    b.limit(SELL, 103, 10)
    trades = b.market(BUY, 8)
    qty = sum(t.qty for t in trades)
    vwap = sum(t.price * t.qty for t in trades) / qty
    assert qty == 8
    assert vwap == pytest.approx((100 * 2 + 101 * 3 + 103 * 3) / 8)
    assert b.l2(5)["asks"] == [(103, 7)]


def test_marketable_limit_partially_fills_then_rests():
    b = OrderBook()
    b.limit(SELL, 100, 3)
    oid = b.limit(BUY, 101, 5)          # crosses: takes 3 @100, rests 2 @101
    assert sum(t.qty for t in b.trades) == 3 and b.trades[0].price == 100
    assert b.best_bid() == 101 and b.depth_at(BUY, 101) == 2 and b.best_ask() is None
    assert b.queue_ahead(oid) == 0


def test_cancel_removes_the_right_order():
    b = OrderBook()
    a, c, d = b.limit(BUY, 100, 1), b.limit(BUY, 100, 2), b.limit(BUY, 100, 3)
    assert b.cancel(c) and not b.cancel(c)
    assert b.queue_ahead(d) == 1 and b.depth_at(BUY, 100) == 4
    b.cancel(a)
    b.cancel(d)
    assert b.best_bid() is None and b.bids == {}


def test_book_invariants_hold_after_10k_random_events():
    rng = np.random.default_rng(42)
    b = OrderBook()
    for i in range(10_000):
        u = rng.random()
        side = BUY if rng.random() < 0.5 else SELL
        if u < 0.6:
            b.limit(side, int(rng.integers(90, 111)), int(rng.integers(1, 6)), ts=i)
        elif u < 0.8:
            b.market(side, int(rng.integers(1, 6)), ts=i)
        elif b.resting_ids():
            ids = b.resting_ids()
            b.cancel(ids[int(rng.integers(len(ids)))])
        bb, ba = b.best_bid(), b.best_ask()
        assert bb is None or ba is None or bb < ba, f"crossed book at event {i}"
        for book in (b.bids, b.asks):
            for q in book.values():
                assert q and all(o.qty > 0 for o in q)
    resting = sum(o.qty for book in (b.bids, b.asks) for q in book.values() for o in q)
    assert len(b.resting_ids()) == sum(len(q) for book in (b.bids, b.asks) for q in book.values())
    assert resting > 0


# ---------- queue-position fill rule (hand-built tape) ---------------------------------------
def tape(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def q(ts, bid, ask, bsz, asz):
    return (ts, "Q", np.nan, 0, "N", bid, ask, bsz, asz)


def t(ts, px, sz, agg, bid, ask, bsz, asz):
    return (ts, "T", px, sz, agg, bid, ask, bsz, asz)


def test_fill_only_after_queue_ahead_is_traded_through():
    # Bid 100 x 5 at t=1. Order placed at t=10 joins behind 5 lots.
    rows = [q(0, 100, 101, 5, 5), q(1, 100, 101, 5, 5),
            t(11, 100, 2, "S", 100, 101, 3, 5),
            t(12, 100, 2, "S", 100, 101, 1, 5),
            t(13, 101, 4, "B", 100, 101, 1, 1),   # wrong side/price: must not count
            t(14, 100, 2, "S", 100, 101, 1, 1),   # cumulative 6 > 5 -> filled at 14
            q(20, 100, 101, 1, 1), q(400, 100, 101, 1, 1)]
    r = qx.run(tape(rows), every_s=10, timeout_s=60)
    buy = r[(r.side == "B") & (r.t0 == 10)].iloc[0]
    assert buy.outcome == "filled" and buy.time_to_fill == pytest.approx(4)
    assert buy.queue_ahead == 5


def test_exactly_queue_size_is_not_enough():
    rows = [q(0, 100, 101, 5, 5), t(11, 100, 5, "S", 100, 101, 0, 5),
            q(11.5, 100, 101, 3, 5), q(400, 100, 101, 3, 5)]
    r = qx.run(tape(rows), every_s=10, timeout_s=60)
    assert r[(r.side == "B") & (r.t0 == 10)].iloc[0].outcome == "pulled"


def test_pulled_when_touch_moves_away_before_prints():
    rows = [q(0, 100, 101, 1, 1), q(11, 101, 102, 4, 4),         # bid moves up: our 100 bid is pulled
            t(12, 100, 9, "S", 99, 102, 1, 4), q(400, 101, 102, 1, 1)]
    r = qx.run(tape(rows), every_s=10, timeout_s=60)
    assert r[(r.side == "B") & (r.t0 == 10)].iloc[0].outcome == "pulled"


def test_ambiguous_when_level_clears_without_enough_prints():
    rows = [q(0, 100, 101, 5, 5), q(11, 99, 101, 7, 5), q(400, 99, 101, 7, 5)]
    r = qx.run(tape(rows), every_s=10, timeout_s=60)
    assert r[(r.side == "B") & (r.t0 == 10)].iloc[0].outcome == "ambiguous"


def test_sell_side_mirror_and_adverse_selection_sign():
    # Our sell at ask 101 behind 2 lots; buyers lift 3 at t=12; at t=13 mid jumps to 102.5.
    rows = [q(0, 100, 101, 5, 2), t(12, 101, 3, "B", 100, 101, 5, 1),
            q(12.5, 102, 103, 1, 1), q(400, 102, 103, 1, 1)]
    r = qx.run(tape(rows), every_s=10, timeout_s=60)
    s = r[(r.side == "S") & (r.t0 == 10)].iloc[0]
    assert s.outcome == "filled"
    assert s.edge_1s == pytest.approx(101 - 102.5)        # sold, then price rose: adverse (negative)


# ---------- synthetic flow ------------------------------------------------------------------
def test_synthetic_tape_schema_and_seed():
    a = load_synthetic(n_events=3_000, seed=3)
    b = load_synthetic(n_events=3_000, seed=3)
    assert list(a.columns) == COLUMNS
    pd.testing.assert_frame_equal(a, b)
    assert (a["bid_px"] < a["ask_px"]).all() and (a["kind"] == "T").any()
