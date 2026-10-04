"""Zero-intelligence order flow, loosely after Cont, Stoikov & Talreja (2010).

Kept deliberately simple (and stated, so it isn't oversold):
  * Events arrive as a Poisson process (exponential gaps, `rate` events/second).
  * Each event is a limit order (p_limit), market order (p_market) or cancel (rest).
  * Limit orders are placed 1..max_offset ticks away from the opposite best quote
    (offset 1 = joining/improving near the touch), side 50/50, size geometric.
  * Cancels pick a random side, a random price level on it, then a random order in that level
    (cheap approximation of per-order cancellation).
Not modelled: order-size/queue-dependent rates, clustering, informed traders.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .book import BUY, SELL, OrderBook


def seed_book(book: OrderBook, mid: int, levels: int, rng: np.random.Generator) -> None:
    for k in range(1, levels + 1):
        for _ in range(3):
            book.limit(BUY, mid - k, int(rng.geometric(0.3)))
            book.limit(SELL, mid + k, int(rng.geometric(0.3)))


def simulate(n_events: int = 200_000, seed: int = 11, start_mid: int = 20_000, rate: float = 50.0,
             p_limit: float = 0.50, p_market: float = 0.12, max_offset: int = 5,
             size_p: float = 0.35) -> pd.DataFrame:
    """Run ZI flow and return an MBP-1-style tape (same schema as the real-data loader):
    ts (seconds), kind ('T' trade / 'Q' quote), price, size, aggressor, bid_px, ask_px, bid_sz, ask_sz.
    Prices are in ticks."""
    rng = np.random.default_rng(seed)
    book = OrderBook()
    seed_book(book, start_mid, max_offset, rng)
    ts = 0.0
    rows = []
    last_top = None

    def snapshot():
        bb, ba = book.best_bid(), book.best_ask()
        return (bb, ba, book.depth_at(BUY, bb) if bb is not None else 0,
                book.depth_at(SELL, ba) if ba is not None else 0)

    for _ in range(n_events):
        ts += rng.exponential(1 / rate)
        u = rng.random()
        side = BUY if rng.random() < 0.5 else SELL
        size = int(rng.geometric(size_p))
        n_tr = len(book.trades)
        if u < p_limit:
            bb, ba = book.best_bid(), book.best_ask()
            off = int(rng.integers(1, max_offset + 1))
            if side == BUY:
                ref = ba if ba is not None else (bb + 1 if bb is not None else start_mid + 1)
                book.limit(BUY, ref - off, size, ts)
            else:
                ref = bb if bb is not None else (ba - 1 if ba is not None else start_mid - 1)
                book.limit(SELL, ref + off, size, ts)
        elif u < p_limit + p_market:
            book.market(side, size, ts)
        else:
            levels = book.bids if side == BUY else book.asks
            if levels:
                keys = list(levels)
                q = levels[keys[int(rng.integers(len(keys)))]]
                book.cancel(q[int(rng.integers(len(q)))].id)
        # Keep both sides alive (ZI can empty a side); refill one level if needed.
        if not book.bids:
            book.limit(BUY, (book.best_ask() or start_mid) - 1, 1, ts)
        if not book.asks:
            book.limit(SELL, (book.best_bid() or start_mid) + 1, 1, ts)

        top = snapshot()
        for t in book.trades[n_tr:]:
            rows.append((t.ts, "T", t.price, t.qty, t.aggressor, *top))
        if top != last_top:
            rows.append((ts, "Q", np.nan, 0, "N", *top))
            last_top = top

    return pd.DataFrame(rows, columns=["ts", "kind", "price", "size", "aggressor",
                                       "bid_px", "ask_px", "bid_sz", "ask_sz"])
