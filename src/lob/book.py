"""Price-time priority limit order book (single instrument, integer tick prices).

Supported: limit orders (marketable ones execute immediately, remainder rests),
market orders (sweep levels until filled or the book is empty), cancels, partial fills.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import count

BUY, SELL = "B", "S"


@dataclass
class Order:
    id: int
    side: str
    price: int
    qty: int
    ts: float


@dataclass(frozen=True)
class Trade:
    ts: float
    price: int
    qty: int
    aggressor: str      # side of the incoming order: "B" lifted the ask, "S" hit the bid
    maker_id: int
    taker_id: int


@dataclass
class OrderBook:
    bids: dict[int, deque] = field(default_factory=dict)
    asks: dict[int, deque] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    _where: dict[int, tuple[str, int]] = field(default_factory=dict)
    _ids: count = field(default_factory=lambda: count(1))

    # ---------- queries ----------
    def best_bid(self) -> int | None:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> int | None:
        return min(self.asks) if self.asks else None

    def depth_at(self, side: str, price: int) -> int:
        q = (self.bids if side == BUY else self.asks).get(price)
        return sum(o.qty for o in q) if q else 0

    def l2(self, n: int = 5) -> dict:
        """Top-n levels: {'bids': [(price, size), ...] best first, 'asks': [...]}"""
        bids = sorted(self.bids, reverse=True)[:n]
        asks = sorted(self.asks)[:n]
        return {"bids": [(p, self.depth_at(BUY, p)) for p in bids],
                "asks": [(p, self.depth_at(SELL, p)) for p in asks]}

    def queue_ahead(self, order_id: int) -> int:
        """Quantity resting in front of this order at its price level."""
        side, price = self._where[order_id]
        ahead = 0
        for o in (self.bids if side == BUY else self.asks)[price]:
            if o.id == order_id:
                return ahead
            ahead += o.qty
        raise KeyError(order_id)

    # ---------- actions ----------
    def limit(self, side: str, price: int, qty: int, ts: float = 0.0) -> int:
        """Submit a limit order. Returns its id (it may be fully filled and not rest)."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        oid = next(self._ids)
        remaining = self._match(side, qty, ts, oid, limit_price=price)
        if remaining > 0:
            book = self.bids if side == BUY else self.asks
            book.setdefault(price, deque()).append(Order(oid, side, price, remaining, ts))
            self._where[oid] = (side, price)
        return oid

    def market(self, side: str, qty: int, ts: float = 0.0) -> list[Trade]:
        oid = next(self._ids)
        n_before = len(self.trades)
        self._match(side, qty, ts, oid, limit_price=None)
        return self.trades[n_before:]

    def cancel(self, order_id: int) -> bool:
        loc = self._where.pop(order_id, None)
        if loc is None:
            return False
        side, price = loc
        book = self.bids if side == BUY else self.asks
        q = book[price]
        for i, o in enumerate(q):
            if o.id == order_id:
                del q[i]
                break
        if not q:
            del book[price]
        return True

    def resting_ids(self) -> list[int]:
        return list(self._where)

    # ---------- internals ----------
    def _match(self, side: str, qty: int, ts: float, taker_id: int, limit_price: int | None) -> int:
        opposite = self.asks if side == BUY else self.bids
        while qty > 0 and opposite:
            best = min(opposite) if side == BUY else max(opposite)
            if limit_price is not None and ((side == BUY and best > limit_price) or
                                            (side == SELL and best < limit_price)):
                break
            q = opposite[best]
            while qty > 0 and q:
                maker = q[0]
                take = min(qty, maker.qty)
                self.trades.append(Trade(ts, best, take, side, maker.id, taker_id))
                maker.qty -= take
                qty -= take
                if maker.qty == 0:
                    q.popleft()
                    del self._where[maker.id]
            if not q:
                del opposite[best]
        return qty
