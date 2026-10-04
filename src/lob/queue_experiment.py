"""Passive-order fill experiment on a top-of-book tape (real ES MBP-1 or synthetic ZI).

Every `every_s` seconds we place a hypothetical 1-lot passive order on each side:
  * buy at the best bid / sell at the best ask, joining the BACK of the queue,
  * queue ahead = displayed size at that price (all of it is ahead of us).

Fill rule (conservative): we are filled only when aggressive volume printed AT our price,
on the side that hits us, exceeds the queue ahead. Cancels ahead of us are ignored
(MBP-1 cannot tell us whether a cancel was ahead of or behind us).

The order is pulled (not filled) when:
  * the touch moves away from us (bid > our buy price / ask < our sell price), or
  * `timeout_s` passes.
If the touch moves *through* our price before enough prints, the outcome is "ambiguous"
(cancels must have cleared the level; we might have been filled). Counted as not filled
and reported separately.

Adverse selection: for each fill, mid-price `h` seconds later minus our fill price, in ticks,
signed so positive = good for us.

Tape schema (both loaders): ts [s], kind 'T'/'Q', price, size, aggressor 'B'/'S'/'N',
bid_px, ask_px, bid_sz, ask_sz   (prices in ticks).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 10, 60)


def run(tape: pd.DataFrame, every_s: float = 30.0, timeout_s: float = 300.0,
        horizons: tuple = HORIZONS) -> pd.DataFrame:
    tape = tape.sort_values("ts", kind="stable").reset_index(drop=True)
    ts = tape["ts"].to_numpy(float)
    bid = tape["bid_px"].to_numpy(float)
    ask = tape["ask_px"].to_numpy(float)
    bsz = tape["bid_sz"].to_numpy(float)
    asz = tape["ask_sz"].to_numpy(float)
    mid = (bid + ask) / 2

    trades = tape[tape["kind"] == "T"]
    tr = {agg: (g["ts"].to_numpy(float), g["price"].to_numpy(float), g["size"].to_numpy(float))
          for agg, g in trades.groupby("aggressor")}
    empty = (np.array([]), np.array([]), np.array([]))

    def state_idx(t: float) -> int:
        return int(np.searchsorted(ts, t, side="right")) - 1

    out = []
    for t0 in np.arange(ts[0] + every_s, ts[-1] - timeout_s, every_s):
        i0 = state_idx(t0)
        if i0 < 0 or not np.isfinite(bid[i0]) or not np.isfinite(ask[i0]) or ask[i0] <= bid[i0]:
            continue
        for side in ("B", "S"):
            price = bid[i0] if side == "B" else ask[i0]
            queue = bsz[i0] if side == "B" else asz[i0]
            hitter = "S" if side == "B" else "B"
            t_end = t0 + timeout_s

            # When does the touch move away / through?
            j0, j1 = i0 + 1, int(np.searchsorted(ts, t_end, side="right"))
            touch = bid[j0:j1] if side == "B" else ask[j0:j1]
            away = touch > price if side == "B" else touch < price
            through = touch < price if side == "B" else touch > price
            t_away = ts[j0 + np.argmax(away)] if away.any() else np.inf
            t_through = ts[j0 + np.argmax(through)] if through.any() else np.inf

            # Prints at our price by the hitting side.
            tt, tp, tsz = tr.get(hitter, empty)
            k0, k1 = np.searchsorted(tt, t0, side="right"), np.searchsorted(tt, t_end, side="right")
            at = tp[k0:k1] == price
            cum = np.cumsum(tsz[k0:k1][at])
            hit = np.nonzero(cum > queue)[0]
            t_fill = tt[k0:k1][at][hit[0]] if hit.size else np.inf

            if t_fill <= min(t_away, t_through, t_end):
                outcome = "filled"
            elif t_through < min(t_away, t_end):
                outcome = "ambiguous"
            else:
                outcome = "pulled"

            row = {"t0": t0, "side": side, "price": price, "queue_ahead": queue,
                   "outcome": outcome, "time_to_fill": t_fill - t0 if outcome == "filled" else np.nan}
            for h in horizons:
                if outcome == "filled" and t_fill + h <= ts[-1]:
                    m = mid[state_idx(t_fill + h)]
                    row[f"edge_{h}s"] = (m - price) if side == "B" else (price - m)
                else:
                    row[f"edge_{h}s"] = np.nan
            out.append(row)
    return pd.DataFrame(out)


def fill_curve(results: pd.DataFrame, n_buckets: int = 5) -> pd.DataFrame:
    """Fill rate by queue-ahead quantile bucket."""
    r = results.copy()
    r["bucket"] = pd.qcut(r["queue_ahead"].rank(method="first"), n_buckets, labels=False)
    g = r.groupby("bucket")
    return pd.DataFrame({
        "queue_lo": g["queue_ahead"].min(), "queue_hi": g["queue_ahead"].max(),
        "queue_median": g["queue_ahead"].median(),
        "orders": g.size(),
        "fill_rate": g["outcome"].apply(lambda s: (s == "filled").mean()),
        "ambiguous_rate": g["outcome"].apply(lambda s: (s == "ambiguous").mean()),
    }).reset_index()


def adverse_selection(results: pd.DataFrame, horizons: tuple = HORIZONS) -> pd.DataFrame:
    f = results[results["outcome"] == "filled"]
    rows = []
    for h in horizons:
        x = f[f"edge_{h}s"].dropna()
        se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
        rows.append({"horizon_s": h, "fills": len(x), "mean_edge_ticks": x.mean(), "se": se})
    return pd.DataFrame(rows)
