"""Tape loaders. Both return the same schema (see lob.queue_experiment docstring).

load_real reads Databento GLBX.MDP3 MBP-1 parquet (one instrument). Raw files stay outside
the repo (or in data/raw/, gitignored). Requires pyarrow.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lob.zero_intelligence import simulate

COLUMNS = ["ts", "kind", "price", "size", "aggressor", "bid_px", "ask_px", "bid_sz", "ask_sz"]
UNDEF_PRICE = np.iinfo(np.int64).max     # Databento sentinel for "no price"
F_SNAPSHOT = 32                           # Databento flag: record is part of a snapshot, not a live event
PRICE_SCALE = 1e9


def load_real(path: str | Path, tick_size: float = 0.25, rth_utc: tuple | None = ("13:30", "20:00")) -> pd.DataFrame:
    """Databento MBP-1 parquet -> tape. Default keeps US regular trading hours only
    (13:30-20:00 UTC = 09:30-16:00 New York during daylight saving)."""
    cols = ["ts_event", "action", "side", "price", "size", "flags",
            "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]
    df = pd.read_parquet(path, columns=cols)
    df = df[(df["flags"].astype(int) & F_SNAPSHOT) == 0]
    if rth_utc:
        sod = (df["ts_event"].to_numpy(np.int64) // 1_000_000_000) % 86_400   # UTC second of day
        lo, hi = (int(h) * 3600 + int(m) * 60 for h, m in (x.split(":") for x in rth_utc))
        df = df[(sod >= lo) & (sod < hi)]

    def ticks(s: pd.Series) -> np.ndarray:
        v = s.to_numpy(np.int64).astype(float)
        v[s.to_numpy(np.int64) == UNDEF_PRICE] = np.nan
        return np.round(v / PRICE_SCALE / tick_size)

    def as_str(s: pd.Series) -> np.ndarray:
        v = s.to_numpy()
        return v.astype("S1").astype(str) if v.dtype == object and len(v) and isinstance(v[0], bytes) else v.astype(str)

    action, side = as_str(df["action"]), as_str(df["side"])
    out = pd.DataFrame({
        "ts": df["ts_event"].to_numpy(np.int64) / 1e9,
        "kind": np.where(action == "T", "T", "Q"),
        "price": ticks(df["price"]),
        "size": df["size"].to_numpy(float),
        # Databento trade side = aggressor side: 'A' = sell aggressor (hit bid), 'B' = buy aggressor.
        "aggressor": np.select([side == "A", side == "B"], ["S", "B"], "N"),
        "bid_px": ticks(df["bid_px_00"]), "ask_px": ticks(df["ask_px_00"]),
        "bid_sz": df["bid_sz_00"].to_numpy(float), "ask_sz": df["ask_sz_00"].to_numpy(float),
    })
    return out[COLUMNS].reset_index(drop=True)


def load_synthetic(n_events: int = 200_000, seed: int = 11, **kwargs) -> pd.DataFrame:
    return simulate(n_events=n_events, seed=seed, **kwargs)[COLUMNS]
