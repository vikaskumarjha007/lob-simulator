"""make demo   -> synthetic ZI experiment + chart (uses committed ES aggregates if present)
make real    -> run the experiment on Databento ES MBP-1 parquet files and write aggregates

Only aggregate tables (fill rate per queue bucket, mean post-fill mid move) are written for
real data. Per-order rows and raw data never leave the machine.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from lob import queue_experiment as qx  # noqa: E402

OUT = Path("outputs")
ES_COLOR, SYN_COLOR = "#2a78d6", "#eb6834"


LOCAL = Path("data/raw/es_order_results")   # per-order rows: gitignored, never committed


def run_real(paths: list[str]) -> None:
    """Process each day once (cached per day in data/raw/), then aggregate everything cached."""
    from data.loaders import load_real
    LOCAL.mkdir(parents=True, exist_ok=True)
    for p in paths:
        day = Path(p).parent.parent.name.replace("date=", "")
        target = LOCAL / f"{day}.csv"
        if target.exists():
            continue
        r = qx.run(load_real(p), every_s=30, timeout_s=300)
        r["day"] = day
        r.to_csv(target, index=False)
        print(day, len(r), r["outcome"].value_counts().to_dict(), flush=True)
    res = pd.concat([pd.read_csv(f) for f in sorted(LOCAL.glob("*.csv"))], ignore_index=True)
    OUT.mkdir(exist_ok=True)
    qx.fill_curve(res).to_csv(OUT / "es_fill_curve.csv", index=False)
    qx.adverse_selection(res).to_csv(OUT / "es_adverse.csv", index=False)
    meta = {"days": res["day"].nunique(), "first_day": res["day"].min(), "last_day": res["day"].max(),
            "orders": len(res),
            **{f"share_{k}": round(v, 4) for k, v in res["outcome"].value_counts(normalize=True).items()},
            "median_time_to_fill_s": res["time_to_fill"].median()}
    pd.Series(meta).to_csv(OUT / "es_meta.csv", header=False)
    print(pd.Series(meta).to_string())


def run_demo() -> None:
    from data.loaders import load_synthetic
    syn = qx.run(load_synthetic(), every_s=10, timeout_s=120)
    syn_fill, syn_adv = qx.fill_curve(syn), qx.adverse_selection(syn)
    OUT.mkdir(exist_ok=True)
    syn_fill.to_csv(OUT / "synthetic_fill_curve.csv", index=False)
    syn_adv.to_csv(OUT / "synthetic_adverse.csv", index=False)

    es_fill = pd.read_csv(OUT / "es_fill_curve.csv") if (OUT / "es_fill_curve.csv").exists() else None
    es_adv = pd.read_csv(OUT / "es_adverse.csv") if (OUT / "es_adverse.csv").exists() else None

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
    series = [("ES futures (real, MBP-1)", es_fill, es_adv, ES_COLOR),
              ("Zero-intelligence sim", syn_fill, syn_adv, SYN_COLOR)]
    for name, fc, adv, color in series:
        if fc is None:
            continue
        a1.plot(fc["queue_median"], fc["fill_rate"] * 100, marker="o", ms=6, lw=2, color=color, label=name)
        a2.errorbar(adv["horizon_s"], adv["mean_edge_ticks"], yerr=1.96 * adv["se"], marker="o", ms=6,
                    lw=2, capsize=3, color=color, label=name)
    a1.set_xscale("log")
    a1.set_xlabel("Queue ahead when joining (contracts, bucket median, log)")
    a1.set_ylabel("Fill rate (%)")
    a1.set_title("Fill probability vs queue position", loc="left", fontsize=11)
    a2.axhline(0, color="#999999", lw=0.8, ls="--")
    a2.set_xscale("log")
    a2.set_xticks([1, 10, 60], ["1s", "10s", "60s"])
    a2.set_xlabel("Time after fill")
    a2.set_ylabel("Mid move in our favour (ticks, ±95% CI)")
    a2.set_title("Adverse selection after a passive fill", loc="left", fontsize=11)
    for ax in (a1, a2):
        ax.grid(axis="y", color="#e6e6e6", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    a2.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "queue_fill_and_adverse_selection.png")
    print(syn_fill.to_string(index=False))
    print(syn_adv.to_string(index=False))
    print(f"Wrote {OUT/'queue_fill_and_adverse_selection.png'}" +
          ("" if es_fill is not None else " (synthetic only: no ES aggregates found)"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", nargs="+", help="Databento MBP-1 parquet files (one instrument each)")
    a = ap.parse_args()
    run_real(a.real) if a.real else run_demo()
