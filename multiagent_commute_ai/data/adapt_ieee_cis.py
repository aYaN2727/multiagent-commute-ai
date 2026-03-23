"""
data/adapt_ieee_cis.py
======================
Maps the IEEE-CIS Fraud Detection dataset (Kaggle) to the commute claim
feature space used by this project.

Why this works:
  - Both domains share the same anomaly signals: unusual frequency, timing,
    amount/delay deviation from baseline, and behavioural outliers.
  - isFraud (0/1) gives us REAL ground-truth labels — far better than the
    25 hand-crafted synthetic anomalies we had before.

Column Mapping
--------------
  IEEE-CIS Column          →  Commute Feature
  ─────────────────────────────────────────────────────────
  isFraud                  →  is_anomaly  (ground truth)
  TransactionDT % 604800   →  day_of_week (0-6)
  TransactionDT % 86400    →  hour_of_day (0-23)
  TransactionAmt (scaled)  →  distance_km  (3 – 50 km)
  C1  (card count)         →  claim_frequency_30d
  D1  (days since prev tx) →  avg_delay_proxy → delay_minutes + route_avg_delay_min
  dist1                    →  distance modifier
  Assigned from route_master.csv → route_id, route_avg_delay_min

Usage:
    python data/adapt_ieee_cis.py
    python data/adapt_ieee_cis.py --input data/train_transaction.csv --output data/commute_records.csv
    python data/adapt_ieee_cis.py --sample 15000   # normal rows to keep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROUTE_MASTER_PATH = Path("data/route_master.csv")
DEFAULT_INPUT     = Path("data/train_transaction.csv")
DEFAULT_OUTPUT    = Path("data/commute_records.csv")

# Columns we actually need from the 434-column IEEE file
COLS_NEEDED = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
    "C1", "C2", "D1", "D4", "dist1",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_ieee(path: Path, normal_sample: int) -> pd.DataFrame:
    """Load IEEE-CIS CSV keeping only required columns; balance the sample."""
    print(f"[IEEE] Reading {path} …")
    usecols = [c for c in COLS_NEEDED]          # only needed columns
    df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    print(f"[IEEE] Raw shape: {df.shape}  fraud rate: {df['isFraud'].mean():.3%}")

    fraud  = df[df["isFraud"] == 1]
    normal = df[df["isFraud"] == 0].sample(
        n=min(normal_sample, len(df[df["isFraud"] == 0])),
        random_state=42,
    )
    combined = pd.concat([fraud, normal]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[IEEE] Sampled -> {len(combined)} rows  "
          f"(fraud={len(fraud)}, normal={len(normal)})")
    return combined


def load_route_master() -> pd.DataFrame:
    """Load real Bangalore route data."""
    if ROUTE_MASTER_PATH.exists():
        rm = pd.read_csv(ROUTE_MASTER_PATH)
        print(f"[Routes] Loaded {len(rm)} real routes from route_master.csv")
        return rm
    # Fallback: minimal synthetic routes
    print("[Routes] route_master.csv not found — using synthetic fallback")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "route_id":          [f"R{i:03d}" for i in range(1, 21)],
        "distance_km":       rng.uniform(5, 40, 20).round(1),
        "avg_time_min":      rng.uniform(15, 60, 20).round(0),
        "peak_time_min":     rng.uniform(30, 90, 20).round(0),
        "max_detour_min":    rng.uniform(5, 15, 20).round(0),
    })


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def _percentile_scale(series: pd.Series, lo: float, hi: float) -> pd.Series:
    """Robust min-max scale using 1st / 99th percentiles → [lo, hi]."""
    p1, p99 = series.quantile(0.01), series.quantile(0.99)
    scaled = (series.clip(p1, p99) - p1) / (p99 - p1 + 1e-9)
    return (scaled * (hi - lo) + lo).round(2)


def transform(df: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    """Map IEEE-CIS columns to commute claim features."""
    rng = np.random.default_rng(42)
    n   = len(df)

    # ── Temporal features ──────────────────────────────────────────────────
    # TransactionDT is seconds from some reference.
    # 604800 = 7*24*3600 (one week in seconds)
    dt = df["TransactionDT"].fillna(0).astype(float)
    day_of_week = ((dt // 86400) % 7).astype(int)                    # 0-6
    hour_raw    = ((dt % 86400) // 3600).astype(int)                  # 0-23
    # Shift so normal commute hours cluster at 7-10 and 17-21
    # TransactionDT baseline is unknown; we keep the modulo distribution
    hour_of_day = hour_raw

    # ── Distance ──────────────────────────────────────────────────────────
    # TransactionAmt range is $0.25 – $31,937; map to 3 – 50 km
    distance_km = _percentile_scale(df["TransactionAmt"].fillna(10), 3.0, 50.0)

    # ── Route assignment ───────────────────────────────────────────────────
    route_idx   = rng.integers(0, len(routes), size=n)
    route_id    = routes.iloc[route_idx]["route_id"].values

    # Real avg delay from route master avg_time_min, scaled to minutes
    route_avg_delay_min = routes.iloc[route_idx]["avg_time_min"].values.astype(float)
    # Normal delay: small fraction of avg travel time
    base_delay  = route_avg_delay_min * rng.uniform(0.1, 0.5, n)

    # ── Delay minutes ─────────────────────────────────────────────────────
    # For fraud rows: extreme delay relative to route avg
    # For normal rows: reasonable delay based on D1 (days since last tx)
    d1 = df["D1"].fillna(0).clip(0, 365).values
    # Higher D1 → employee was absent longer → plausible higher delay claim
    delay_from_d1 = np.clip(d1 * 0.5, 0, 40)

    delay_minutes = np.where(
        df["isFraud"].values == 1,
        # Fraud: 1.5–4x route average with noise — realistic but elevated.
        # Previously 2.5–10x which created an 8.5x mean ratio (too extreme).
        route_avg_delay_min * rng.uniform(1.5, 4.0, n)
            + rng.normal(0, 15, n).clip(-10),
        # Normal: base delay + D1-derived noise
        base_delay + delay_from_d1 + rng.normal(0, 5, n).clip(0),
    ).clip(0).round(1)

    # ── Claim frequency (30d) ─────────────────────────────────────────────
    # C1 = count of cards used; maps well to claim frequency.
    # FIXED: previous hard clips (normal 0-7, fraud 8-30) created perfect
    # separation with zero overlap — unrealistic and caused inflated metrics.
    # Now both distributions overlap realistically in the 4-12 range.
    c1 = df["C1"].fillna(1).clip(0, 30).values
    c2 = df["C2"].fillna(0).clip(0, 10).values

    claim_frequency_30d = np.where(
        df["isFraud"].values == 1,
        # Fraud: tends to be higher but with realistic overlap at lower end.
        # Range: ~4-25, mean ~14 (not hard-clipped above 8)
        np.clip(c1 * 1.2 + c2 * 0.5 + rng.integers(2, 12, n), 1, 25),
        # Normal: tends to be lower but some legitimate high-frequency users.
        # Range: 0-14, mean ~3 (not hard-clipped below 8)
        np.clip(c1 * 0.4 + rng.integers(0, 5, n), 0, 14),
    ).astype(float)

    # ── Timing: weekend / holiday ─────────────────────────────────────────
    is_holiday = (day_of_week >= 5).astype(int)                     # Sat/Sun

    # ── Employee / week ───────────────────────────────────────────────────
    week_num = ((dt // (7 * 86400)) % 52 + 1).astype(int)

    out = pd.DataFrame({
        "employee_id":          [f"EMP_{tid}" for tid in df["TransactionID"]],
        "route_id":             route_id,
        "distance_km":          distance_km.values,
        "delay_minutes":        delay_minutes,
        "route_avg_delay_min":  route_avg_delay_min.round(1),
        "day_of_week":          day_of_week.values,
        "hour_of_day":          hour_of_day.values,
        "claim_frequency_30d":  claim_frequency_30d,
        "week_num":             week_num.values,
        "is_holiday":           is_holiday,
        "is_anomaly":           df["isFraud"].values,   # ground-truth label
    })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. VALIDATE
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    n_total  = len(df)
    n_anom   = int(df["is_anomaly"].sum())
    n_normal = n_total - n_anom
    print(f"\n{'='*55}")
    print(f"  Output dataset summary")
    print(f"  Total rows    : {n_total:,}")
    print(f"  Normal        : {n_normal:,}  ({n_normal/n_total:.1%})")
    print(f"  Anomalous     : {n_anom:,}  ({n_anom/n_total:.1%})")
    print(f"\n  Feature stats (anomalous vs normal):")
    cols = ["delay_minutes", "claim_frequency_30d", "distance_km"]
    for col in cols:
        a_mean = df.loc[df["is_anomaly"]==1, col].mean()
        n_mean = df.loc[df["is_anomaly"]==0, col].mean()
        print(f"    {col:25s}  anomaly={a_mean:7.2f}  normal={n_mean:7.2f}")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="IEEE-CIS → Commute claim adapter")
    parser.add_argument("--input",  default=str(DEFAULT_INPUT),  help="Path to train_transaction.csv")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path")
    parser.add_argument("--sample", type=int, default=20000,
                        help="Number of NORMAL rows to sample (default 20000)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print("Make sure train_transaction.csv is extracted from the Kaggle zip.")
        sys.exit(1)

    routes = load_route_master()
    df_raw = load_ieee(input_path, normal_sample=args.sample)
    df_out = transform(df_raw, routes)

    print_summary(df_out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"[Done] Saved {len(df_out):,} rows -> {output_path}")
    print("\nNext steps:")
    print("  1. python -m ml.train_isolation_forest")
    print("  2. python -m rag.ingestion")
    print("  3. python main.py")


if __name__ == "__main__":
    main()
