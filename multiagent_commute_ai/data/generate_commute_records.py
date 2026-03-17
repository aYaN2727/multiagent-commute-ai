"""
Script to generate data/commute_records.csv with 500 rows.
Run this once: python data/generate_commute_records.py
"""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

records = []

# ── 475 Normal records ────────────────────────────────────────────────────────
for i in range(475):
    distance_km = rng.uniform(3, 35)
    delay_minutes = max(0.0, rng.normal(8, 6))
    route_avg_delay_min = rng.uniform(5, 20)
    # Mostly weekdays (Mon–Fri), occasional weekend
    day_of_week = int(rng.choice([0, 1, 2, 3, 4, 5, 6], p=[
        0.20, 0.20, 0.20, 0.20, 0.15, 0.03, 0.02
    ]))
    # Morning or evening commute
    if rng.random() < 0.55:
        hour_of_day = int(rng.integers(7, 11))
    else:
        hour_of_day = int(rng.integers(17, 22))
    claim_frequency_30d = float(rng.choice([0, 0, 0, 1, 1, 2, 2, 3, 4, 5], p=[
        0.25, 0.20, 0.15, 0.15, 0.10, 0.07, 0.04, 0.02, 0.01, 0.01
    ]))
    week_num = int(rng.integers(1, 53))
    is_holiday = int(rng.random() < 0.03)
    records.append({
        "employee_id": f"EMP_{rng.integers(1000, 9999)}",
        "route_id": f"ROUTE_{rng.integers(1, 30):02d}",
        "distance_km": round(distance_km, 2),
        "delay_minutes": round(delay_minutes, 1),
        "route_avg_delay_min": round(route_avg_delay_min, 1),
        "day_of_week": day_of_week,
        "hour_of_day": hour_of_day,
        "claim_frequency_30d": claim_frequency_30d,
        "week_num": week_num,
        "is_holiday": is_holiday,
    })

# ── 25 Anomalous records ──────────────────────────────────────────────────────
anomaly_types = ["high_delay", "high_frequency", "mismatch"]
for i in range(25):
    atype = anomaly_types[i % 3]
    distance_km = rng.uniform(3, 35)
    route_avg_delay_min = rng.uniform(5, 20)
    day_of_week = int(rng.integers(0, 5))
    hour_of_day = int(rng.integers(7, 11))
    week_num = int(rng.integers(1, 53))
    is_holiday = 0

    if atype == "high_delay":
        delay_minutes = float(rng.integers(60, 181))
        claim_frequency_30d = float(rng.integers(1, 6))
    elif atype == "high_frequency":
        delay_minutes = max(0.0, rng.normal(8, 6))
        claim_frequency_30d = float(rng.integers(15, 26))
    else:  # mismatch: high reported delay but low route avg
        delay_minutes = float(rng.integers(90, 181))
        route_avg_delay_min = rng.uniform(3, 5)
        claim_frequency_30d = float(rng.integers(8, 15))

    records.append({
        "employee_id": f"EMP_{rng.integers(1000, 9999)}",
        "route_id": f"ROUTE_{rng.integers(1, 30):02d}",
        "distance_km": round(distance_km, 2),
        "delay_minutes": round(delay_minutes, 1),
        "route_avg_delay_min": round(route_avg_delay_min, 1),
        "day_of_week": day_of_week,
        "hour_of_day": hour_of_day,
        "claim_frequency_30d": claim_frequency_30d,
        "week_num": week_num,
        "is_holiday": is_holiday,
    })

df = pd.DataFrame(records)
os.makedirs("data", exist_ok=True)
df.to_csv("data/commute_records.csv", index=False)
print(f"Generated {len(df)} rows -> data/commute_records.csv")
print(df.describe())
