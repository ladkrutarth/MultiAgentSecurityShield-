"""
Generate Spending DNA Dataset — 10,000 realistic rows.
Simulates the 8-axis financial fingerprint per user:
  1. avg_txn_amount       — Average transaction size
  2. location_entropy     — Geographic diversity score
  3. weekend_ratio        — Weekend spending fraction
  4. category_diversity   — Number of unique categories used (normalized)
  5. time_of_day_pref     — Preferred hour bucket (0=morning, 1=afternoon, 2=evening, 3=night)
  6. risk_appetite_score  — Weighted % of high-risk category spend
  7. spending_velocity    — Avg transactions per week
  8. merchant_loyalty_score — Repeat merchant fraction

Each row also includes: trust_score, dna_deviation_score, is_anomalous_session.

Run: python scripts/generate_spending_dna_dataset.py
Output: dataset/csv_data/spending_dna_dataset.csv
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "spending_dna_dataset.csv"

random.seed(7)
np.random.seed(7)

# ── Config ──────────────────────────────────────────────────────────────────
N_USERS = 150
TARGET_ROWS = 10_000

CATEGORIES = [
    "Groceries", "Dining", "Entertainment", "Subscriptions", "Travel",
    "Clothing", "Electronics", "Jewelry", "Gas", "Healthcare",
    "Utilities", "Sports & Fitness", "Coffee Shops", "Home Improvement",
]

RISK_WEIGHTS = {
    "Jewelry":        0.90,
    "Electronics":    0.70,
    "Travel":         0.55,
    "Entertainment":  0.40,
    "Clothing":       0.35,
    "Online Shopping":0.30,
    "Dining":         0.20,
    "Subscriptions":  0.15,
    "Groceries":      0.10,
    "Gas":            0.08,
    "Healthcare":     0.05,
    "Utilities":      0.03,
    "Sports & Fitness":0.12,
    "Coffee Shops":   0.04,
    "Home Improvement":0.10,
}

CITIES = [
    ("Houston", "TX"), ("Austin", "TX"), ("Dallas", "TX"),
    ("Los Angeles", "CA"), ("San Francisco", "CA"),
    ("New York", "NY"), ("Chicago", "IL"),
    ("Seattle", "WA"), ("Phoenix", "AZ"), ("Miami", "FL"),
    ("Denver", "CO"), ("Atlanta", "GA"),
]

MERCHANTS_BY_CATEGORY = {
    "Groceries":        ["Whole Foods", "Kroger", "Walmart", "H-E-B", "ALDI"],
    "Dining":           ["Chipotle", "McDonald's", "Chick-fil-A", "Local Bistro", "Panda Express"],
    "Entertainment":    ["AMC Theaters", "Netflix (in-store)", "Regal Cinema", "Steam Games"],
    "Subscriptions":    ["Netflix", "Spotify", "Amazon Prime", "Hulu", "Disney+"],
    "Travel":           ["Delta Airlines", "Marriott", "Airbnb", "Uber", "Enterprise"],
    "Clothing":         ["Zara", "H&M", "Nike", "Gap", "Nordstrom"],
    "Electronics":      ["Best Buy", "Apple Store", "B&H Photo"],
    "Jewelry":          ["Tiffany & Co.", "Kay Jewelers", "Zales"],
    "Gas":              ["Shell", "Chevron", "BP"],
    "Healthcare":       ["CVS Pharmacy", "Walgreens", "Urgent Care"],
    "Utilities":        ["AT&T", "Comcast", "Duke Energy"],
    "Sports & Fitness": ["Nike", "Dick's Sporting Goods", "Planet Fitness"],
    "Coffee Shops":     ["Starbucks", "Dunkin'", "Peet's Coffee"],
    "Home Improvement": ["Home Depot", "Lowe's", "IKEA"],
}


def build_user_dna(user_id: str) -> dict:
    """Generate a stable per-user DNA fingerprint."""
    return {
        "user_id":              user_id,
        "primary_city":        random.choice(CITIES),
        "avg_txn_amount":      round(np.random.lognormal(4.2, 0.7), 2),   # ~$67 median
        "location_entropy":    round(random.uniform(0.1, 2.8), 4),
        "weekend_ratio":       round(random.uniform(0.1, 0.65), 4),
        "category_diversity":  round(random.uniform(0.2, 1.0), 4),
        "time_of_day_pref":    random.choice([0, 1, 2, 3]),   # 0=morning 1=afternoon 2=evening 3=night
        "risk_appetite_score": round(random.uniform(0.05, 0.45), 4),
        "spending_velocity":   round(random.uniform(2.0, 20.0), 2),         # txn/week
        "merchant_loyalty":    round(random.uniform(0.2, 0.9), 4),
        "fav_categories":      random.sample(CATEGORIES, k=random.randint(3, 7)),
    }


def hour_from_pref(pref: int) -> int:
    buckets = {0: (6, 11), 1: (11, 16), 2: (16, 21), 3: (21, 24)}
    lo, hi = buckets[pref]
    return random.randint(lo, hi - 1)


def generate_session_row(dna: dict, is_anomalous: bool = False) -> dict:
    # Transaction date within 2024
    base = datetime(2024, 1, 1)
    txn_date = base + timedelta(days=random.randint(0, 364), hours=random.randint(0, 23))
    is_weekend = txn_date.weekday() >= 5

    # Location — anomalous sessions drift to new city
    if is_anomalous and random.random() < 0.7:
        city_obj = random.choice([c for c in CITIES if c != dna["primary_city"]])
    else:
        city_obj = dna["primary_city"]

    city, state = city_obj

    # Category — anomalous favors high-risk
    if is_anomalous and random.random() < 0.6:
        category = random.choice(["Jewelry", "Electronics", "Travel"])
    else:
        category = random.choice(dna["fav_categories"])

    merchant = random.choice(MERCHANTS_BY_CATEGORY.get(category, ["Misc"]))

    # Amount — anomalous sessions have outlier amounts
    base_amt = dna["avg_txn_amount"]
    if is_anomalous:
        amount = round(base_amt * random.uniform(3.0, 12.0), 2)
    else:
        amount = round(max(1.0, np.random.normal(base_amt, base_amt * 0.4)), 2)

    # Hour
    hour = hour_from_pref(dna["time_of_day_pref"])
    if is_anomalous and random.random() < 0.5:
        # Off-hours
        hour = random.choice(range(0, 6))

    txn_date = txn_date.replace(hour=hour)

    # ── Compute per-row DNA axis values ─────────────────────────────────
    risk_score = RISK_WEIGHTS.get(category, 0.2)

    # ── Deviation scores ─────────────────────────────────────────────────
    amount_dev  = min(abs(amount - dna["avg_txn_amount"]) / max(dna["avg_txn_amount"], 1), 1.0)
    loc_dev     = 0.9 if city_obj != dna["primary_city"] else 0.05
    time_pref   = dna["time_of_day_pref"]
    current_bucket = 0 if hour < 11 else (1 if hour < 16 else (2 if hour < 21 else 3))
    time_dev    = min(abs(current_bucket - time_pref) / 3, 1.0)
    cat_in_fav  = category in dna["fav_categories"]
    cat_dev     = 0.1 if cat_in_fav else 0.7

    dna_deviation = round((amount_dev * 0.35 + loc_dev * 0.30 + time_dev * 0.15 + cat_dev * 0.20), 4)
    trust_score   = round(max(0.0, 1.0 - dna_deviation + random.uniform(-0.05, 0.05)), 4)

    return {
        "user_id":              dna["user_id"],
        "session_id":           f"SES_{random.randint(100000, 999999)}",
        "transaction_date":     txn_date.strftime("%Y-%m-%d"),
        "transaction_time":     txn_date.strftime("%H:%M:%S"),
        "hour_of_day":          hour,
        "is_weekend":           is_weekend,
        "city":                 city,
        "state":                state,
        "category":             category,
        "merchant":             merchant,
        "amount":               amount,
        # ── 8 DNA Axes ──────────────────────────────────────────────────
        "avg_txn_amount":       round(dna["avg_txn_amount"], 2),
        "location_entropy":     round(dna["location_entropy"], 4),
        "weekend_ratio":        round(dna["weekend_ratio"], 4),
        "category_diversity":   round(dna["category_diversity"], 4),
        "time_of_day_pref":     dna["time_of_day_pref"],
        "risk_appetite_score":  round(dna["risk_appetite_score"], 4),
        "spending_velocity":    round(dna["spending_velocity"], 2),
        "merchant_loyalty_score": round(dna["merchant_loyalty"], 4),
        # ── Trust / Deviation ────────────────────────────────────────────
        "dna_deviation_score":  dna_deviation,
        "trust_score":          trust_score,
        "is_anomalous_session": is_anomalous,
    }


def build_dataset() -> pd.DataFrame:
    dna_profiles = [build_user_dna(f"USER_{i:04d}") for i in range(N_USERS)]
    rows_per_user = TARGET_ROWS // N_USERS
    rows = []

    for dna in dna_profiles:
        for _ in range(rows_per_user):
            is_anomalous = random.random() < 0.08  # 8% anomalous sessions
            rows.append(generate_session_row(dna, is_anomalous=is_anomalous))

    df = pd.DataFrame(rows)
    if len(df) < TARGET_ROWS:
        pad = df.sample(TARGET_ROWS - len(df), replace=True).copy()
        df = pd.concat([df, pad], ignore_index=True)
    return df.head(TARGET_ROWS)


if __name__ == "__main__":
    print("🔄 Generating Spending DNA dataset (10,000 rows)…")
    df = build_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved {len(df):,} rows → {OUTPUT_PATH}")
    print(f"   Columns ({len(df.columns)}): {df.columns.tolist()}")
    print(f"   Unique users: {df['user_id'].nunique()}")
    print(f"   Anomalous sessions: {df['is_anomalous_session'].sum():,} ({df['is_anomalous_session'].mean():.1%})")
    print(f"   Avg trust score: {df['trust_score'].mean():.3f}")
