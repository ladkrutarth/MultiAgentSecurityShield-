"""
Generate AI Financial Advisor Dataset — 10,000 realistic rows.
Simulates monthly spending per user across categories with subscription tracking,
credit score impact, and month-over-month comparisons.

Run: python scripts/generate_financial_advisor_dataset.py
Output: dataset/csv_data/financial_advisor_dataset.csv
"""

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "financial_advisor_dataset.csv"

# ── Seed for reproducibility ───────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── User archetypes ────────────────────────────────────────────────────────
USER_ARCHETYPES = {
    "frugal_saver":      {"base": 800,  "std": 120, "sub_prob": 0.3,  "risky_prob": 0.05},
    "average_spender":   {"base": 2200, "std": 400, "sub_prob": 0.55, "risky_prob": 0.15},
    "lifestyle_spender": {"base": 4500, "std": 900, "sub_prob": 0.75, "risky_prob": 0.25},
    "high_earner":       {"base": 8000, "std": 2000,"sub_prob": 0.85, "risky_prob": 0.35},
}

CATEGORIES = [
    "Groceries", "Dining", "Entertainment", "Subscriptions", "Travel",
    "Clothing", "Electronics", "Jewelry", "Gas", "Healthcare",
    "Utilities", "Sports & Fitness", "Coffee Shops", "Home Improvement", "Online Shopping",
]

CATEGORY_CREDIT_IMPACT = {
    "Groceries":          "positive",
    "Dining":             "neutral",
    "Entertainment":      "neutral",
    "Subscriptions":      "negative",
    "Travel":             "positive",
    "Clothing":           "neutral",
    "Electronics":        "neutral",
    "Jewelry":            "negative",
    "Gas":                "positive",
    "Healthcare":         "positive",
    "Utilities":          "positive",
    "Sports & Fitness":   "positive",
    "Coffee Shops":       "neutral",
    "Home Improvement":   "positive",
    "Online Shopping":    "neutral",
}

SUBSCRIPTION_MERCHANTS = [
    "Netflix", "Spotify", "Amazon Prime", "Hulu", "Disney+",
    "HBO Max", "Apple One", "YouTube Premium", "Adobe CC", "Microsoft 365",
    "Planet Fitness", "HelloFresh", "Blue Apron", "LinkedIn Premium", "Audible",
]

REGULAR_MERCHANTS = {
    "Groceries":        ["Whole Foods", "Kroger", "Walmart", "Trader Joe's", "ALDI", "H-E-B"],
    "Dining":           ["McDonald's", "Chipotle", "Olive Garden", "Chick-fil-A", "Local Diner"],
    "Entertainment":    ["AMC Theaters", "Regal Cinema", "Dave & Buster's", "Sky Zone"],
    "Travel":           ["Delta Airlines", "United Airlines", "Marriott", "Airbnb", "Uber"],
    "Clothing":         ["H&M", "Zara", "Nike", "Forever 21", "Nordstrom", "Gap"],
    "Electronics":      ["Best Buy", "Apple Store", "B&H Photo", "Micro Center"],
    "Jewelry":          ["Zales", "Kay Jewelers", "Tiffany & Co.", "Pandora"],
    "Gas":              ["Shell", "Chevron", "ExxonMobil", "BP", "Circle K"],
    "Healthcare":       ["CVS Pharmacy", "Walgreens", "LabCorp", "Urgent Care"],
    "Utilities":        ["AT&T", "Comcast", "Duke Energy", "Water Dept"],
    "Sports & Fitness": ["Nike", "Dick's Sporting Goods", "REI", "Academy Sports"],
    "Coffee Shops":     ["Starbucks", "Dunkin'", "Peet's Coffee", "Local Café"],
    "Home Improvement": ["Home Depot", "Lowe's", "IKEA", "Menards"],
    "Online Shopping":  ["Amazon", "eBay", "Etsy", "Wayfair", "Target"],
}

STATES = ["TX", "CA", "FL", "NY", "IL", "WA", "AZ", "GA", "OH", "NC", "PA", "CO"]

def generate_user_profile(user_id: str) -> dict:
    archetype_name = random.choice(list(USER_ARCHETYPES.keys()))
    arch = USER_ARCHETYPES[archetype_name]
    state = random.choice(STATES)
    return {
        "user_id": user_id,
        "archetype": archetype_name,
        "base_spend": arch["base"],
        "std_spend": arch["std"],
        "sub_prob": arch["sub_prob"],
        "state": state,
    }


def generate_rows(n_users: int = 200, target_rows: int = 10_000) -> pd.DataFrame:
    rows = []
    users = [generate_user_profile(f"USER_{i:04d}") for i in range(n_users)]

    # Distribute rows proportionally across users
    rows_per_user = target_rows // n_users

    base_date = datetime(2024, 1, 1)

    for user in users:
        # Monthly totals tracker for prev/curr comparison
        monthly_totals: dict[str, float] = {}

        for _ in range(rows_per_user):
            # Pick a random date in the past 12 months
            days_offset = random.randint(0, 364)
            txn_date = base_date + timedelta(days=days_offset)
            month_key = txn_date.strftime("%Y-%m")

            # Pick category
            is_subscription = random.random() < user["sub_prob"] * 0.25
            if is_subscription:
                category = "Subscriptions"
                merchant = random.choice(SUBSCRIPTION_MERCHANTS)
                amount = round(random.choice([9.99, 12.99, 14.99, 17.99, 19.99, 29.99, 34.99, 49.99]), 2)
            else:
                category = random.choice([c for c in CATEGORIES if c != "Subscriptions"])
                merchant = random.choice(REGULAR_MERCHANTS.get(category, ["Misc Store"]))
                base = user["base_spend"] / 30  # daily budget
                amount = round(max(1.0, np.random.normal(base, user["std_spend"] / 15)), 2)

            # Track monthly totals
            monthly_totals[month_key] = monthly_totals.get(month_key, 0) + amount

            # Previous month total
            prev_dt = (txn_date.replace(day=1) - timedelta(days=1))
            prev_month_key = prev_dt.strftime("%Y-%m")
            prev_total = monthly_totals.get(prev_month_key, user["base_spend"] * random.uniform(0.8, 1.2))
            curr_total = monthly_totals[month_key]

            mom_change = ((curr_total - prev_total) / max(prev_total, 1)) * 100

            # Subscription-specific fields
            sub_freq = "monthly" if is_subscription else "one-time"
            avg_monthly = round(user["base_spend"] * random.uniform(0.85, 1.15), 2)

            # Spending velocity: transactions in last 7 days (simulated)
            velocity_7d = random.randint(2, 18)

            rows.append({
                "user_id":                    user["user_id"],
                "archetype":                  user["archetype"],
                "state":                      user["state"],
                "transaction_date":           txn_date.strftime("%Y-%m-%d"),
                "month":                      txn_date.month,
                "year":                       txn_date.year,
                "month_key":                  month_key,
                "category":                   category,
                "merchant":                   merchant,
                "amount":                     amount,
                "is_subscription":            is_subscription,
                "subscription_frequency":     sub_freq,
                "monthly_total":              round(curr_total, 2),
                "prev_month_total":           round(prev_total, 2),
                "month_over_month_change_pct": round(mom_change, 2),
                "avg_monthly_spend":          avg_monthly,
                "credit_score_impact_category": CATEGORY_CREDIT_IMPACT.get(category, "neutral"),
                "spending_velocity_7d":       velocity_7d,
            })

    df = pd.DataFrame(rows)
    # Pad to exact target
    if len(df) < target_rows:
        extra = df.sample(target_rows - len(df), replace=True).copy()
        df = pd.concat([df, extra], ignore_index=True)
    return df.head(target_rows)


if __name__ == "__main__":
    print("🔄 Generating AI Financial Advisor dataset (10,000 rows)…")
    df = generate_rows(n_users=200, target_rows=10_000)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved {len(df):,} rows → {OUTPUT_PATH}")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Date range: {df['transaction_date'].min()} → {df['transaction_date'].max()}")
    print(f"   Unique users: {df['user_id'].nunique()}")
    print(f"   Subscriptions: {df['is_subscription'].sum():,} ({df['is_subscription'].mean():.1%})")
