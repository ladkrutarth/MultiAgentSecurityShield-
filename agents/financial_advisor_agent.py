"""
Financial Advisor Agent — Tool-based conversational financial analysis.
Supports: fraud detection, category advice, savings plan, spending chart, suspicious activity.
"""

from pathlib import Path
from typing import Any
import re
import pandas as pd
import numpy as np
import random

PROJECT_ROOT       = Path(__file__).resolve().parent.parent
ADVISOR_DATA_PATH  = PROJECT_ROOT / "dataset" / "csv_data" / "financial_advisor_dataset.csv"

# Category advice knowledge-base
CATEGORY_ADVICE: dict[str, dict] = {
    "coffee": {
        "label": "☕ Coffee Shops",
        "avg_weekly": 35,
        "tips": [
            "Brew at home: a $5 bag of beans brews ~30 cups vs 5–6 café visits.",
            "Switch to a 3x/week café limit — saves ~$300/year.",
            "Use a loyalty card (e.g. Starbucks Stars) to offset costs.",
            "Try office coffee or a French press for work-day caffeine.",
        ],
        "annual_waste_estimate": "$900–$1,800 for daily buyers.",
        "risk": "low",
    },
    "dining": {
        "label": "🍽️ Dining & Restaurants",
        "avg_weekly": 120,
        "tips": [
            "Meal-prep Sunday: cover 4–5 weekday lunches and save $200+/month.",
            "Limit restaurant visits to 2×/week — budget $40 per outing.",
            "Use apps like Too Good To Go for discounted restaurant leftovers.",
            "Avoid weekday dinner delivery fees — cook or batch-cook instead.",
        ],
        "annual_waste_estimate": "$2,400–$4,800 for frequent diners.",
        "risk": "medium",
    },
    "club": {
        "label": "🎉 Clubs & Entertainment",
        "avg_weekly": 80,
        "tips": [
            "Set a hard monthly entertainment budget ($150 cap recommended).",
            "Drink at home before going out — pre-gaming saves 40–60% on bar spend.",
            "Look for free local events, open-mic nights, or free museum days.",
            "Audit recurring club memberships — cancel the ones you visit < 2×/month.",
        ],
        "annual_waste_estimate": "$3,000–$6,000+ for frequent club-goers.",
        "risk": "high",
    },
    "gambling": {
        "label": "🎲 Gambling & Casinos",
        "avg_weekly": 150,
        "tips": [
            "Set a hard session loss limit — walk away when it's hit.",
            "Treat gambling budget as entertainment expense, never 'investment'.",
            "Avoid chasing losses — it statistically increases total loss.",
            "If habitual, consider a self-exclusion program or cooling-off period.",
            "Redirect even 50% of gambling spend to an emergency fund.",
        ],
        "annual_waste_estimate": "$5,000–$20,000+ for regular gamblers.",
        "risk": "critical",
        "warning": "⚠️ Gambling-related spending has a **negative credit score impact** and is flagged by fraud systems.",
    },
    "shopping": {
        "label": "🛍️ Shopping & Retail",
        "avg_weekly": 90,
        "tips": [
            "Practice the 24-hour rule: wait a day before non-essential purchases.",
            "Unsubscribe from retailer emails to reduce impulse buys.",
            "Buy off-season clothing (up to 70% off in clearance).",
            "Use cashback cards (1–5% back on purchases you'd make anyway).",
        ],
        "annual_waste_estimate": "$2,000–$5,000 for impulse buyers.",
        "risk": "medium",
    },
    "gas": {
        "label": "⛽ Gas & Transportation",
        "avg_weekly": 60,
        "tips": [
            "Use GasBuddy or Waze to find cheapest nearby fuel.",
            "Inflate tires properly — saves up to 3% on fuel economy.",
            "Carpool or combine errands into single trips.",
            "Consider an EV or hybrid for long commutes.",
        ],
        "annual_waste_estimate": "$500–$1,500 vs optimized alternatives.",
        "risk": "low",
    },
    "subscriptions": {
        "label": "📦 Subscriptions",
        "avg_weekly": 30,
        "tips": [
            "Audit ALL subscriptions — the average American pays for 12+ they forgot about.",
            "Share streaming plans (Netflix, Spotify, etc.) with family.",
            "Rotate services: subscribe for 1 month, binge, cancel, repeat.",
            "Use card-linked offers to get cash back on subscriptions you keep.",
        ],
        "annual_waste_estimate": "$800–$2,400 on forgotten subscriptions.",
        "risk": "medium",
    },
}

# Fraud signal categories that are inherently suspicious
HIGH_RISK_CATEGORIES = {"gambling", "alcohol", "adult_entertainment", "wire_transfer", "crypto"}
SUSPICIOUS_MERCHANTS  = {"cash advance", "western union", "wire", "pawn", "payday loan"}


class FinancialAdvisorAgent:
    """Conversational agent that answers spending questions via tool calls."""

    def __init__(self):
        self._df: pd.DataFrame | None = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(ADVISOR_DATA_PATH)
            self._df["transaction_date"] = pd.to_datetime(self._df["transaction_date"])
        return self._df

    # ── TOOLS ──────────────────────────────────────────────────────────────

    def tool_monthly_comparison(self, user_id: str) -> dict[str, Any]:
        """Compare current month vs. previous month spending."""
        user_df = self.df[self.df["user_id"] == user_id].copy()
        if user_df.empty:
            return {"error": f"No data for {user_id}"}

        months = sorted(user_df["month_key"].unique())
        if len(months) < 2:
            return {"error": "Insufficient history for comparison"}

        curr_key, prev_key = months[-1], months[-2]
        curr = user_df[user_df["month_key"] == curr_key]["amount"].sum()
        prev = user_df[user_df["month_key"] == prev_key]["amount"].sum()
        change_pct = ((curr - prev) / max(prev, 1)) * 100

        curr_breakdown = (
            user_df[user_df["month_key"] == curr_key]
            .groupby("category")["amount"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .to_dict()
        )

        return {
            "tool": "monthly_comparison",
            "current_month": curr_key,
            "previous_month": prev_key,
            "current_spend": round(curr, 2),
            "previous_spend": round(prev, 2),
            "change_pct": round(change_pct, 1),
            "trend": "📈 UP" if change_pct > 5 else ("📉 DOWN" if change_pct < -5 else "➡️ STABLE"),
            "top_categories_this_month": {k: round(v, 2) for k, v in curr_breakdown.items()},
            "show_chart": True,
        }

    def tool_find_cancellable_subscriptions(self, user_id: str, target_savings: float = 100.0) -> dict[str, Any]:
        """Find subscriptions to cancel to reach a savings target."""
        subs = self.df[
            (self.df["user_id"] == user_id) & (self.df["is_subscription"] == True)
        ].copy()
        if subs.empty:
            return {"tool": "find_cancellable_subscriptions", "message": "No subscriptions found.", "items": []}

        by_merchant = (
            subs.groupby("merchant")["amount"]
            .agg(["sum", "count", "mean"])
            .rename(columns={"sum": "total", "count": "transactions", "mean": "avg"})
            .reset_index()
            .sort_values("total", ascending=False)
        )

        selected, running_total = [], 0.0
        for _, row in by_merchant.iterrows():
            if running_total >= target_savings:
                break
            selected.append({
                "merchant": row["merchant"],
                "monthly_cost": round(row["avg"], 2),
                "annual_cost": round(row["avg"] * 12, 2),
            })
            running_total += row["avg"]

        return {
            "tool": "find_cancellable_subscriptions",
            "target_savings": target_savings,
            "projected_monthly_savings": round(running_total, 2),
            "recommended_cancellations": selected,
        }

    def tool_credit_score_impact(self, user_id: str, target_category: str | None = None) -> dict[str, Any]:
        """Estimate credit score impact of spending patterns."""
        user_df = self.df[self.df["user_id"] == user_id]
        if user_df.empty:
            return {"error": f"No data for {user_id}"}

        impact_counts = user_df["credit_score_impact_category"].value_counts().to_dict()
        total = len(user_df)
        pos_ratio = impact_counts.get("positive", 0) / max(total, 1)
        neg_ratio = impact_counts.get("negative", 0) / max(total, 1)

        base = 680
        score_adj = round((pos_ratio * 60) - (neg_ratio * 80))
        estimated_score = base + score_adj

        result: dict[str, Any] = {
            "tool": "credit_score_impact",
            "user_id": user_id,
            "positive_ratio": round(pos_ratio, 3),
            "negative_ratio": round(neg_ratio, 3),
            "estimated_credit_score": min(850, max(580, estimated_score)),
            "impact_breakdown": impact_counts,
        }

        if target_category:
            cat_df = user_df[user_df["category"].str.lower() == target_category.lower()]
            cat_impact = cat_df["credit_score_impact_category"].mode().iloc[0] if not cat_df.empty else "neutral"
            cat_spend = round(cat_df["amount"].sum(), 2)
            result["target_category"] = target_category
            result["target_category_impact"] = cat_impact
            result["target_category_spend"] = cat_spend
            result["recommendation"] = (
                f"Reducing {target_category} spend could improve your score by ~15 pts."
                if cat_impact == "negative"
                else f"{target_category} has a {cat_impact} effect on your credit health."
            )
        return result

    def tool_spending_summary(self, user_id: str) -> dict[str, Any]:
        """Get a full spending summary for the user."""
        user_df = self.df[self.df["user_id"] == user_id]
        if user_df.empty:
            return {"error": f"No data for {user_id}"}

        by_cat = (
            user_df.groupby("category")["amount"]
            .sum()
            .sort_values(ascending=False)
            .head(8)
            .to_dict()
        )
        top_merchant = user_df.groupby("merchant")["amount"].sum().idxmax()
        total = round(user_df["amount"].sum(), 2)
        avg_monthly = round(user_df.groupby("month_key")["amount"].sum().mean(), 2)

        return {
            "tool": "spending_summary",
            "user_id": user_id,
            "total_spend": total,
            "avg_monthly_spend": avg_monthly,
            "top_categories": {k: round(v, 2) for k, v in by_cat.items()},
            "top_merchant": top_merchant,
            "archetype": user_df["archetype"].iloc[0],
            "velocity_7d": int(user_df["spending_velocity_7d"].mean()),
            "show_chart": True,
        }

    # ── NEW TOOL: Category Advice ───────────────────────────────────────────

    def tool_category_advice(self, user_id: str, category_keyword: str) -> dict[str, Any]:
        """Give personalized spending advice for a specific category (coffee, dining, club, gambling…)."""
        user_df = self.df[self.df["user_id"] == user_id]
        key = category_keyword.lower().strip()

        # Map keyword synonyms to our advice keys
        synonym_map = {
            "coffee shop": "coffee", "café": "coffee", "cafe": "coffee", "starbucks": "coffee",
            "food": "dining", "restaurant": "dining", "eat": "dining", "takeout": "dining",
            "bar": "club", "nightclub": "club", "party": "club", "entertainment": "club",
            "casino": "gambling", "bet": "gambling", "lottery": "gambling", "poker": "gambling",
            "shop": "shopping", "retail": "shopping", "amazon": "shopping",
            "petrol": "gas", "fuel": "gas", "uber": "gas", "transport": "gas",
            "subscription": "subscriptions", "netflix": "subscriptions", "spotify": "subscriptions",
        }
        resolved = key
        for syn, mapped in synonym_map.items():
            if syn in key:
                resolved = mapped
                break

        advice = CATEGORY_ADVICE.get(resolved, None)

        # Actual user spend in this category (fuzzy match)
        user_spend_this_cat = 0.0
        if not user_df.empty:
            mask = user_df["category"].str.lower().str.contains(resolved, na=False)
            user_spend_this_cat = round(user_df[mask]["amount"].sum(), 2)

        return {
            "tool": "category_advice",
            "category": resolved,
            "label": advice["label"] if advice else f"📂 {resolved.title()}",
            "user_spend": user_spend_this_cat,
            "tips": advice["tips"] if advice else ["Track this category for 30 days to identify saving opportunities."],
            "annual_waste_estimate": advice.get("annual_waste_estimate", "N/A") if advice else "N/A",
            "risk_level": advice.get("risk", "medium") if advice else "medium",
            "warning": advice.get("warning", "") if advice else "",
        }

    # ── NEW TOOL: Savings Plan ──────────────────────────────────────────────

    def tool_savings_plan(self, user_id: str) -> dict[str, Any]:
        """Generate a personalized month-by-month savings plan based on spending habits."""
        user_df = self.df[self.df["user_id"] == user_id]
        if user_df.empty:
            return {"error": f"No data for {user_id}"}

        monthly_avg = round(user_df.groupby("month_key")["amount"].sum().mean(), 2)

        by_cat = (
            user_df.groupby("category")["amount"]
            .sum()
            .sort_values(ascending=False)
            .to_dict()
        )

        # Identify high-spend, high-reducible categories
        savings_opportunities = []
        total_potential_savings = 0.0
        for cat, total in list(by_cat.items())[:8]:
            monthly = round(total / max(len(user_df["month_key"].unique()), 1), 2)
            reduction = 0.0
            tip = ""
            cat_lower = cat.lower()

            if any(k in cat_lower for k in ["coffee", "café", "cafe"]):
                reduction = round(monthly * 0.5, 2)
                tip = "Brew at home 5 days/week"
            elif any(k in cat_lower for k in ["dining", "food", "restaurant", "takeout"]):
                reduction = round(monthly * 0.35, 2)
                tip = "Meal prep + limit dine-out to twice weekly"
            elif any(k in cat_lower for k in ["shopping", "retail", "clothing"]):
                reduction = round(monthly * 0.30, 2)
                tip = "Apply 24-hour rule before purchases"
            elif any(k in cat_lower for k in ["gambling", "casino", "bet"]):
                reduction = round(monthly * 0.70, 2)
                tip = "Set hard session limits"
            elif any(k in cat_lower for k in ["club", "entertainment", "bar"]):
                reduction = round(monthly * 0.40, 2)
                tip = "Set $150/month entertainment cap"
            elif any(k in cat_lower for k in ["subscription"]):
                reduction = round(monthly * 0.45, 2)
                tip = "Audit and cancel unused subscriptions"
            elif any(k in cat_lower for k in ["gas", "fuel", "transport"]):
                reduction = round(monthly * 0.15, 2)
                tip = "Carpool + use GasBuddy"

            if reduction > 0:
                savings_opportunities.append({
                    "category": cat,
                    "monthly_spend": monthly,
                    "potential_saving": reduction,
                    "tip": tip,
                })
                total_potential_savings += reduction

        return {
            "tool": "savings_plan",
            "user_id": user_id,
            "monthly_avg_spend": monthly_avg,
            "potential_monthly_savings": round(total_potential_savings, 2),
            "potential_annual_savings": round(total_potential_savings * 12, 2),
            "opportunities": savings_opportunities[:6],
            "archetype": user_df["archetype"].iloc[0] if "archetype" in user_df.columns else "unknown",
        }

    # ── NEW TOOL: Real-time Fraud Detection ────────────────────────────────

    def tool_realtime_fraud_check(self, user_id: str, latest_n: int = 10) -> dict[str, Any]:
        """Scan the most recent transactions for suspicious/fraudulent patterns."""
        user_df = self.df[self.df["user_id"] == user_id].copy()
        if user_df.empty:
            return {"error": f"No data for {user_id}"}

        # Sort by date, take last N
        recent = user_df.sort_values("transaction_date", ascending=False).head(latest_n)

        alerts = []
        for _, row in recent.iterrows():
            flags = []
            cat = str(row.get("category", "")).lower()
            merchant = str(row.get("merchant", "")).lower()
            amt = float(row.get("amount", 0))
            risk = float(row.get("risk_score", 0))
            fraud_flag = bool(row.get("is_fraud_flag", False))
            hour = int(pd.to_datetime(row["transaction_date"]).hour) if pd.notna(row.get("transaction_date")) else 12

            if fraud_flag or risk > 0.75:
                flags.append(f"🚨 High fraud score ({risk:.2f}) detected by ML model")
            if any(k in cat for k in HIGH_RISK_CATEGORIES):
                flags.append(f"⚠️ High-risk category: {cat}")
            if any(k in merchant for k in SUSPICIOUS_MERCHANTS):
                flags.append(f"🔴 Suspicious merchant keyword: {merchant}")
            if hour < 2 or hour > 23:
                flags.append(f"🌙 Late-night transaction at {hour:02d}:00")
            if amt > 500 and risk > 0.5:
                flags.append(f"💰 Large transaction (${amt:.2f}) with elevated risk")

            if flags:
                alerts.append({
                    "transaction_date": str(row["transaction_date"])[:10],
                    "merchant": row.get("merchant", "Unknown"),
                    "category": row.get("category", "Unknown"),
                    "amount": round(amt, 2),
                    "risk_score": round(risk, 3),
                    "flags": flags,
                    "severity": "CRITICAL" if (fraud_flag or risk > 0.85) else ("HIGH" if risk > 0.65 else "MEDIUM"),
                })

        avg_risk = round(float(recent["risk_score"].mean()), 3) if "risk_score" in recent.columns else 0.0

        return {
            "tool": "realtime_fraud_check",
            "user_id": user_id,
            "transactions_scanned": len(recent),
            "alerts_found": len(alerts),
            "avg_risk_score": avg_risk,
            "overall_status": (
                "🚨 CRITICAL — Immediate action required!" if any(a["severity"] == "CRITICAL" for a in alerts)
                else ("⚠️ WARNING — Suspicious activity detected" if alerts
                else "✅ CLEAR — No suspicious activity detected")
            ),
            "alerts": alerts,
        }

    # ── NEW TOOL: Suspicious Activity Monitor (real-time watch) ───────────

    def tool_suspicious_activity_monitor(self, user_id: str) -> dict[str, Any]:
        """Watch for account-level suspicious patterns: velocity, location, category shifts."""
        user_df = self.df[self.df["user_id"] == user_id].copy()
        if user_df.empty:
            return {"error": f"No data for {user_id}"}

        alerts = []
        recent = user_df.sort_values("transaction_date", ascending=False).head(30)

        # 1. Velocity spike
        velocity_7d = float(recent["spending_velocity_7d"].mean()) if "spending_velocity_7d" in recent.columns else 0
        velocity_baseline = float(user_df["spending_velocity_7d"].mean()) if "spending_velocity_7d" in user_df.columns else 0
        if velocity_baseline > 0 and velocity_7d > velocity_baseline * 1.8:
            alerts.append({
                "type": "velocity_spike",
                "emoji": "⚡",
                "title": "Spending Velocity Spike",
                "detail": f"7-day spend velocity is {velocity_7d:.0f} — {(velocity_7d/velocity_baseline - 1)*100:.0f}% above your norm.",
                "severity": "HIGH",
            })

        # 2. High-risk category surge
        if "category" in recent.columns:
            risky_txns = recent[recent["category"].str.lower().isin(HIGH_RISK_CATEGORIES)]
            if len(risky_txns) >= 3:
                alerts.append({
                    "type": "risky_category",
                    "emoji": "🎲",
                    "title": f"Risky Category Activity",
                    "detail": f"{len(risky_txns)} transactions in high-risk categories (gambling/cash/wire) in last 30 txns.",
                    "severity": "HIGH",
                })

        # 3. High fraud-flag rate
        if "is_fraud_flag" in recent.columns:
            flag_rate = recent["is_fraud_flag"].mean()
            if flag_rate > 0.15:
                alerts.append({
                    "type": "fraud_flags",
                    "emoji": "🚩",
                    "title": "Multiple Fraud Flags",
                    "detail": f"{flag_rate:.0%} of your recent transactions triggered the fraud model.",
                    "severity": "CRITICAL",
                })

        # 4. Unusual hour activity
        if "transaction_date" in recent.columns:
            recent = recent.copy()
            recent["hour"] = pd.to_datetime(recent["transaction_date"]).dt.hour
            late_night = recent[(recent["hour"] >= 0) & (recent["hour"] <= 4)]
            if len(late_night) >= 3:
                alerts.append({
                    "type": "late_night",
                    "emoji": "🌙",
                    "title": "Late-Night Transaction Cluster",
                    "detail": f"{len(late_night)} transactions between midnight–4am in recent activity.",
                    "severity": "MEDIUM",
                })

        # 5. Large one-off transactions
        amt_mean = float(recent["amount"].mean())
        large_txns = recent[recent["amount"] > amt_mean * 3]
        if len(large_txns) >= 2:
            alerts.append({
                "type": "large_txn",
                "emoji": "💸",
                "title": "Unusually Large Transactions",
                "detail": f"{len(large_txns)} transactions 3× above your average (avg: ${amt_mean:.2f}).",
                "severity": "MEDIUM",
            })

        overall = (
            "🚨 CRITICAL" if any(a["severity"] == "CRITICAL" for a in alerts)
            else ("⚠️ WARNING" if alerts else "✅ ALL CLEAR")
        )

        return {
            "tool": "suspicious_activity_monitor",
            "user_id": user_id,
            "alert_count": len(alerts),
            "overall_status": overall,
            "alerts": alerts,
        }

    # ── CHAT ROUTER ────────────────────────────────────────────────────────

    _CHART_KEYWORDS = [
        "chart", "graph", "show me", "visuali", "bar", "plot",
        "category", "categories", "breakdown", "spending", "how much",
        "summary", "overview", "what am i", "total",
    ]

    def chat(self, message: str, user_id: str) -> dict[str, Any]:
        """Route a natural-language question to the right tool(s)."""
        msg = message.lower()
        results: list[dict] = []
        show_chart = any(k in msg for k in self._CHART_KEYWORDS)

        # Monthly comparison
        if any(k in msg for k in ["more this month", "spending this month", "month", "compare", "increase", "less", "trend", "last month"]):
            results.append(self.tool_monthly_comparison(user_id))

        # Subscriptions / savings target
        if any(k in msg for k in ["cancel", "subscription", "save $", "saving", "cut", "reduce"]):
            match = re.search(r"\$(\d+)", message)
            target = float(match.group(1)) if match else 100.0
            results.append(self.tool_find_cancellable_subscriptions(user_id, target_savings=target))

        # Credit score
        if any(k in msg for k in ["credit score", "credit", "score", "impact", "jewelry", "stop using"]):
            known_cats = ["jewelry", "electronics", "groceries", "dining", "travel", "subscriptions", "gas", "clothing"]
            target_cat = next((c for c in known_cats if c in msg), None)
            results.append(self.tool_credit_score_impact(user_id, target_category=target_cat))

        # Category-specific advice
        category_synonyms = {
            "coffee": ["coffee", "café", "cafe", "starbucks", "latte"],
            "dining": ["dining", "restaurant", "food", "eat out", "takeout", "dinner", "lunch"],
            "club": ["club", "bar", "nightclub", "party", "entertainment", "going out"],
            "gambling": ["gambl", "casino", "bet", "lottery", "poker", "slot"],
            "shopping": ["shop", "retail", "amazon", "mall", "clothing"],
            "gas": ["gas", "petrol", "fuel", "transport", "uber", "lyft"],
            "subscriptions": ["subscription", "netflix", "spotify", "hulu", "streaming"],
        }
        for cat_key, synonyms in category_synonyms.items():
            if any(s in msg for s in synonyms):
                results.append(self.tool_category_advice(user_id, cat_key))
                break

        # Savings plan
        if any(k in msg for k in ["save money", "savings plan", "how to save", "budget", "save more", "cut back", "afford", "frugal"]):
            results.append(self.tool_savings_plan(user_id))

        # Real-time fraud detection
        if any(k in msg for k in ["fraud", "suspicious", "hacked", "stolen", "alert", "detect", "realtime", "real-time", "watch", "monitor", "flag", "scam"]):
            results.append(self.tool_realtime_fraud_check(user_id))
            results.append(self.tool_suspicious_activity_monitor(user_id))

        # Fallback: spending summary
        if not results:
            results.append(self.tool_spending_summary(user_id))
            show_chart = True

        reply = self._compose_reply(message, results)
        return {
            "reply": reply,
            "tool_results": results,
            "user_id": user_id,
            "show_chart": show_chart,
        }

    def _compose_reply(self, question: str, tool_results: list[dict]) -> str:
        parts = []
        for r in tool_results:
            tool = r.get("tool", "")
            if "error" in r:
                parts.append(f"⚠️ {r['error']}")
                continue

            if tool == "monthly_comparison":
                trend = r.get("trend", "")
                chg = r.get("change_pct", 0)
                parts.append(
                    f"**Monthly Comparison** {trend}\n"
                    f"You spent **${r['current_spend']:,.2f}** in {r['current_month']} vs "
                    f"**${r['previous_spend']:,.2f}** in {r['previous_month']} "
                    f"({'↑' if chg > 0 else '↓'}{abs(chg):.1f}%).\n"
                    f"Top categories: {', '.join(f'{k} (${v:,.0f})' for k,v in list(r['top_categories_this_month'].items())[:3])}."
                )

            elif tool == "find_cancellable_subscriptions":
                subs = r.get("recommended_cancellations", [])
                if subs:
                    sub_list = "\n".join(f"  • **{s['merchant']}** — ${s['monthly_cost']:.2f}/mo (${s['annual_cost']:.2f}/yr)" for s in subs)
                    parts.append(
                        f"**💡 Subscription Optimizer**\n"
                        f"Cancel these to save **${r['projected_monthly_savings']:.2f}/month**:\n{sub_list}"
                    )

            elif tool == "credit_score_impact":
                score = r.get("estimated_credit_score", 0)
                badge = "🟢" if score >= 740 else ("🟡" if score >= 670 else "🔴")
                parts.append(
                    f"**Credit Score Estimate** {badge} **~{score}**\n"
                    f"Positive habits: {r['positive_ratio']:.0%} of transactions | "
                    f"Negative: {r['negative_ratio']:.0%}.\n"
                    + (r.get("recommendation", ""))
                )

            elif tool == "spending_summary":
                parts.append(
                    f"**Your Spending Overview 📊**\n"
                    f"Total spend: **${r['total_spend']:,.2f}** | Monthly avg: **${r['avg_monthly_spend']:,.2f}**\n"
                    f"Archetype: *{r['archetype']}* | Top merchant: {r['top_merchant']}\n"
                    f"Top categories: {', '.join(list(r['top_categories'].keys())[:4])}."
                )

            elif tool == "category_advice":
                label = r["label"]
                tips = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(r["tips"]))
                risk_badge = {"low": "🟢 LOW", "medium": "🟡 MEDIUM", "high": "🔴 HIGH", "critical": "🚨 CRITICAL"}.get(r["risk_level"], "⚪")
                reply_text = (
                    f"**{label} — Spending Advice**\n"
                    f"Your spend in this category: **${r['user_spend']:,.2f}**\n"
                    f"Estimated annual waste: *{r['annual_waste_estimate']}*\n"
                    f"Risk level: {risk_badge}\n\n"
                    f"**💡 Tips to save:**\n{tips}"
                )
                if r.get("warning"):
                    reply_text += f"\n\n{r['warning']}"
                parts.append(reply_text)

            elif tool == "savings_plan":
                ops = r.get("opportunities", [])
                savings_list = "\n".join(
                    f"  • **{o['category']}** — Cut ${o['potential_saving']:.2f}/mo ({o['tip']})"
                    for o in ops
                )
                parts.append(
                    f"**💰 Personalized Savings Plan**\n"
                    f"Based on your *{r['archetype']}* spending archetype, here's how to save "
                    f"**${r['potential_monthly_savings']:.2f}/month** (${r['potential_annual_savings']:,.0f}/year):\n\n"
                    f"{savings_list}"
                )

            elif tool == "realtime_fraud_check":
                alerts = r.get("alerts", [])
                status = r["overall_status"]
                parts.append(f"**🔍 Real-Time Fraud Scan**\nScanned {r['transactions_scanned']} recent transactions. {status}")
                if alerts:
                    for a in alerts[:5]:
                        flags = " | ".join(a["flags"][:2])
                        parts.append(f"  • `{a['transaction_date']}` — **{a['merchant']}** ${a['amount']:.2f} → {flags}")

            elif tool == "suspicious_activity_monitor":
                status = r["overall_status"]
                parts.append(f"**👁️ Suspicious Activity Monitor** — {status}")
                for a in r.get("alerts", []):
                    parts.append(f"  {a['emoji']} **{a['title']}**: {a['detail']}")

        return "\n\n".join(parts) if parts else "I couldn't find relevant insights. Try asking about fraud, spending categories, or savings."

    def get_all_users(self) -> list[str]:
        return sorted(self.df["user_id"].unique().tolist())

    def get_chart_data(self, user_id: str) -> dict[str, float]:
        """Return category → amount dict for bar chart rendering."""
        user_df = self.df[self.df["user_id"] == user_id]
        if user_df.empty:
            return {}
        return (
            user_df.groupby("category")["amount"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .round(2)
            .to_dict()
        )
