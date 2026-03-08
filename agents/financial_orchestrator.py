"""
Financial Orchestrator — routes user questions to the right agent(s), executes them,
and synthesizes one coherent reply. Output shape: reply, tool_results, show_chart.
"""

from typing import Any

from agents.current_transaction_analyst import CurrentTransactionAnalyst
from agents.transaction_calculation_agent import TransactionCalculationAgent
from agents.historical_review_agent import HistoricalReviewAgent


# Keywords for routing (plan: current → analyst, calculation → calc, history → historical)
CURRENT_KEYWORDS = [
    "current", "this month", "recent", "last 30 days", "last 60", "last 90",
    "latest transactions", "right now", "latest", "this month's", "recent activity",
]
CALCULATION_KEYWORDS = [
    "total", "average", "how much", "calculate", "forecast", "compare months",
    "breakdown", "percentage", "growth", "subscription", "subscriptions",
    "month over month", "mom", "spending by category",
]
HISTORICAL_KEYWORDS = [
    "last 2 years", "past year", "year over year", "yoy", "history", "historical",
    "bank statement", "credit card history", "long term", "trend over time",
    "last year vs", "two years", "24 months", "statement",
]


class FinancialOrchestrator:
    """Routes to current analyst, calculation agent, and/or historical reviewer; synthesizes reply."""

    def __init__(self):
        self.current_analyst = CurrentTransactionAnalyst()
        self.calc_agent = TransactionCalculationAgent()
        self.historical_agent = HistoricalReviewAgent()

    def _route(self, message: str) -> list[str]:
        """Return list of agent keys to call: current, calculation, historical."""
        msg = message.lower()
        agents = []
        if any(k in msg for k in CURRENT_KEYWORDS):
            agents.append("current")
        if any(k in msg for k in CALCULATION_KEYWORDS):
            agents.append("calculation")
        if any(k in msg for k in HISTORICAL_KEYWORDS):
            agents.append("historical")
        if not agents:
            agents.append("current")  # default: recent activity
        return agents

    def _run_agents(self, user_id: str, message: str, agent_keys: list[str]) -> list[dict]:
        """Execute chosen agents and collect structured outputs."""
        results = []
        msg = message.lower()

        if "current" in agent_keys:
            period = "last_30"
            if "last 60" in msg or "60 days" in msg:
                period = "last_60"
            elif "last 90" in msg or "90 days" in msg:
                period = "last_90"
            out = self.current_analyst.run(user_id, period=period)
            results.append({"agent": "current_analyst", "data": out})

        if "calculation" in agent_keys:
            calc = "summary"
            if "forecast" in msg or "next month" in msg:
                calc = "forecast"
            elif "subscription" in msg:
                calc = "subscriptions"
            elif "month over month" in msg or "mom" in msg or "compare months" in msg:
                calc = "mom_change"
            elif "average" in msg:
                calc = "average_by_category"
            out = self.calc_agent.run(user_id, calculation=calc)
            results.append({"agent": "calculation", "data": out})

        if "historical" in agent_keys:
            out = self.historical_agent.run(user_id)
            results.append({"agent": "historical", "data": out})

        return results

    def _synthesize(self, message: str, agent_results: list[dict]) -> str:
        """Turn structured agent results into one coherent reply (template-based)."""
        parts = []
        for item in agent_results:
            agent_name = item.get("agent", "")
            data = item.get("data", item)
            if isinstance(data, dict) and data.get("error"):
                parts.append(f"⚠️ {data['error']}")
                continue

            if agent_name == "current_analyst":
                cm = data.get("current_month") or {}
                if isinstance(cm, dict) and "error" not in cm and "current_month_total" in cm:
                    parts.append(
                        f"**Current month** ({cm.get('current_month', 'N/A')}): "
                        f"**${cm['current_month_total']:,.2f}** across {cm.get('transaction_count', 0)} transactions. "
                        f"Top categories: {', '.join(list(cm.get('by_category', {}).keys())[:5])}."
                    )
                ln = data.get("last_n_days") or {}
                if isinstance(ln, dict) and "error" not in ln and "total" in ln:
                    parts.append(
                        f"**Last {ln.get('days', 30)} days:** **${ln['total']:,.2f}** total. "
                        f"By category: {ln.get('by_category', {})}."
                    )
                fr = data.get("recent_fraud_risk") or {}
                if isinstance(fr, dict) and fr.get("alerts_count", 0) > 0:
                    parts.append(
                        f"**Fraud/risk:** {fr['alerts_count']} alert(s) in recent transactions."
                    )

            elif agent_name == "calculation":
                if "total" in data:
                    parts.append(f"**Total spend (range):** **${data['total']:,.2f}**.")
                if "growth_pct" in data:
                    parts.append(
                        f"**Month-over-month:** {data.get('growth_pct', 0):.1f}% change "
                        f"({data.get('previous_month')} → {data.get('current_month')})."
                    )
                if "forecast_total" in data:
                    parts.append(
                        f"**Forecast (next month):** ~**${data['forecast_total']:,.2f}** "
                        f"(subscriptions ~${data.get('subscription_component', 0):,.2f}, variable ~${data.get('variable_component', 0):,.2f})."
                    )
                if "monthly_total" in data and data.get("tool") == "subscription_totals":
                    parts.append(f"**Subscriptions:** **${data['monthly_total']:,.2f}/month** by merchant: {data.get('by_merchant', {})}.")

            elif agent_name == "historical":
                yt_block = data.get("yearly_totals") or {}
                if isinstance(yt_block, dict) and "yearly_totals" in yt_block and "error" not in yt_block:
                    yt = yt_block.get("yearly_totals", {})
                    parts.append(f"**Yearly totals (last 2 years):** {yt}.")
                yoy_block = data.get("yoy_change") or {}
                if isinstance(yoy_block, dict) and "yoy_change_pct" in yoy_block:
                    yoy = yoy_block
                    parts.append(
                        f"**Year-over-year:** {yoy.get('year_previous')} → {yoy.get('year_latest')}: "
                        f"**{yoy['yoy_change_pct']:.1f}%** change "
                        f"(${yoy.get('total_previous', 0):,.2f} → ${yoy.get('total_latest', 0):,.2f})."
                    )
                evo_block = data.get("category_evolution") or {}
                if isinstance(evo_block, dict) and "monthly_avg_by_category" in evo_block:
                    evo = evo_block.get("monthly_avg_by_category", {})
                    top = dict(list(evo.items())[:5])
                    parts.append(f"**Historical monthly average by category (top 5):** {top}.")

        if not parts:
            return "I've run the requested analysis but couldn't summarize results. Please try a more specific question (e.g. 'current month spending', 'total last 30 days', 'year over year comparison')."
        return "\n\n".join(parts)

    def chat(self, message: str, user_id: str) -> dict[str, Any]:
        """Route → execute → synthesize. Same output shape as FinancialAdvisorAgent.chat."""
        agent_keys = self._route(message)
        agent_results = self._run_agents(user_id, message, agent_keys)
        # Flatten for tool_results (list of dicts like existing advisor)
        tool_results = []
        for ar in agent_results:
            data = ar.get("data", ar)
            if isinstance(data, dict):
                tool_results.append(data)
            else:
                tool_results.append({"agent": ar.get("agent"), "result": data})
        reply = self._synthesize(message, agent_results)
        show_chart = any(
            k in message.lower()
            for k in ["chart", "graph", "breakdown", "category", "trend", "overview", "summary"]
        )
        return {
            "reply": reply,
            "tool_results": tool_results,
            "user_id": user_id,
            "show_chart": show_chart,
        }
