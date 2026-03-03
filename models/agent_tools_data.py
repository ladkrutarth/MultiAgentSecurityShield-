"""
Data-only tools for risk and user profile. No MLX/LLM imports.
Used by the API so /api/fraud/high-risk and /api/user/{id}/risk work without loading GuardAgent.
"""

from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATHS = {
    "features": PROJECT_ROOT / "dataset" / "csv_data" / "features_output.csv",
    "fraud_scores": PROJECT_ROOT / "dataset" / "csv_data" / "fraud_scores_output.csv",
    "auth_profiles": PROJECT_ROOT / "dataset" / "csv_data" / "auth_profiles_output.csv",
    "transactions": PROJECT_ROOT / "dataset" / "csv_data" / "transactions_3000.csv",
}

_fraud_df = None
_auth_df = None


def _load_cache():
    global _fraud_df, _auth_df
    fraud_path = DATA_PATHS["fraud_scores"]
    if fraud_path.exists() and _fraud_df is None:
        _fraud_df = pd.read_csv(fraud_path)
        _fraud_df.columns = [c.upper() for c in _fraud_df.columns]
    auth_path = DATA_PATHS["auth_profiles"]
    if auth_path.exists() and _auth_df is None:
        _auth_df = pd.read_csv(auth_path)
        _auth_df.columns = [c.upper() for c in _auth_df.columns]


def tool_get_user_risk_profile(user_id: str) -> Dict[str, Any]:
    _load_cache()
    result = {"user_id": user_id, "found": False}
    if _auth_df is not None:
        user_data = _auth_df[_auth_df["USER_ID"] == user_id]
        if not user_data.empty:
            row = user_data.iloc[0]
            result.update({
                "found": True,
                "security_level": str(row.get("RECOMMENDED_SECURITY_LEVEL", "N/A")),
                "avg_risk": float(row.get("AVG_RISK", 0)),
                "high_risk_count": int(row.get("HIGH_RISK_COUNT", 0))
            })
    if _fraud_df is not None:
        user_data = _fraud_df[_fraud_df["USER_ID"] == user_id]
        if not user_data.empty:
            result.update({
                "found": True,
                "txn_count": len(user_data),
                "avg_combined_score": float(user_data["COMBINED_RISK_SCORE"].mean()),
                "risk_distribution": user_data["RISK_LEVEL"].value_counts().to_dict()
            })
    return result


def tool_get_high_risk_transactions(limit: int = 10) -> List[Dict]:
    _load_cache()
    if _fraud_df is None:
        return []
    df = _fraud_df.sort_values("COMBINED_RISK_SCORE", ascending=False).head(limit)
    return df.to_dict("records")
