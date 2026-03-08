import requests
import json

BASE_URL = "http://localhost:8000/api"

def test_security_chat_deception():
    print("Testing Security Chat Deception...")
    # Normal query (no session_id)
    payload = {"message": "How do I secure my account?"}
    resp = requests.post(f"{BASE_URL}/security/chat", json=payload, timeout=30)
    print(f"Normal Response Status: {resp.status_code}")
    reply = resp.json().get("reply") or ""
    print(f"Normal Reply: {reply[:100]}...")
    # Attacker query -> diversion + fast decoy response
    payload = {"message": "I want to bypass the authentication and get root access to the database.", "session_id": "test-session-1"}
    resp = requests.post(f"{BASE_URL}/security/chat", json=payload, timeout=30)
    data = resp.json()
    print(f"Attacker Response Status: {resp.status_code}")
    print(f"Deception Status: {data.get('status')}")
    print(f"Deception Reply: {data.get('reply')}")
    assert data.get("status") == "deception", "Expected status=deception for attack keywords"

def test_risk_based_diversion():
    print("\nTesting Risk-Based Diversion (transaction validate)...")
    # High-risk-like transaction with session_id
    payload = {"session_id": "txn-session-1", "category": "transfer", "amt": 4000.0, "gender": "M", "state": "CA", "merchant": "fraud_Acme", "hour": 14, "day_of_week": 3}
    resp = requests.post(f"{BASE_URL}/transactions/validate", json=payload, timeout=10)
    data = resp.json()
    print(f"Validate Status: {resp.status_code}, risk_score={data.get('risk_score')}, diverted={data.get('diverted')}")
    if data.get("risk_score", 0) >= 20 and data.get("diverted"):
        print("  -> Session diverted to decoy (ADDF).")
    # Same session gets decoy user risk
    resp2 = requests.get(f"{BASE_URL}/user/USER_001/risk", params={"session_id": "txn-session-1"}, timeout=10)
    print(f"User risk (diverted session): {resp2.status_code}, found={resp2.json().get('found')}")

def test_threat_intel_export():
    print("\nTesting Threat Intel Export...")
    resp = requests.get(f"{BASE_URL}/honeypot/threat-intel", timeout=10)
    data = resp.json()
    logs = data.get("logs", [])
    print(f"Threat Intel Status: {resp.status_code}, entries={len(logs)}")
    if logs:
        entry = logs[0]
        print(f"  Sample: tactic={entry.get('tactic_category')}, faas={entry.get('faas_indicators')}")

def test_honeypot_endpoints():
    print("\nTesting Honeypot Endpoints...")
    resp = requests.get(f"{BASE_URL}/honeypot/filesystem?depth=2", timeout=10)
    print(f"Filesystem Status: {resp.status_code}")
    resp = requests.get(f"{BASE_URL}/honeypot/transactions/USER_ATTACKER?count=3", timeout=10)
    print(f"Transactions Status: {resp.status_code}")
    resp = requests.get(f"{BASE_URL}/honeypot/logs", timeout=10)
    print(f"Logs Status: {resp.status_code}, count={len(resp.json().get('logs', []))}")

if __name__ == "__main__":
    try:
        test_security_chat_deception()
        test_risk_based_diversion()
        test_threat_intel_export()
        test_honeypot_endpoints()
        print("\n✅ ADDF verification successful!")
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        print("Note: Ensure the API server is running (uvicorn api.main:app --reload)")
