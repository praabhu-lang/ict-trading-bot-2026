import os
import requests
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
env_path = os.path.join(project_root, "env.yaml")

client_id = os.getenv("SCHWAB_CLIENT_ID")
client_secret = os.getenv("SCHWAB_CLIENT_SECRET")
refresh_token = os.getenv("SCHWAB_REFRESH_TOKEN")

if not client_id or not refresh_token:
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if "SCHWAB_CLIENT_ID" in line:
                    client_id = line.split(":", 1)[1].strip().strip('"')
                elif "SCHWAB_CLIENT_SECRET" in line:
                    client_secret = line.split(":", 1)[1].strip().strip('"')
                elif "SCHWAB_REFRESH_TOKEN" in line:
                    refresh_token = line.split(":", 1)[1].strip().strip('"')

print("=" * 60)
print("TESTING CHARLES SCHWAB MARKET DATA API INTEGRATION")
print("=" * 60)

# 1. OAuth Access Token Exchange
token_url = "https://api.schwabapi.com/v1/oauth/token"
token_headers = {"Content-Type": "application/x-www-form-urlencoded"}
token_data = {
    "grant_type": "refresh_token",
    "refresh_token": refresh_token,
}

try:
    token_resp = requests.post(token_url, headers=token_headers, data=token_data, auth=(client_id, client_secret), timeout=15)
    if token_resp.status_code == 200:
        access_token = token_resp.json().get("access_token")
        print("✅ Access Token successfully generated!")
    else:
        print(f"❌ Token Exchange Failed ({token_resp.status_code}): {token_resp.text}")
        exit(1)
except Exception as e:
    print(f"❌ Connection error during token exchange: {e}")
    exit(1)

# 2. Fetch Live Quote for SPY
market_headers = {"Authorization": f"Bearer {access_token}"}
quote_url = "https://api.schwabapi.com/marketdata/v1/quotes?symbols=SPY"

try:
    quote_resp = requests.get(quote_url, headers=market_headers, timeout=15)
    if quote_resp.status_code == 200:
        spy_data = quote_resp.json().get("SPY", {})
        last_price = spy_data.get("quote", {}).get("lastPrice", 0.0)
        print(f"✅ Live SPY Quote Fetched: ${last_price:.2f}")
    else:
        print(f"⚠️ Quote Error ({quote_resp.status_code}): {quote_resp.text}")
except Exception as e:
    print(f"⚠️ Quote Request Exception: {e}")

# 3. Fetch Optimized Live Option Chain for SPY
chain_url = "https://api.schwabapi.com/marketdata/v1/chains"
params = {
    "symbol": "SPY",
    "contractType": "ALL",
    "strikeCount": "10",
    "includeUnderlyingQuote": "TRUE",
    "strategy": "SINGLE"
}

try:
    chain_resp = requests.get(chain_url, headers=market_headers, params=params, timeout=30)
    if chain_resp.status_code == 200:
        chain_data = chain_resp.json()
        underlying_price = chain_data.get("underlyingPrice", 0.0)
        calls = chain_data.get("callExpDateMap", {})
        puts = chain_data.get("putExpDateMap", {})
        print(f"✅ Live SPY Option Chain Fetched | Spot Price: ${underlying_price:.2f}")
        print(f"   Near-the-Money Call Dates: {len(calls)} | Put Dates: {len(puts)}")
    else:
        print(f"⚠️ Option Chain Error ({chain_resp.status_code}): {chain_resp.text}")
except Exception as e:
    print(f"⚠️ Option Chain Request Exception: {e}")

print("=" * 60)
