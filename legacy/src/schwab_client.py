import os
import logging
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class SchwabMarketDataClient:
    def __init__(self, env_path="env.yaml"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, ".."))

        self.client_id = os.getenv("SCHWAB_CLIENT_ID")
        self.client_secret = os.getenv("SCHWAB_CLIENT_SECRET")
        self.refresh_token = os.getenv("SCHWAB_REFRESH_TOKEN")
        self.access_token = None

        # Fallback: Parse .env or env.yaml manually if not in OS environment
        if not self.client_id or not self.refresh_token:
            for fname in [".env", "env.yaml"]:
                fpath = os.path.join(project_root, fname)
                if os.path.exists(fpath):
                    with open(fpath, "r") as f:
                        for line in f:
                            clean_line = line.strip()
                            if not clean_line or clean_line.startswith("#"):
                                continue
                            delimiter = ":" if ":" in clean_line else "=" if "=" in clean_line else None
                            if not delimiter:
                                continue
                            k, v = clean_line.split(delimiter, 1)
                            k, v = k.strip(), v.strip().strip('"').strip("'")

                            if k == "SCHWAB_CLIENT_ID" and not self.client_id:
                                self.client_id = v
                            elif k == "SCHWAB_CLIENT_SECRET" and not self.client_secret:
                                self.client_secret = v
                            elif k == "SCHWAB_REFRESH_TOKEN" and not self.refresh_token:
                                self.refresh_token = v

    def get_access_token(self) -> str:
        """Exchanges long-lived refresh token for short-lived 30-min OAuth access token."""
        token_url = "https://api.schwabapi.com/v1/oauth/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token
        }
        try:
            resp = requests.post(token_url, headers=headers, data=data, auth=(self.client_id, self.client_secret), timeout=15)
            if resp.status_code == 200:
                self.access_token = resp.json().get("access_token")
                return self.access_token
            else:
                logging.error(f"❌ Schwab Token Exchange Failed ({resp.status_code}): {resp.text}")
                return None
        except Exception as e:
            logging.error(f"❌ Schwab Token Exchange Exception: {e}")
            return None

    def get_option_chain(self, symbol: str) -> dict:
        """Fetches near-the-money option chain and Greeks from Schwab API."""
        token = self.get_access_token()
        if not token:
            return None

        chain_url = "https://api.schwabapi.com/marketdata/v1/chains"
        headers = {"Authorization": f"Bearer {token}"}
        params = {
            "symbol": symbol,
            "contractType": "ALL",
            "strikeCount": "10",
            "includeUnderlyingQuote": "TRUE",
            "strategy": "SINGLE"
        }
        try:
            resp = requests.get(chain_url, headers=headers, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            else:
                logging.error(f"❌ Schwab Option Chain Fetch Failed ({resp.status_code}): {resp.text}")
                return None
        except Exception as e:
            logging.error(f"❌ Schwab Option Chain Fetch Exception: {e}")
            return None
