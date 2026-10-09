import os
import sys
import logging
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Path resolution for project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

def get_resend_api_key() -> str:
    api_key = os.getenv("RESEND_API_KEY")
    if api_key:
        return api_key

    # Fallback to .env and env.yaml
    for fname in [".env", "env.yaml"]:
        fpath = os.path.join(project_root, fname)
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                for line in f:
                    clean_line = line.strip()
                    if "RESEND_API_KEY" in clean_line and not clean_line.startswith("#"):
                        delimiter = ":" if ":" in clean_line else "=" if "=" in clean_line else None
                        if delimiter:
                            return clean_line.split(delimiter, 1)[1].strip().strip('"').strip("'")
    return None

def send_html_email(subject: str, html_content: str):
    to_email = os.getenv("ALERT_EMAIL_TO", "praabhu@gmail.com")
    resend_key = get_resend_api_key()

    if not resend_key:
        logging.error("❌ Resend API key not found in environment, .env, or env.yaml.")
        return False

    url = "https://api.resend.com/emails"
    headers = {
        "Authorization": f"Bearer {resend_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "from": "ICT Trading Bot <onboarding@resend.dev>",
        "to": [to_email],
        "subject": subject,
        "html": html_content
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        if resp.status_code in [200, 201]:
            logging.info(f"📧 Alert email successfully sent via Resend API to {to_email}")
            return True
        else:
            logging.error(f"❌ Resend API Error ({resp.status_code}): {resp.text}")
            return False
    except Exception as e:
        logging.error(f"❌ Resend Dispatch Exception: {e}")
        return False

if __name__ == "__main__":
    send_html_email("🧪 Resend Integration Test", "<h1>Resend API Connected Successfully</h1><p>Your ICT Trading Bot alert system is ready.</p>")
