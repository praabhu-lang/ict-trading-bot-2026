import requests
import json
import urllib.parse

CLIENT_ID = __import__("os").environ["SCHWAB_CLIENT_ID"]
CLIENT_SECRET = __import__("os").environ["SCHWAB_CLIENT_SECRET"]
REDIRECT_URI = "https://ai-trading-dashboard-464783405434.us-central1.run.app/"

print("=" * 60)
print("SCHWAB OAUTH TOKEN GENERATOR")
print("=" * 60)

auth_url = f"https://api.schwabapi.com/v1/oauth/authorize?client_id={CLIENT_ID}&redirect_uri={urllib.parse.quote(REDIRECT_URI)}"
print(f"\n1. Open this URL in your browser:\n\n{auth_url}\n")
print("2. Log in with your Schwab retail credentials and click Approve.")
print("3. Paste the FULL redirected URL from your browser address bar below.")
print("=" * 60)

redirected_url = input("\nPaste Full Redirected URL here: ").strip()

# Extract the raw code param
if "?code=" in redirected_url:
    raw_code = redirected_url.split("?code=")[1].split("&")[0]
else:
    raw_code = redirected_url

# Unquote HTML encoding (%40 -> @)
auth_code = urllib.parse.unquote(raw_code)

print(f"\n⚡ Extracted Code: {auth_code}")
print("⚡ Exchanging Code for Refresh Token...")

token_url = "https://api.schwabapi.com/v1/oauth/token"
headers = {"Content-Type": "application/x-www-form-urlencoded"}
data = {
    "grant_type": "authorization_code",
    "code": auth_code,
    "redirect_uri": REDIRECT_URI
}

resp = requests.post(token_url, headers=headers, data=data, auth=(CLIENT_ID, CLIENT_SECRET))

if resp.status_code == 200:
    res_data = resp.json()
    refresh_token = res_data.get("refresh_token")
    print("\n" + "=" * 60)
    print("SUCCESS! Schwab Refresh Token Generated:")
    print("=" * 60)
    print(refresh_token)
    print("=" * 60)
    
    # Write directly to env.yaml
    with open("env.yaml", "a") as f:
        f.write(f"\nSCHWAB_CLIENT_ID: \"{CLIENT_ID}\"\n")
        f.write(f"SCHWAB_CLIENT_SECRET: \"{CLIENT_SECRET}\"\n")
        f.write(f"SCHWAB_REFRESH_TOKEN: \"{refresh_token}\"\n")
    print("\n✅ Automatically appended credentials to env.yaml!")
else:
    print(f"\n❌ Exchange Failed ({resp.status_code}): {resp.text}")
