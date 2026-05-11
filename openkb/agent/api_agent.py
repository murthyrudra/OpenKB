import time 
from datetime import datetime, timedelta
import requests
import json

BASE_URL = "https://api.agriwatch.in/apgov"
HEADERS = {"Content-Type": "application/json"}
TOKEN_USERID = "apgov_agriwatch"
MAX_RETRIES = 3
RETRY_DELAY = 2 # seconds




class AgriwatchClient:
    def __init__(self):
        self.token_key = None
        self.get_token()

    def get_token(self):
        url = f"{BASE_URL}/token.php"
        payload = {"token_userid": TOKEN_USERID}

        response = requests.post(url, headers=HEADERS, json=payload)
        response.raise_for_status()

        data = response.json()
        if data.get("status_code") != 200:
            raise Exception(f"Failed to get token: {data}")

        self.token_key = data["token_key"]
        print("🔑 New token generated")

    def call_api(self, endpoint, extra_payload=None):
        url = f"{BASE_URL}/{endpoint}"

        for attempt in range(MAX_RETRIES):
            payload = {"token_userid": TOKEN_USERID, "token_key": self.token_key}

            if extra_payload:
                payload.update(extra_payload)

            try:
                response = requests.post(url, headers=HEADERS, json=payload)

                # If unauthorized or token expired → regenerate token
                if response.status_code in [401, 403]:
                    print("⚠️ Token expired/invalid. Regenerating...")
                    self.get_token()
                    continue

                response.raise_for_status()
                data = response.json()

                # Optional: detect token failure via response body
                if isinstance(data, dict) and data.get("status") == "error":
                    if "token" in str(data).lower():
                        print("⚠️ Token issue in response. Regenerating...")
                        self.get_token()
                        continue

                return data

            except Exception as e:
                print(f"Retry {attempt + 1}/{MAX_RETRIES} failed: {e}")
                time.sleep(RETRY_DELAY)

        raise Exception(f"Failed API call after retries: {endpoint}")

BASE_ENAM_URL = "https://enam.gov.in/StateBoardWebSrv/rest/stateboard"

MAX_RETRIES = 1
RETRY_DELAY = 2  # seconds


class ENAMClient:
    def __init__(self):
        pass

    def call_api(self, endpoint, method="POST", headers=None, payload=None):
        url = f"{BASE_ENAM_URL}/{endpoint}"

        for attempt in range(MAX_RETRIES):
            try:
                if method == "GET":
                    response = requests.get(url, headers=headers, params=payload)
                else:
                    response = requests.post(url, headers=headers, data=payload)

                response.raise_for_status()
                return response.json()

            except Exception as e:
                print(f"Retry {attempt + 1}/{MAX_RETRIES} failed: {e}")
                time.sleep(RETRY_DELAY)

        raise Exception(f"Failed API call after retries: {endpoint}")
