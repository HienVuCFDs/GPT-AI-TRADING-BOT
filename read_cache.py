import json
import os

cache_file = os.path.expanduser('~/.trading_bot/license/license_cache.json')
with open(cache_file) as f:
    data = json.load(f)

print("Access Token:", data.get('access_token')[:50] if data.get('access_token') else 'None')
print("Refresh Token:", data.get('refresh_token')[:50] if data.get('refresh_token') else 'None')
print("Username:", data.get('username'))
print("All keys:", list(data.keys()))
