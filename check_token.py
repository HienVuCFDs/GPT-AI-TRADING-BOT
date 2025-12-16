import json
import base64
import datetime

token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzY1NTYxOTQ3LCJpYXQiOjE3NjU1NTgzNDcsImp0aSI6ImIxOGY1NzUwY2MyYjQwODQ5ZTkzMThiYjY4Y2NhNTdlIiwidXNlcl9pZCI6IjE5In0.JOJ1fIZuTblgydLo7I9Uw0xM6TDjXjrgKpjduupKwW0'
payload = token.split('.')[1]
decoded = base64.b64decode(payload + '==')
data = json.loads(decoded)
now = int(datetime.datetime.now().timestamp())

print(f'Now: {now}')
print(f'Expires: {data["exp"]}')
print(f'Expired: {now > data["exp"]}')
if data["exp"] - now > 0:
    print(f'Valid for: {(data["exp"] - now) / 3600:.2f} hours')
else:
    print(f'Expired {abs(data["exp"] - now) / 3600:.2f} hours ago')
