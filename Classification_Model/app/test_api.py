import requests
import json

url = "http://localhost:5000/predict"

# Dữ liệu JSON
data = {
    "symptoms": [1, 200, 30, 68, 2],
    "num": 10
}

# Gửi yêu cầu POST
response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(data))

# In ra kết quả
print(response.json())