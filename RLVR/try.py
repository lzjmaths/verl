import requests

url = "http://127.0.0.1:10000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

data = {
    "model": "Qwen3-14B",
    "messages": [
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
}

response = requests.post(url, headers=headers, json=data)
print(response.json())