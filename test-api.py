import os
import requests
from dotenv import load_dotenv

# Load the API key from your .env file
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

print("Pinging OpenRouter...")

try:
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "liquid/lfm-2.5-2.6b:free",
            "messages": [
                {"role": "user", "content": "Hello! Are you online? "}
            ]
        }
    )

    # Check if the request was successful (HTTP 200)
    if response.status_code == 200:
        data = response.json()
        print("\n✅ API is Working!")
        print("Bot Reply:", data["choices"][0]["message"]["content"])
    else:
        print(f"\n❌ API Failed (Status Code: {response.status_code})")
        print("Error Details:", response.text)

except Exception as e:
    print(f"\n❌ A network or code error occurred: {e}")