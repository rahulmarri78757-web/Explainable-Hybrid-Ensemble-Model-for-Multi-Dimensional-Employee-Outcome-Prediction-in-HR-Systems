import httpx
import sys

BASE_URL = "http://localhost:8000"

def check_health():
    try:
        response = httpx.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Backend is running")
            return True
        else:
            print(f"❌ Backend returned status {response.status_code}")
            return False
    except httpx.ConnectError:
        print("❌ Could not connect to backend (is it running?)")
        return False

if __name__ == "__main__":
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)
