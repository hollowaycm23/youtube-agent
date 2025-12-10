
import requests
import sys

def check_connection(url):
    print(f"Testing connection to {url}...")
    try:
        response = requests.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("SUCCESS: Connection established.")
        else:
            print("WARNING: Connection returned non-200 status.")
    except Exception as e:
        print(f"ERROR: Failed to connect. Reason: {e}")

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    check_connection("https://huggingface.co")
    check_connection("https://cdn-lfs.hf.co")
    print("Done.")
