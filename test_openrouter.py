import httpx, os, base64, json
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")
api_key = os.getenv("OPENROUTER_API_KEY", "")
print(f"API key loaded: {'YES (' + api_key[:8] + '...)' if api_key else 'NO - KEY MISSING'}")

# Test 1: simple text-only request (no image)
print("\nTest 1: text-only request to OpenRouter...")
try:
    r = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "openrouter/auto:free",
            "max_tokens": 5,
            "messages": [{"role": "user", "content": "Say hi"}]
        },
        timeout=30,
    )
    print(f"  Status: {r.status_code}")
    print(f"  Response: {r.text[:200]}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")
    print("  → This means OpenRouter itself is unreachable from your network")
    print("  → Check your VPN, firewall, or proxy settings")