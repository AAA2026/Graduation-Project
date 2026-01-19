
import json
import uuid
from datetime import datetime
from pathlib import Path

# Define data path
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "backend" / "data"
DEMO_REQUESTS_FILE = DATA_DIR / "demo_requests.json"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_demo_requests():
    """Load all demo requests."""
    if not DEMO_REQUESTS_FILE.exists():
        return []
    try:
        with open(DEMO_REQUESTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[DEMO SERVICE ERROR] Failed to load requests: {e}")
        return []

def save_demo_request(data):
    """Save a new demo request."""
    requests = get_demo_requests()
    
    new_request = {
        "id": str(uuid.uuid4()),
        "fullName": data.get("fullName", ""),
        "email": data.get("email", ""),
        "phone": data.get("phone", ""),
        "organization": data.get("organization", ""),
        "role": data.get("role", ""),
        "cameras": data.get("cameras", ""),
        "message": data.get("message", ""),
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "created_at_human": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "updated_at": None
    }
    
    requests.insert(0, new_request)  # Newest first
    
    try:
        with open(DEMO_REQUESTS_FILE, "w", encoding="utf-8") as f:
            json.dump(requests, f, indent=2, ensure_ascii=False)
        return new_request
    except Exception as e:
        print(f"[DEMO SERVICE ERROR] Failed to save request: {e}")
        return None

def update_demo_request_status(request_id, status):
    """Update the status of a request."""
    requests = get_demo_requests()
    updated = False
    
    for req in requests:
        if req["id"] == request_id:
            req["status"] = status
            req["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            updated = True
            break
            
    if updated:
        try:
            with open(DEMO_REQUESTS_FILE, "w", encoding="utf-8") as f:
                json.dump(requests, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"[DEMO SERVICE ERROR] Failed to update request: {e}")
            return False
    return False
