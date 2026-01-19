
import os
import io
import csv
import json
import uuid
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
try:
    import pdfkit
except ImportError:
    pdfkit = None

# --- Flask App and In-Memory Stores ---
app = Flask(__name__)
CORS(app)
# Force threading mode for Windows stability / dev server compatibility
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
PROJECT_ROOT = Path(__file__).parent.parent
VIDEO_DIR = PROJECT_ROOT / 'Videos'

# Root route serving removed to prevent conflicts
# All videos are served via /videos/<path:filename> defined below

if not hasattr(app, 'users'):
    app.users = {
        'admin@vigil.com': {'password': 'admin123', 'role': 'admin', 'name': 'Admin User'},
        'officer@vigil.com': {'password': 'officer123', 'role': 'officer', 'name': 'Officer Smith'},
        'security@vigil.com': {'password': 'security123', 'role': 'security', 'name': 'Chief Martinez'},
    }
if not hasattr(app, 'notification_prefs'):
    app.notification_prefs = {}
if not hasattr(app, 'notification_rules'):
    app.notification_rules = {}

# --- Model Retraining Service ---
def retrain_model(training_data_path=None):
    """
    Triggers the model retraining pipeline.
    """
    try:
        from backend.ai.retrainer import retrain_pipeline
        return retrain_pipeline()
    except Exception as e:
        print(f"[RETRAIN ERROR] {e}")
        return {"status": "error", "message": str(e)}


# --- Camera Simulator and State ---
try:
    from backend.config import DEFAULT_CAMERAS, VIOLENCE_THRESHOLD, ACCIDENT_THRESHOLD
    from backend.services.camera_simulator import CameraSimulator
    from backend.services.camera_manager import camera_states, get_offline_mode_state, set_offline_mode_state
    from backend.services.incident_storage import add_incident, get_incidents
    from backend.ai.inference import run_inference
except ImportError:
    from config import DEFAULT_CAMERAS, VIOLENCE_THRESHOLD, ACCIDENT_THRESHOLD
    from services.camera_simulator import CameraSimulator
    from services.camera_manager import camera_states, get_offline_mode_state, set_offline_mode_state
    from services.incident_storage import add_incident, get_incidents, get_incident_by_id, mark_incident_resolved, acknowledge_incident, dispatch_incident, list_security_roster, clear_incidents, get_incident_stats, ack_all_incidents
    from ai.inference import run_inference
import time

# Start camera simulator on app startup
simulator = None
def start_simulator():
    global simulator
    if simulator is None:
        simulator = CameraSimulator(camera_ids=DEFAULT_CAMERAS, video_dir=VIDEO_DIR, rotation_interval=5, violence_probability=0.15)
        simulator.start()
        print("[DEBUG] CameraSimulator started.")
        # Link simulator state to global camera_states
        def sync_states():
            while True:
                for cid, state in simulator.camera_states.items():
                    camera_states.set(cid, state)
                # [DEBUG] Print current camera_states
                # print(f"[DEBUG] camera_states: {camera_states.all()}")
                time.sleep(1)
        threading.Thread(target=sync_states, daemon=True).start()

    # Force initial camera states if empty (guarantee frontend always gets cameras)
    if not camera_states.all():
        print("[DEBUG] Forcing initial camera states...")
        video_root = Path(__file__).parent.parent / "Videos"
        video_files = []
        for subfolder in video_root.iterdir():
            if subfolder.is_dir():
                video_files += list(subfolder.glob("*.mp4"))
                video_files += list(subfolder.glob("*.MP4"))
        for idx, cid in enumerate(DEFAULT_CAMERAS):
            chosen_video = video_files[idx % len(video_files)] if video_files else None
            rel_path = f"{chosen_video.parent.name}/{chosen_video.name}" if chosen_video else None
            camera_states.set(cid, {
                "camera_id": cid,
                "status": "online",
                "video": rel_path,
                "event": "none",
                "confidence": 0.0,
                "last_update": None
            })
        print("[DEBUG] Initial camera states set.")

start_simulator()

@app.route('/api/live-status', methods=['GET'])
def live_status():
    # live_status endpoint should simply return the current state from camera_states
    # which is being updated by the simulator thread.
    all_states_dict = camera_states.all()
    states = []
    
    for cid in DEFAULT_CAMERAS:
        # Get state or default
        state = all_states_dict.get(cid)
        if not state:
            # Fallback if simulator hasn't initialized this camera yet
            state = {
                "camera_id": cid,
                "status": "offline",
                "video": "",
                "event": "none", 
                "confidence": 0.0
            }
        states.append(state)

    return jsonify({"cameras": states})




@app.route('/auth/login', methods=['POST'])
def auth_login():
    # Simple demo login (replace with real auth as needed)
    data = request.get_json()
    email = data.get('email', '').lower()
    password = data.get('password', '')
    role = data.get('role', '')
    users = app.users
    user = users.get(email)
    if user and user['password'] == password and user['role'] == role:
        token = f'demo-token-{role}'
        return jsonify({'token': token, 'role': user['role'], 'user': user['name']})
    return jsonify({'message': 'Invalid credentials'}), 401

# --- User Registration Endpoint ---
@app.route('/auth/register', methods=['POST'])
def auth_register():
    data = request.get_json()
    email = data.get('email', '').lower()
    password = data.get('password', '')
    role = data.get('role', 'officer')
    name = data.get('name', '')
    if not email or not password or not name:
        return jsonify({'error': 'Missing required fields'}), 400
    users = app.users
    if email in users:
        return jsonify({'error': 'User already exists'}), 409
    users[email] = {'password': password, 'role': role, 'name': name}
    return jsonify({'success': True, 'email': email, 'role': role, 'name': name})

# --- Password Reset Endpoint (Demo) ---
@app.route('/auth/reset-password', methods=['POST'])
def auth_reset_password():
    data = request.get_json()
    email = data.get('email', '').lower()
    new_password = data.get('new_password', '')
    if not email or not new_password:
        return jsonify({'error': 'Missing required fields'}), 400
    users = app.users
    if email not in users:
        return jsonify({'error': 'User not found'}), 404
    users[email]['password'] = new_password
    return jsonify({'success': True, 'email': email})

# --- User Role Management Endpoint (Demo) ---
@app.route('/auth/set-role', methods=['POST'])
def auth_set_role():
    data = request.get_json()
    email = data.get('email', '').lower()
    new_role = data.get('role', '')
    if not email or not new_role:
        return jsonify({'error': 'Missing required fields'}), 400
    users = app.users
    if email not in users:
        return jsonify({'error': 'User not found'}), 404
    users[email]['role'] = new_role
    return jsonify({'success': True, 'email': email, 'role': new_role})

offline_mode = False

# ...existing code...



# --- Model Retraining Endpoint ---
@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    """
    Trigger model retraining. Accepts optional file upload or data path.
    Example: POST /api/retrain with JSON {"data_path": "..."} or multipart file.
    """
    # Handle file upload (multipart/form-data)
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        # Save to temp location
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)
        result = retrain_model(training_data_path=file_path)
        # Clean up temp file if needed
        # os.remove(file_path)
        return jsonify(result)
    # Handle JSON data (data_path)
    data = request.get_json(silent=True) or {}
    data_path = data.get('data_path')
    result = retrain_model(training_data_path=data_path)
    return jsonify(result)

# Simulator stats endpoint (debug/monitoring)
@app.route("/api/simulator-stats", methods=["GET"])
def simulator_stats():
    """
    Returns camera simulator statistics.
    Useful for monitoring the live demo system.
    """
    simulator = get_simulator()
    return jsonify(simulator.get_stats())

# Offline mode endpoints (toggle AI incident creation)
@app.route("/api/offline-mode", methods=["GET"])
def get_offline_mode():
    """Get current offline mode status."""
    offline = get_offline_mode_state()
    return jsonify({
        "offline_mode": offline,
        "status": "offline" if offline else "online"
    })

@app.route("/api/offline-mode", methods=["POST"])
def toggle_offline_mode():
    """Toggle offline mode (prevents incident creation)."""
    data = request.get_json(silent=True) or {}
    
    current = get_offline_mode_state()
    # If explicit state is provided, use it; otherwise toggle
    if "offline_mode" in data:
        new_state = data.get("offline_mode", False)
    else:
        new_state = not current
    
    set_offline_mode_state(new_state)
    
    status_msg = "🔴 OFFLINE: AI incident detection PAUSED" if new_state else "🟢 ONLINE: AI incident detection ACTIVE"
    print(status_msg)
    
    return jsonify({
        "success": True,
        "offline_mode": new_state,
        "status": "offline" if new_state else "online",
        "message": status_msg
    })

# Demo processing endpoint to simulate AI-triggered incident
@app.route("/api/process-demo", methods=["POST"])
def demo_incident():
    # In a real system, you'd pass the actual video path per camera
    result = process_video("CAM-01", "demo.mp4")
    return jsonify({"status": "processed", "result": result})

# Batch processing endpoint to simulate multi-camera incidents
@app.route("/api/process-batch", methods=["POST"])
def process_batch():
    for cam in DEFAULT_CAMERAS:
        process_video(cam, "demo.mp4")
    return jsonify({"status": "batch processed", "cameras": DEFAULT_CAMERAS})

# Process custom video
@app.route("/api/process-video", methods=["POST"])
def process_custom_video():
    data = request.get_json()
    video_path = data.get("video_path", "demo.mp4")
    camera_id = data.get("camera_id", "CAM-01")
    
    result = process_video(camera_id, video_path)
    return jsonify({"status": "processed", "camera_id": camera_id, "result": result})

# Stream placeholder endpoint (no real CCTV required)
@app.route("/api/stream/<camera_id>", methods=["GET"])
def stream_placeholder(camera_id):
    return jsonify({
        "camera_id": camera_id,
        "stream_url": "/static/demo_feed.mp4",
        "status": "ready"
    })



# ...existing code...

# Serve video files from the videos/ directory (must be last route!)
@app.route("/videos/<path:filename>", methods=["GET"])
def serve_video(filename):
    """
    Serve video files to frontend from any subfolder in Videos/.
    Robustly handles nested folders and path normalization.
    """
    import os
    print(f"[VIDEO SERVE] Requested: {filename}")
    
    # 1. Resolve base video directory absolute path
    base_dir = VIDEO_DIR.resolve()
    
    # 2. Join requested filename with base dir
    # Note: We strip leading slashes/backslashes manually to prevent join() from treating it as root
    clean_filename = filename.lstrip("/\\")
    requested_path = (base_dir / clean_filename).resolve()
    
    print(f"[VIDEO SERVE] Resolved path: {requested_path}")
    
    # 3. Security Check: Ensure requested path is still inside base_dir
    # commonpath raises ValueError if paths are on different drives, which is good
    try:
        common = os.path.commonpath([base_dir, requested_path])
        if str(common) != str(base_dir):
            print(f"[VIDEO SERVE] Security alert: Path traversal attempted! {requested_path} is outside {base_dir}")
            return jsonify({"error": "Invalid video path"}), 403
    except ValueError:
        print("[VIDEO SERVE] Security alert: Path on different drive?")
        return jsonify({"error": "Invalid video path"}), 403

    # 4. Check existence and serve
    if requested_path.exists() and requested_path.is_file():
        print(f"[VIDEO SERVE] Serving: {requested_path}")
        return send_from_directory(requested_path.parent, requested_path.name)
        
    # 5. Fallback for case-insensitive extension issues (common in mixed Linux/Windows/iOS envs)
    alt_ext = None
    if requested_path.suffix.lower() == ".mp4":
        alt_ext = requested_path.with_suffix(".MP4")
    elif requested_path.suffix.upper() == ".MP4":
        alt_ext = requested_path.with_suffix(".mp4")
        
    if alt_ext and alt_ext.exists() and alt_ext.is_file():
        print(f"[VIDEO SERVE] Serving alternate extension: {alt_ext}")
        return send_from_directory(alt_ext.parent, alt_ext.name)

    print(f"[VIDEO SERVE] File not found: {requested_path}")
    return jsonify({"error": "Video file not found"}), 404



# Get specific incident by ID
@app.route("/api/incidents/<incident_id>", methods=["GET"])
def api_get_incident(incident_id):
    """Get details for a specific incident."""
    incident = get_incident_by_id(incident_id)
    if incident:
        return jsonify(incident)
    else:
        return jsonify({"error": "Incident not found"}), 404

# Resolve incident
@app.route("/api/incidents/<incident_id>/resolve", methods=["POST"])
def api_resolve_incident(incident_id):
    data = request.get_json(silent=True) or {}
    resolution_type = data.get("resolution_type", "resolved")
    
    # Validate resolution type
    if resolution_type not in ["resolved", "not_resolved"]:
        resolution_type = "resolved"

    from backend.services.incident_storage import mark_incident_resolved, get_incident_by_id
    success = mark_incident_resolved(incident_id, resolution_type)
    if success:
        # Emit update
        updated_incident = get_incident_by_id(incident_id)
        if updated_incident:
            emit_incident_update(updated_incident)

        pass
    else:
        return jsonify({"error": "Incident not found"}), 404
    
    return jsonify({"success": True, "id": incident_id, "resolution_type": resolution_type})

# Acknowledge incident
@app.route("/api/incidents/<incident_id>/ack", methods=["POST"])
def api_ack_incident(incident_id):
    import traceback
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get("user_id", "unknown")
        
        from backend.services.incident_storage import acknowledge_incident, get_incident_by_id
        success = acknowledge_incident(incident_id, user_id)
        if success:
            # Emit update
            updated_incident = get_incident_by_id(incident_id)
            if updated_incident:
                emit_incident_update(updated_incident)
            
            # Do NOT reset simulator video cache here.
            pass
        if success:
            return jsonify({"success": True, "id": incident_id, "ack_by": user_id})
        else:
            return jsonify({"error": "Incident not found"}), 404
    except Exception as e:
        print("[ACK-INCIDENT ERROR]", e)
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# Submit Feedback (Confirm/Reject - False Alarm)
@app.route("/api/incidents/<incident_id>/feedback", methods=["POST"])
def api_incident_feedback(incident_id):
    try:
        data = request.get_json(silent=True) or {}
        feedback_type = data.get("feedback_type") # 'confirm' or 'reject'
        
        if feedback_type not in ["confirm", "reject"]:
             return jsonify({"error": "Invalid feedback type"}), 400

        from backend.services.incident_storage import save_incident_feedback, get_incident_by_id
        success = save_incident_feedback(incident_id, feedback_type)
        
        if success:
             # Emit update
            updated_incident = get_incident_by_id(incident_id)
            if updated_incident:
                emit_incident_update(updated_incident)
            return jsonify({"success": True, "id": incident_id, "feedback": feedback_type})
        else:
            return jsonify({"error": "Incident not found"}), 404
            
    except Exception as e:
        print(f"[FEEDBACK ERROR] {e}")
        return jsonify({"error": str(e)}), 500


# Dispatch incident to security
@app.route("/api/incidents/<incident_id>/dispatch", methods=["POST"])
def api_dispatch_incident(incident_id):
    import traceback
    try:
        data = request.get_json(silent=True) or {}
        security_id = data.get("security_id", "unknown")
        
        from backend.services.incident_storage import dispatch_incident, get_incident_by_id
        success = dispatch_incident(incident_id, security_id)
        
        if success:
            # Emit update
            updated_incident = get_incident_by_id(incident_id)
            if updated_incident:
                emit_incident_update(updated_incident)
            return jsonify({"success": True, "id": incident_id, "dispatched_to": security_id})
        else:
            return jsonify({"error": "Incident not found"}), 404
    except Exception as e:
        print("[DISPATCH-INCIDENT ERROR]", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Acknowledge ALL incidents
@app.route("/api/incidents/ack-all", methods=["POST"])
def api_ack_all_incidents():
    import traceback
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get("user_id", "unknown")
        count = ack_all_incidents(user_id)
        # Also reset simulator cache so videos can be re-detected
        simulator = get_simulator()
        if simulator:
            simulator.clear_all_processed_videos()
        
        # Emit cleared event
        socketio.emit('incidents_cleared', {'by': user_id})

        return jsonify({"success": True, "count": count, "ack_by": user_id})
    except Exception as e:
        print("[ACK-ALL ERROR]", e)
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# Dispatch security to incident


# Proxy notifications to incidents feed (unified source)
@app.route("/api/incidents", methods=["GET"])
@app.route("/api/notifications", methods=["GET"])
def api_notifications():
    limit = int(request.args.get("limit", 100))
    return jsonify(get_incidents(limit=limit))

# Security roster
@app.route("/api/security", methods=["GET"])
def api_security_roster():
    return jsonify(list_security_roster())

# Get incident statistics
@app.route("/api/incidents/stats", methods=["GET"])
def api_incident_stats():
    """Get summary statistics for incidents."""
    return jsonify(get_incident_stats())

# Clear all incidents (for testing)
@app.route("/api/incidents/clear", methods=["POST"])
def api_clear_incidents():
    """Clear all stored incidents."""
    clear_incidents()
    # Also reset simulator cache so videos can be re-detected
    simulator = get_simulator()
    if simulator:
        simulator.clear_all_processed_videos()
    
    # Emit cleared event
    socketio.emit('incidents_cleared', {'by': 'system'})
        
    return jsonify({"success": True, "message": "All incidents cleared and detection reset"})

# Save report endpoint
@app.route("/api/reports/save", methods=["POST"])
def save_report():
    """
    Save a report to the backend.
    Expected JSON:
    {
        "name": "December Incident Summary",
        "type": "incidents" | "analytics" | "cameras" | "performance",
        "format": "PDF" | "JSON" | "CSV",
        "data": {...}  // Report data/content
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        report_name = data.get("name", "Untitled Report")
        report_type = data.get("type", "incidents")
        report_format = data.get("format", "JSON")
        report_data = data.get("data", {})
        
        # Generate unique report ID
        report_id = f"RPT-{datetime.now().strftime('%Y-%m-%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"
        
        # Create report directory if it doesn't exist
        reports_dir = PROJECT_ROOT / "backend" / "reports" / report_type
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Create report metadata
        report_metadata = {
            "id": report_id,
            "name": report_name,
            "type": report_type,
            "format": report_format,
            "generated_date": datetime.now().isoformat(),
            "data": report_data
        }
        
        # Save report as JSON file
        report_file = reports_dir / f"{report_id}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_metadata, f, indent=2, ensure_ascii=False)
        
        # Get file size
        file_size = report_file.stat().st_size
        size_kb = file_size / 1024
        
        if size_kb < 1024:
            size_str = f"{size_kb:.1f} KB"
        else:
            size_str = f"{size_kb / 1024:.1f} MB"
        
        return jsonify({
            "success": True,
            "report_id": report_id,
            "file_path": str(report_file.relative_to(PROJECT_ROOT)),
            "size": size_str,
            "message": f"Report saved successfully"
        }), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Get all saved reports
@app.route("/api/reports", methods=["GET"])
def get_reports():
    """Get list of all saved reports with optional type filter."""
    try:
        report_type = request.args.get("type")  # Filter by type if provided
        
        reports_dir = PROJECT_ROOT / "backend" / "reports"
        all_reports = []
        
        # Scan all report type directories
        for type_dir in reports_dir.iterdir():
            if type_dir.is_dir():
                # Skip if filtering and doesn't match
                if report_type and type_dir.name != report_type:
                    continue
                
                # Read all JSON files in this directory
                for report_file in type_dir.glob("*.json"):
                    try:
                        with open(report_file, "r", encoding="utf-8") as f:
                            report_data = json.load(f)
                            
                            # Get file size
                            file_size = report_file.stat().st_size
                            size_kb = file_size / 1024
                            
                            if size_kb < 1024:
                                size_str = f"{size_kb:.1f} KB"
                            else:
                                size_str = f"{size_kb / 1024:.1f} MB"
                            
                            all_reports.append({
                                "id": report_data.get("id"),
                                "name": report_data.get("name"),
                                "type": report_data.get("type"),
                                "format": report_data.get("format"),
                                "generated_date": report_data.get("generated_date"),
                                "size": size_str,
                                "status": "completed"
                            })
                    except Exception as e:
                        print(f"Error reading report {report_file}: {e}")
        
        # Sort by generated date (newest first)
        all_reports.sort(key=lambda x: x.get("generated_date", ""), reverse=True)
        
        return jsonify(all_reports)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Demo Bookings Endpoint ---
@app.route("/api/demo-bookings", methods=["GET"])
def api_get_demo_bookings():
    from backend.services.demo_service import get_demo_requests
    return jsonify(get_demo_requests())

@app.route("/api/demo-bookings", methods=["POST"])
def api_create_demo_booking():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    from backend.services.demo_service import save_demo_request
    new_req = save_demo_request(data)
    
    if new_req:
        return jsonify(new_req), 201
    else:
        return jsonify({"error": "Failed to save request"}), 500

@app.route("/api/demo-bookings/<request_id>/status", methods=["PUT"])
def api_update_demo_status(request_id):
    data = request.get_json()
    status = data.get("status")
    if not status:
        return jsonify({"error": "No status provided"}), 400
        
    from backend.services.demo_service import update_demo_request_status
    success = update_demo_request_status(request_id, status)
    
    if success:
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Request not found or update failed"}), 404


# --- User Feedback and Dataset Management ---
FEEDBACK_LOG_FILE = PROJECT_ROOT / "backend" / "data" / "feedback_log.json"
DATASET_DIR = PROJECT_ROOT / "backend" / "data" / "dataset"
FEEDBACK_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)



@app.route("/api/feedback", methods=["GET"])
def get_feedback_logs():
    """Get all recorded feedback logs."""
    try:
        if FEEDBACK_LOG_FILE.exists():
            with open(FEEDBACK_LOG_FILE, "r") as f:
                data = json.load(f)
            return jsonify(data)
        return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Download a specific report
@app.route("/api/reports/<report_id>", methods=["GET"])
def download_report(report_id):
    """Download a specific report file."""
    try:
        reports_dir = PROJECT_ROOT / "backend" / "reports"
        
        # Search for the report file in all type directories
        for type_dir in reports_dir.iterdir():
            if type_dir.is_dir():
                report_file = type_dir / f"{report_id}.json"
                if report_file.exists():
                    with open(report_file, "r", encoding="utf-8") as f:
                        report_data = json.load(f)
                    return jsonify(report_data)
        
        return jsonify({"error": "Report not found"}), 404
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Global search endpoint

@app.route("/api/search", methods=["GET"])
def global_search():
    """
    Advanced search across incidents, reports, and cameras.
    Query params:
      - q: text query (optional)
      - type: event type (incident, report, camera, violence, car_crash, etc.)
      - camera: camera id (optional)
      - status: incident status (active, resolved, etc.)
      - start: start timestamp (ISO8601, optional)
      - end: end timestamp (ISO8601, optional)
    """
    try:
        query = request.args.get("q", "").lower()
        event_type = request.args.get("type")
        camera_id = request.args.get("camera")
        status = request.args.get("status")
        start = request.args.get("start")
        end = request.args.get("end")
        results = []

        # Helper: parse ISO8601 timestamps
        def parse_time(ts):
            from datetime import datetime
            try:
                return datetime.fromisoformat(ts)
            except Exception:
                return None

        # Search incidents
        incidents = get_incidents()
        for incident in incidents:
            # Filter by type
            if event_type and event_type not in (incident.get("type", "") or incident.get("event", "")):
                continue
            # Filter by camera
            if camera_id and camera_id != incident.get("camera", ""):
                continue
            # Filter by status
            if status and status != incident.get("status", ""):
                continue
            # Filter by time range
            if start or end:
                ts = incident.get("timestamp")
                if ts:
                    t = parse_time(ts) if isinstance(ts, str) else None
                    if start:
                        t0 = parse_time(start)
                        if t and t0 and t < t0:
                            continue
                    if end:
                        t1 = parse_time(end)
                        if t and t1 and t > t1:
                            continue
            # Filter by query
            if query and not (
                query in (incident.get("type", "").lower()) or
                query in (incident.get("location", "").lower()) or
                query in (incident.get("camera", "").lower())
            ):
                continue
            results.append({
                "id": incident.get("id", ""),
                "type": "incident",
                "title": incident.get("type", "Unknown Incident"),
                "subtitle": f"{incident.get('location', 'Unknown')} • {incident.get('camera', 'Unknown')}",
                "timestamp": incident.get("timestamp", ""),
                "severity": incident.get("severity", "")
            })

        # Search reports
        reports_dir = PROJECT_ROOT / "backend" / "reports"
        if reports_dir.exists():
            for report_type_dir in reports_dir.iterdir():
                if report_type_dir.is_dir():
                    for report_file in report_type_dir.glob("*.json"):
                        try:
                            with open(report_file, "r", encoding="utf-8") as f:
                                report = json.load(f)
                            if event_type and event_type != report.get("type", ""):
                                continue
                            if query and not (
                                query in report.get("name", "").lower() or
                                query in report.get("type", "").lower()
                            ):
                                continue
                            results.append({
                                "id": report.get("id", ""),
                                "type": "report",
                                "title": report.get("name", "Unknown Report"),
                                "subtitle": f"{report.get('type', 'Unknown').title()} Report • Generated by {report.get('generated_by', 'Unknown')}",
                                "timestamp": report.get("generated_at", "")
                            })
                        except:
                            pass

        # Search cameras
        camera_states = get_all_camera_states()
        for camera in camera_states:
            cam_id = camera.get("camera_id", "")
            if camera_id and camera_id != cam_id:
                continue
            if event_type and event_type != camera.get("event", ""):
                continue
            if query and query not in cam_id.lower():
                continue
            results.append({
                "id": cam_id,
                "type": "camera",
                "title": cam_id,
                "subtitle": f"Status: {camera.get('event', 'normal').title()} • {camera.get('last_update', '')}",
                "status": camera.get("event", "normal")
            })

        # Limit results to 50
        results = results[:50]
        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e), "results": []}), 500


# --- WebSocket (SocketIO) Real-Time Events ---
@socketio.on('connect')
def handle_connect():
    print(f"[WebSocket] Client connected: {request.sid}")
    emit('connected', {'message': 'WebSocket connection established'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"[WebSocket] Client disconnected: {request.sid}")

# Demo: emit incident update
def emit_incident_update(incident):
    socketio.emit('incident_update', incident)

# Demo: emit camera status update
def emit_camera_update(camera_state):
    socketio.emit('camera_update', camera_state)

# Example: Hook into incident creation (add_incident) and camera state update to emit events
# (In production, call emit_incident_update and emit_camera_update in the relevant service logic)

if __name__ == "__main__":
    print("🚀 Starting VIGIL Backend on http://127.0.0.1:5000")
    print("📹 Loading video dataset...")
    print("🎥 Initializing cameras...")
    print("🎬 Starting live camera simulator...")
    start_simulator()
    print("✅ System ready - cameras are now 'live'")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=True)

