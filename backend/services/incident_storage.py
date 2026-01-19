"""
Incident Storage Service

Stores detected incidents (violence/crash) with video clips for frontend viewing.
Maintains a history of detections that can be queried via API.
"""

import time
import threading
import shutil
import os
from typing import List, Dict, Optional
from pathlib import Path

# In-memory incident storage (thread-safe)
_incidents: List[Dict] = []
_incident_id_counter = 1
_lock = threading.Lock()
_max_incidents = 200  # Keep last 200 incidents
_update_window_seconds = 40  # merge duplicate events per camera/type within this window

# Simple registry of security personnel for dispatch (demo-only)
_security_roster = [
    {"id": "SEC-101", "name": "Officer Malik", "status": "available"},
    {"id": "SEC-102", "name": "Officer Chen", "status": "available"},
    {"id": "SEC-103", "name": "Officer Rivera", "status": "available"},
    {"id": "SEC-104", "name": "Chief Martinez", "status": "available"},
    {"id": "SEC-105", "name": "Officer Smith", "status": "available"},
]


def add_incident(camera_id: str, event_type: str, confidence: float, video_path: str, model: str, extra: dict = None) -> Dict:
    """
    Store a new incident detection.
    
    Args:
        camera_id: Camera identifier (e.g., "CAM-01")
        event_type: Type of incident ("violence" or "traffic")
        confidence: Detection confidence (0.0 - 1.0)
        video_path: Relative path to video file
        model: Model name used for detection
    
    Returns:
        Created incident dict
    """
    global _incident_id_counter
    
    # Normalize event_type for crash
    if event_type == "traffic":
        event_type = "crash"
    # Determine severity based on confidence
    severity = (
        "critical" if confidence > 0.9 else
        "high" if confidence > 0.75 else
        "medium" if confidence > 0.6 else
        "low"
    )
    
    now_ts = time.time()

    with _lock:
        # Check for recent incident for same camera/type to reduce spam
        for existing in _incidents:
            # Condition 1: Same video file (Strict Dedup for Simulator)
            # If it's the exact same video file, we assume it's the same event, regardless of time.
            same_video = existing.get("video_url") == video_path or existing.get("videoUrl") == video_path
            
            # Condition 2: Same camera/type within time window (Standard Dedup)
            in_time_window = (now_ts - existing.get("timestamp", 0) <= _update_window_seconds)
            
            if (existing.get("cameraId") == camera_id and existing.get("type") == event_type and (same_video or in_time_window)):
                update_data = {
                    "timestamp": now_ts,
                    "timestamp_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts)),
                    "confidence": round(confidence * 100, 1),
                    "severity": severity,
                    "description": _get_description(event_type, confidence),
                    "video_url": video_path,
                    "videoUrl": video_path,
                    # Keep existing status so we don't "revive" a resolved/acked incident
                    "status": existing.get("status", "active"),
                    "model": model,
                }
                if extra:
                     update_data.update(extra)
                existing.update(update_data)
                print(f"📝 Updated incident (merged): {existing['id']} - {event_type} (Status: {existing.get('status')})")
                
                # Emit update for merged incident so frontend sees confidence/severity changes
                try:
                    from backend.app import emit_incident_update
                    emit_incident_update(existing)
                except Exception as e:
                    print(f"[WARN] Could not emit merged incident update: {e}")
                    
                return existing

        # Create incident record
        incident = {
            "id": f"INC-{int(now_ts)}-{camera_id}",
            "incident_number": _incident_id_counter,
            "type": event_type,  # "violence", "traffic", or "people_count"
            "severity": severity,
            "location": f"Camera {camera_id}",  # TODO: Map camera to real location
            "cameraId": camera_id,
            "timestamp": now_ts,
            "timestamp_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts)),
            "status": "active",  # active | acknowledged | dispatched | resolved
            "acknowledged": False,
            "ack_by": None,
            "dispatched_to": [],  # list of security IDs
            "assigned_security": None,  # primary dispatch
            "description": _get_description(event_type, confidence),
            "confidence": round(confidence * 100, 1),  # percentage
            "video_url": video_path,  # relative path (snake_case)
            "videoUrl": video_path,   # legacy camelCase for older UI paths
            "model": model,
        }
        if extra:
            incident.update(extra)
        
        _incidents.insert(0, incident)  # Add to front (newest first)
        _incident_id_counter += 1
        
        # Keep only last N incidents to prevent memory bloat
        if len(_incidents) > _max_incidents:
            _incidents.pop()
    
    print(f"📝 Stored incident: {incident['id']} - {event_type} @ {confidence:.2f}")
    # Emit websocket notification if available
    try:
        from backend.app import emit_incident_update
        emit_incident_update(incident)
    except Exception as e:
        print(f"[WARN] Could not emit incident update: {e}")
    return incident


def get_incidents(limit: int = 50, event_type: Optional[str] = None, status: Optional[str] = None) -> List[Dict]:
    """
    Retrieve stored incidents with optional filtering.
    
    Args:
        limit: Max number of incidents to return
        event_type: Filter by type ("violence" or "traffic"), None for all
        status: Filter by status ("active" or "resolved"), None for all
    
    Returns:
        List of incident dicts (newest first)
    """
    with _lock:
        filtered = _incidents
        
        if event_type:
            filtered = [inc for inc in filtered if inc["type"] == event_type]
        
        if status:
            filtered = [inc for inc in filtered if inc["status"] == status]
        
        return filtered[:limit]


def get_incident_by_id(incident_id: str) -> Optional[Dict]:
    """Get a specific incident by ID."""
    with _lock:
        for incident in _incidents:
            if incident["id"] == incident_id:
                return incident
    return None


def mark_incident_resolved(incident_id: str, resolution_type: str = "resolved") -> bool:
    """
    Mark an incident as resolved with a specific resolution type.
    resolution_type: 'resolved' (True Positive) or 'not_resolved' (False Positive / Unverified)
    """
    with _lock:
        for incident in _incidents:
            if incident["id"] == incident_id:
                incident["status"] = "resolved"
                incident["resolution_type"] = resolution_type
                
                # If it was assigned to someone, free them up (conceptually)
                # In a real DB, we'd update the user's status key.
                
                print(f"✅ Incident {incident_id} marked {resolution_type}")
                return True
    return False


def acknowledge_incident(incident_id: str, user_id: str) -> bool:
    """Mark an incident as acknowledged by a user."""
    with _lock:
        for incident in _incidents:
            if incident["id"] == incident_id:
                incident["acknowledged"] = True
                incident["ack_by"] = user_id
                if incident["status"] == "active":
                    incident["status"] = "acknowledged"
                print(f"🤝 Incident {incident_id} acknowledged by {user_id}")
                return True
    return False


def ack_all_incidents(user_id: str) -> int:
    """
    HARD RESET: Delete all incidents from memory.
    Originally 'acknowledge all', now repurposed to 'Dismiss All & Reset' per user request.
    """
    global _incidents
    count = 0
    with _lock:
        count = len(_incidents)
        # Clear the list explicitly here or call clear_incidents
        # Calling clear_incidents() would require releasing lock if it also takes lock,
        # but clear_incidents has its own lock.
        # So we should call clear_incidents() OUTSIDE the lock or just inline the clear logic to be safe/atomic.
        # Let's inline the clearing to match clear_incidents logic but inside this function's scope if needed,
        # OR just call clear_incidents() since it handles the lock.
        pass
    
    # Safe to call clear_incidents which handles its own lock
    clear_incidents()
    
    if count > 0:
        print(f"🗑️  Reset all {count} incidents triggered by {user_id}")
    return count


def dispatch_incident(incident_id: str, security_id: str, assigned: bool = True) -> bool:
    """Dispatch a security officer to an incident."""
    with _lock:
        for incident in _incidents:
            if incident["id"] == incident_id:
                if security_id not in incident["dispatched_to"]:
                    incident["dispatched_to"].append(security_id)
                if assigned:
                    incident["assigned_security"] = security_id
                incident["status"] = "dispatched"
                print(f"🚓 Incident {incident_id} dispatched to {security_id}")
                return True
    return False


def list_security_roster() -> List[Dict]:
    """
    Return security roster with dynamic status based on active assignments.
    """
    # Create a copy of the roster to avoid mutating the original template if we were caching it
    roster_snapshot = [officer.copy() for officer in _security_roster]
    
    with _lock:
        # Find all currently assigned security IDs
        busy_officers = set()
        for inc in _incidents:
            if inc["status"] in ["dispatched", "acknowledged"] and inc.get("assigned_security"):
                 busy_officers.add(inc["assigned_security"])
    
    for officer in roster_snapshot:
        if officer["id"] in busy_officers:
            officer["status"] = "Busy"
        else:
            officer["status"] = "Available"
            
    return roster_snapshot


def clear_incidents():
    """Clear all stored incidents (for testing/reset)."""
    global _incident_id_counter
    with _lock:
        _incidents.clear()
        _incident_id_counter = 1
    print("🗑️  Cleared all incidents")


def _get_description(event_type: str, confidence: float) -> str:
    """Generate human-readable description for incident."""
    if event_type == "violence":
        if confidence > 0.9:
            return "High-confidence violent activity detected"
        elif confidence > 0.75:
            return "Physical altercation detected"
        elif confidence > 0.6:
            return "Aggressive behavior detected"
        else:
            return "Suspicious activity detected"
    elif event_type == "crash":
        if confidence > 0.9:
            return "Severe vehicle collision detected"
        elif confidence > 0.75:
            return "Vehicle collision detected"
        elif confidence > 0.6:
            return "Traffic accident detected"
        else:
            return "Possible traffic incident"
    else:
        return "Incident detected"


def get_incident_stats() -> Dict:
    """Get summary statistics for stored incidents."""
    with _lock:
        total = len(_incidents)
        violence = sum(1 for inc in _incidents if inc["type"] == "violence")
        traffic = sum(1 for inc in _incidents if inc["type"] == "traffic")
        active = sum(1 for inc in _incidents if inc["status"] == "active")
        resolved = sum(1 for inc in _incidents if inc["status"] == "resolved")
        
        return {
            "total": total,
            "violence": violence,
            "traffic": traffic,
            "active": active,
            "resolved": resolved,
        }


def save_incident_feedback(incident_id: str, feedback_type: str) -> bool:
    """
    Process user feedback (confirm/reject) and save data for retraining.
    """
    with _lock:
        incident = None
        for inc in _incidents:
            if inc["id"] == incident_id:
                incident = inc
                break
        
        if not incident:
            return False
            
        # Update status
        incident["aiFeedback"] = True
        incident["feedbackType"] = feedback_type
        
        if feedback_type == "reject":
            incident["status"] = "resolved"
            incident["resolution_type"] = "false_positive"
            print(f"❌ Incident {incident_id} rejected (False Alarm)")
        elif feedback_type == "confirm":
             incident["status"] = "confirmed"
             print(f"✅ Incident {incident_id} confirmed (True Positive)")

        # Save for retraining
        try:
            _copy_for_retraining(incident, feedback_type)
        except Exception as e:
            print(f"[RETRAIN ERROR] Failed to save data: {e}")
            
        return True


def _copy_for_retraining(incident: Dict, feedback_type: str):
    """Copy video clip to retraining dataset."""
    try:
        video_path = incident.get("video_url") # relative path e.g. "Videos/..."
        if not video_path:
            return
    
        # Determine Source - handle potential path variations
        # Assuming current working directory is project root
        project_root = Path.cwd()
        source_path = (project_root / video_path).resolve()
            
        if not source_path.exists():
            # Try plain "Videos" prefix if path was stored without it or differently
            if "Videos" not in str(video_path):
                 source_path = (project_root / "Videos" / video_path).resolve()
        
        if not source_path.exists():
            print(f"[RETRAIN INFO] Video source not found: {source_path}")
            return
    
        # Determine Dest
        # backend/data/retraining/false_positives OR true_positives
        category = "false_positives" if feedback_type == "reject" else "true_positives"
        dest_dir = project_root / "backend" / "data" / "retraining" / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Filename: [label]_[timestamp]_[orig_name]
        label = incident.get("type", "unknown")
        timestamp = int(incident.get("timestamp", 0))
        dest_filename = f"{label}_{timestamp}_{source_path.name}"
        dest_path = dest_dir / dest_filename
        
        shutil.copy2(source_path, dest_path)
        print(f"💾 Saved for retraining: {dest_path}")
    except Exception as e:
        print(f"[RETRAIN INTERNAL ERROR] {e}")


