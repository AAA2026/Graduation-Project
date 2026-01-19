import { useState, useEffect, useRef } from 'react';


export interface CameraStatus {
  location: string;
  camera_id: string;
  status: string;
  video?: string;
  event?: string;
  confidence?: number;
  last_update?: number;
}

export interface LiveStatusResponse {
  cameras: CameraStatus[];
  systemStatus: string;
  timestamp: string;
}


const API_URL = import.meta.env.VITE_API_URL || '';

export function useLiveStatus() {
  const [cameraStatuses, setCameraStatuses] = useState<Map<string, CameraStatus>>(new Map());
  const [systemStatus, setSystemStatus] = useState<string>('offline');
  const [isPolling, setIsPolling] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<string>(new Date().toISOString());
  const [offlineMode, setOfflineMode] = useState<boolean>(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Fetch live status from backend
  const fetchLiveStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/api/live-status`);
      if (!response.ok) throw new Error('Failed to fetch');
      const data = await response.json();
      // Accept both {cameras: [...]} and plain array
      const cameras = Array.isArray(data) ? data : data.cameras;
      if (!Array.isArray(cameras)) throw new Error('Invalid cameras data');
      // Deep compare to avoid re-renders if data matches
      const newMap = new Map();
      cameras.forEach((camera: CameraStatus) => {
        newMap.set(camera.camera_id, camera);
      });

      setCameraStatuses(prev => {
        // Simple size check first
        if (prev.size !== newMap.size) return newMap;

        // Check content
        let changed = false;
        for (const [key, val] of newMap.entries()) {
          const prevVal = prev.get(key);
          if (JSON.stringify(prevVal) !== JSON.stringify(val)) {
            changed = true;
            break;
          }
        }
        return changed ? newMap : prev;
      });

      setSystemStatus('operational');
      setLastUpdate(new Date().toISOString());
    } catch (error) {
      setSystemStatus('offline');
      setCameraStatuses(new Map());
    }
  };

  // Fetch offline mode from backend
  const fetchOfflineMode = async () => {
    try {
      const response = await fetch(`${API_URL}/api/offline-mode`);
      if (!response.ok) throw new Error('Failed to fetch offline mode');
      const data = await response.json();
      setOfflineMode(data.offline_mode);
    } catch {
      setOfflineMode(false);
    }
  };

  // Toggle offline mode on backend
  const toggleOfflineMode = async () => {
    try {
      const response = await fetch(`${API_URL}/api/offline-mode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ offline_mode: !offlineMode })
      });
      if (!response.ok) throw new Error('Failed to toggle offline mode');
      const data = await response.json();
      setOfflineMode(data.offline_mode);
      // Force refresh after toggling so header updates instantly
      await fetchOfflineMode();
      await fetchLiveStatus();
    } catch (error) {
      throw error;
    }
  };

  useEffect(() => {
    if (!isPolling) return;
    fetchLiveStatus();
    fetchOfflineMode();
    // Poll every 6 seconds (optimized from 3s)
    intervalRef.current = setInterval(() => {
      fetchLiveStatus();
      fetchOfflineMode();
    }, 6000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPolling]);

  const getCameraStatus = (cameraId: string): CameraStatus | undefined => {
    return cameraStatuses.get(cameraId);
  };

  const togglePolling = () => {
    setIsPolling((prev) => !prev);
  };

  return {
    cameraStatuses,
    systemStatus,
    lastUpdate,
    isPolling,
    getCameraStatus,
    togglePolling,
    refreshStatus: fetchLiveStatus,
    offlineMode,
    toggleOfflineMode,
  };
}
