/// <reference types="vite/client" />
import { useState, useEffect, useCallback, useRef } from 'react';
import { toast } from 'sonner';
import { useIncidentWebSocket } from './useIncidentWebSocket';
import { Siren, AlertTriangle, ShieldAlert, X, Users } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || '';

export interface Incident {
  id: string; // Changed to string as backend usually returns string or we convert
  type: 'violence' | 'crash' | string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  location: string;
  cameraId: string; // or camera
  timestamp: string;
  status: 'active' | 'resolved' | 'dismissed' | 'pending_review' | 'confirmed' | 'dispatched' | 'acknowledged';
  confidence: number;
  thumbnail?: string;
  description?: string;
  people_count?: number;
  assigned_security?: string;
  acknowledged?: boolean;
  dispatched_to?: string[];
  requiresReview?: boolean;
  aiFeedback?: boolean;
  dispatchedOfficerName?: string;
  fieldResponse?: string;
  ack_by?: string;
  resolution_type?: 'resolved' | 'not_resolved'; // Added resolution type
}


export function useRealtimeIncidents(
  onIncidentReceived?: (id: string) => void,
  offlineMode: boolean = false,
  currentUserName?: string
) {
  const [incidents, setIncidents] = useState<any[]>([]);
  const [soundEnabled, setSoundEnabled] = useState(true);



  const alertedIncidentsRef = useRef(new Set<string>());

  // Audio Logic moved up
  const playAlertSound = useCallback(() => {
    if (!soundEnabled) return;
    try {
      // Use the real file in public folder
      const audio = new Audio('/alert.mp3');
      audio.volume = 0.5;
      const playPromise = audio.play();

      if (playPromise !== undefined) {
        playPromise.catch(e => {
          console.warn("Audio play blocked by browser policy", e);
        });
      }
    } catch (e) {
      console.error("Audio error", e);
    }
  }, [soundEnabled]);



  // Unified Alert Logic
  const processNewIncident = useCallback((incident: Incident) => {
    if (alertedIncidentsRef.current.has(incident.id)) return false;

    // Mark as alerted
    alertedIncidentsRef.current.add(incident.id);

    // Filter out user's own actions if needed (optional based on requirements)
    // For now, we alert on EVERYTHING new to the system for visibility

    // Play Sound
    playAlertSound();

    // Show Banner with Custom Icon and Styling
    const isViolence = incident.type === 'violence';
    const isCrash = incident.type === 'crash';

    // Choose Icon
    const IconComponent = isViolence ? Siren : (isCrash ? AlertTriangle : ShieldAlert);
    const iconColor = isViolence ? 'text-red-600' : 'text-orange-500';

    toast.custom((t) => (
      <div className="w-96 md:w-[400px] relative overflow-hidden rounded-lg bg-white dark:bg-slate-900 border-l-4 border-l-red-500 shadow-xl p-4 flex gap-4 ring-1 ring-black/5 pointer-events-auto transition-all animate-in slide-in-from-top-2 duration-300">
        {/* Icon */}
        <div className="shrink-0 pt-0.5">
          <IconComponent className={`w-6 h-6 ${iconColor} animate-pulse`} />
        </div>

        {/* Content */}
        <div className="flex-1 w-0 flex flex-col">
          <p className="font-extrabold text-red-700 dark:text-red-400 text-base flex items-center gap-2 mb-0.5">
            NEW INCIDENT: {(incident.type || 'Unknown').toUpperCase()}
          </p>
          <p className="text-sm font-semibold text-gray-800 dark:text-gray-200 font-mono mb-1">
            {incident.location}
          </p>
          {incident.description && (
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-2 line-clamp-2">{incident.description}</p>
          )}

          {isViolence && incident.people_count !== undefined && (
            <p className="text-xs font-bold mb-3 text-red-600 dark:text-red-300 bg-red-100 dark:bg-red-900/30 px-2 py-0.5 rounded-full w-fit">
              {incident.people_count} {incident.people_count === 1 ? 'Person' : 'People'} Detected
            </p>
          )}

          {/* Action Button */}
          <div className="flex justify-end mt-2">
            <button
              onClick={(e) => {
                e.stopPropagation();
                toast.dismiss(t);
                if (onIncidentReceived) onIncidentReceived(incident.id);
              }}
              className="bg-red-600 hover:bg-red-700 text-white text-xs font-bold px-4 py-1.5 rounded transition-colors shadow-sm"
            >
              View Incident
            </button>
          </div>
        </div>

        {/* Close Button */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            toast.dismiss(t);
          }}
          className="absolute top-2 right-2 p-1.5 text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    ), { duration: 8000 });

    return true;
  }, [playAlertSound, onIncidentReceived]);

  useEffect(() => {
    if (offlineMode) return;

    const fetchIncidents = async () => {
      try {
        const res = await fetch(`${API_URL}/api/incidents`);
        if (res.ok) {
          const data = await res.json();
          if (Array.isArray(data)) {
            // Check for new incidents from polling
            data.forEach((incident: any) => {
              // If incident is "active" or "dispatched" (new-ish) and we haven't alerted, do it.
              // We only alert if it's not resolved/dismissed to avoid spamming on reload
              if (incident.status !== 'resolved' && incident.status !== 'dismissed') {
                processNewIncident(incident);
              }
            });

            setIncidents(prev => {
              const newIds = new Set(data.map((d: any) => d.id));
              const filteredPrev = prev.filter(i => newIds.has(i.id));

              if (JSON.stringify(prev) === JSON.stringify(data)) return prev;
              return data;
            });
          }
        }
      } catch (e) {
        console.error("Failed to fetch initial incidents", e);
      }
    };

    fetchIncidents();
    // Poll every 6 seconds (optimized from 4s) to clean up ghosts
    const interval = setInterval(fetchIncidents, 6000);
    return () => clearInterval(interval);
  }, [offlineMode, processNewIncident]); // Added processNewIncident dep

  // Socket update
  // Audio Logic
  // Socket update
  // Audio logic moved up

  // Socket update handler
  const handleSocketUpdate = useCallback((updatedIncident: any) => {
    if (offlineMode) return;

    // Attempt alert (will skip if already alerted by polling)
    processNewIncident(updatedIncident);

    setIncidents(prev => {
      const exists = prev.find(i => i.id === updatedIncident.id);
      if (exists) {
        return prev.map(i => i.id === updatedIncident.id ? updatedIncident : i);
      }
      return [updatedIncident, ...prev];
    });

    if (onIncidentReceived && !incidentsRef.current.find(i => i.id === updatedIncident.id)) {
      onIncidentReceived(updatedIncident.id);
    }

  }, [offlineMode, onIncidentReceived, processNewIncident]);

  const handleSocketClear = useCallback(() => {
    if (offlineMode) return;
    setIncidents([]);
  }, [offlineMode]);

  // Socket update
  useIncidentWebSocket(handleSocketUpdate, handleSocketClear);

  // Optionally, you can implement resolve/dismiss by calling backend endpoints here

  const resolveIncident = useCallback(async (id: string, resolutionType: string = 'resolved') => {
    try {
      const res = await fetch(`${API_URL}/api/incidents/${id}/resolve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resolution_type: resolutionType })
      });

      if (res.status === 404) {
        setIncidents(prev => prev.filter(inc => inc.id !== id));
        toast.error('Incident synced (removed ghost item)');
        return;
      }

      toast.success(`Incident marked as ${resolutionType.replace('_', ' ')}`, {
        duration: 3000,
        position: 'top-right',
      });

      // Optimistic update
      // If 'not_resolved', keep it confirmed (active) but tag it
      const newStatus = resolutionType === 'not_resolved' ? 'confirmed' : 'resolved';

      setIncidents(prev => prev.map(inc => inc.id === id ? {
        ...inc,
        status: newStatus as any,
        resolution_type: resolutionType as any
      } : inc));
    } catch (e) {
      console.error("Failed to resolve incident:", e);
      toast.error("Failed to resolve incident");
    }
  }, []);

  const dismissIncident = useCallback(async (id: string) => {
    try {
      const res = await fetch(`${API_URL}/api/incidents/${id}/resolve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resolution_type: 'dismissed' })
      });

      if (res.status === 404) {
        setIncidents(prev => prev.filter(inc => inc.id !== id));
        toast.error('Incident synced (removed ghost item)');
        return;
      }

      toast.info('Incident Dismissed', {
        duration: 3000,
        position: 'top-right',
      });
      // Optimistic update - set to resolved so it filters out of active views
      setIncidents(prev => prev.map(inc => inc.id === id ? { ...inc, status: 'resolved' as any } : inc));
    } catch (e) {
      console.error("Failed to dismiss incident:", e);
      toast.error("Failed to dismiss incident");
    }
  }, []);


  const toggleSound = useCallback(() => {
    setSoundEnabled((prev) => !prev);
    toast.info(soundEnabled ? 'Alert sounds disabled' : 'Alert sounds enabled', {
      duration: 2000,
    });
  }, [soundEnabled]);

  const dismissAllIncidents = useCallback(async () => {
    try {
      await fetch(`${API_URL}/api/incidents/ack-all`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: 'current-user' })
      });
      toast.info('All incidents dismissed');
      setIncidents(prev => prev.map(inc => inc.status === 'active' ? { ...inc, status: 'acknowledged' as any } : inc));
    } catch (e) {
      console.error("Failed to dismiss all incidents:", e);
      toast.error("Failed to dismiss all");
    }
  }, []);

  const confirmIncident = useCallback(async (id: string) => {
    try {
      const res = await fetch(`${API_URL}/api/incidents/${id}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ feedback_type: 'confirm' })
      });

      if (res.status === 404) {
        setIncidents(prev => prev.filter(inc => inc.id !== id));
        return;
      }

      toast.success('Incident verified as True Positive', { duration: 2000 });
      // Optimistic update
      setIncidents(prev => prev.map(inc => inc.id === id ? { ...inc, status: 'confirmed' as any } : inc));
    } catch (e) {
      console.error("Failed to confirm incident:", e);
      toast.error("Failed to submit feedback");
    }
  }, []);

  const rejectIncident = useCallback(async (id: string) => {
    try {
      const res = await fetch(`${API_URL}/api/incidents/${id}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ feedback_type: 'reject' })
      });

      if (res.status === 404) {
        setIncidents(prev => prev.filter(inc => inc.id !== id));
        return;
      }

      toast.success('Incident marked as False Positive', { duration: 2000 });
      // Implicitly remove it from active list or mark as rejected
      setIncidents(prev => prev.filter(inc => inc.id !== id));
    } catch (e) {
      console.error("Failed to reject incident:", e);
      toast.error("Failed to submit feedback");
    }
  }, []);


  const dispatchIncident = useCallback(async (id: string, securityId: string) => {
    try {
      const res = await fetch(`${API_URL}/api/incidents/${id}/dispatch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ security_id: securityId })
      });

      if (res.status === 404) {
        toast.error('Incident no longer exists (removing from view)');
        setIncidents(prev => prev.filter(inc => inc.id !== id));
        return;
      }

      if (!res.ok) {
        throw new Error(`Error ${res.status}: ${res.statusText}`);
      }

      toast.success('Incident dispatched successfully', { duration: 2000 });
      setIncidents(prev => prev.map(inc =>
        inc.id === id ? { ...inc, status: 'dispatched' as any, dispatchedOfficerName: securityId } : inc
      ));
    } catch (e: any) {
      console.error("Failed to dispatch incident:", e);
      toast.error(`Failed to dispatch: ${e.message}`);
    }
  }, []);

  const activeIncidents = incidents.filter((i) => i.status === 'active' || i.status === 'dispatched' || i.status === 'acknowledged');

  // Use Ref to access latest incidents in simulation interval without resetting it
  const incidentsRef = useRef(incidents);
  useEffect(() => {
    incidentsRef.current = incidents;
  }, [incidents]);

  // Simulation Loop for "Alive" Dashboard
  useEffect(() => {
    if (offlineMode || !currentUserName) return;

    const simulationInterval = setInterval(() => {
      const currentIncidents = incidentsRef.current;

      console.log("Simulation Tick. Active:", currentIncidents.length);

      // 1. Simulate "Verification" (Dispatched -> Confirmed)
      const dispatchedToOthers = currentIncidents.filter(inc =>
        inc.status === 'dispatched' &&
        inc.dispatchedOfficerName &&
        inc.dispatchedOfficerName !== currentUserName
      );

      console.log("Simulation Candidates (Dispatch->Confirm):", dispatchedToOthers.length, dispatchedToOthers.map(i => i.dispatchedOfficerName));

      if (dispatchedToOthers.length > 0) {
        if (Math.random() > 0.2) { // 80% chance (increased from 60%)
          const target = dispatchedToOthers[Math.floor(Math.random() * dispatchedToOthers.length)];
          console.log(`Simulating Confirm for incident ${target.id} assigned to ${target.dispatchedOfficerName}`);
          confirmIncident(target.id);
        }
      }

      // 2. Simulate "Resolution" (Confirmed -> Resolved)
      const confirmedByOthers = currentIncidents.filter(inc =>
        inc.status === 'confirmed' &&
        inc.dispatchedOfficerName &&
        inc.dispatchedOfficerName !== currentUserName
      );

      console.log("Simulation Candidates (Confirm->Resolve):", confirmedByOthers.length);

      if (confirmedByOthers.length > 0) {
        if (Math.random() > 0.2) { // 80% chance (increased from 60%)
          const target = confirmedByOthers[Math.floor(Math.random() * confirmedByOthers.length)];
          console.log(`Simulating Resolve for incident ${target.id}`);
          resolveIncident(target.id, 'resolved');
        }
      }

    }, 5000); // Check every 5 seconds (optimized)

    return () => clearInterval(simulationInterval);
  }, [offlineMode, currentUserName, confirmIncident, resolveIncident]);

  return {
    incidents,
    activeIncidents,
    resolveIncident,
    dismissIncident,
    dispatchIncident,
    soundEnabled,
    toggleSound,
    dismissAllIncidents,
    confirmIncident,
    rejectIncident,
  };
}
