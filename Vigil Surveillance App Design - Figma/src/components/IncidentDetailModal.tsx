import { useState, useEffect } from 'react';
import { toast } from 'sonner';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogDescription } from './ui/dialog';
import { VideoModal } from './VideoModal';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Textarea } from './ui/textarea';
import { vigilClasses } from './vigil-theme';
import {
  AlertCircle,
  Car,
  Users,
  MapPin,
  Clock,
  Eye,
  Camera,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Volume2,
  VolumeX,
  Download,
  Maximize2,
  CheckCircle,
  XCircle,
  User,
  FileText,
  Image as ImageIcon,
  Shield
} from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { DialogFooter, DialogDescription } from './ui/dialog';
import { SECURITY_PERSONNEL } from '../data/roster';
import { motion } from 'motion/react';

import type { Incident } from '../hooks/useRealtimeIncidents';

interface IncidentDetailModalProps {
  incident: Incident | null;
  isOpen: boolean;
  onClose: () => void;
  onDismiss?: (id: string) => void;
  onConfirm?: (id: string) => void;
  onReject?: (id: string) => void;
  onDispatch?: (id: string, userId: string) => void;
  // Update onResolve signature to allow type
  onResolve?: (id: string, type?: 'resolved' | 'not_resolved') => void;
  userRole?: string;
  userName?: string;
}

export function IncidentDetailModal({
  incident,
  isOpen,
  onClose,
  onResolve,
  onDismiss,
  onConfirm,
  onReject,
  onDispatch,
  userRole = 'admin',
  userName = 'Security',
}: IncidentDetailModalProps) {
  console.log("IncidentDetailModal userRole:", userRole);
  const [isPlaying, setIsPlaying] = useState(true);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0); // seconds
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [dispatchDialogOpen, setDispatchDialogOpen] = useState(false);
  const [selectedOfficer, setSelectedOfficer] = useState('');

  const handleConfirmDispatch = () => {
    if (selectedOfficer && onDispatch) {
      onDispatch(incident.id, selectedOfficer);
      setDispatchDialogOpen(false);
      onClose();
      toast.success(`Dispatched to ${SECURITY_PERSONNEL.find(p => p.id === selectedOfficer)?.name || 'Officer'}`);
    }
  };
  const [notes, setNotes] = useState('');
  // Use actual video duration if available, fallback to 120s
  const totalDuration = 120;

  // Get video URL from incident (backend provides videoPath)
  const API_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:5000';

  // Debug incident object
  console.log("Incident details:", incident);

  // Accept all possible backend video keys: video_url, videoUrl, videoPath, video
  const videoPath = incident && (
    (incident as any).video_url ||
    (incident as any).videoUrl ||
    (incident as any).videoPath ||
    (incident as any).video ||
    ''
  );

  // Normalize path: Replace backslashes with forward slashes for URL compatibility
  let normalizedPath = typeof videoPath === 'string' ? videoPath.replace(/\\/g, '/') : '';

  // Ensure path starts with /videos/ if it's not already there
  if (normalizedPath && !normalizedPath.startsWith('/videos/') && !normalizedPath.startsWith('videos/')) {
    normalizedPath = `videos/${normalizedPath}`;
  }
  // Ensure leading slash for URL construction
  if (normalizedPath && !normalizedPath.startsWith('/')) {
    normalizedPath = `/${normalizedPath}`;
  }

  const videoUrl = normalizedPath ? `${API_URL.replace(/\/$/, '')}${normalizedPath}` : '';
  console.log("Resolved Video URL:", videoUrl);

  if (!incident) return null;

  const TypeIcon = incident.type === 'violence' ? Users : Car;

  const getSeverityColor = (severity: string) => {
    if (severity === 'critical') return 'bg-red-600 text-white';
    if (severity === 'high') return 'bg-orange-600 text-white';
    if (severity === 'medium') return 'bg-amber-600 text-white';
    return 'bg-blue-600 text-white';
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleSkipBack = () => {
    setCurrentTime(Math.max(0, currentTime - 10));
  };

  const handleSkipForward = () => {
    setCurrentTime(Math.min(totalDuration, currentTime + 10));
  };

  const nearbyFootage = [
    { cameraId: 'CAM-043', location: 'Main St & 6th Ave', distance: '50m' },
    { cameraId: 'CAM-041', location: 'Main St & 4th Ave', distance: '120m' },
    { cameraId: 'CAM-090', location: '5th Ave North', distance: '200m' },
  ];

  const activityLog = [
    { time: '2 mins ago', user: 'System', action: 'Incident detected by AI', type: 'system' },
    { time: '2 mins ago', user: 'System', action: 'Alert sent to active officers', type: 'system' },
    { time: '1 min ago', user: 'Officer Smith', action: 'Viewed incident details', type: 'user' },
    { time: '30 secs ago', user: 'Officer Smith', action: 'Started video review', type: 'user' },
  ];

  // Reset playback when modal opens/closes
  useEffect(() => {
    if (isOpen) {
      setIsPlaying(true);
      setCurrentTime(0);
    } else {
      setIsPlaying(false);
    }
  }, [isOpen]);

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent
        className={`w-full max-w-2xl md:max-w-3xl lg:max-w-4xl h-auto max-h-[92vh] overflow-y-auto ${vigilClasses.card} p-4 md:p-6 border-2 border-cyan-500 shadow-2xl bg-white dark:bg-gray-950/95`}
        style={{ zIndex: 9999, boxShadow: '0 8px 32px rgba(0,0,0,0.25)', borderRadius: '1.25rem', margin: '0 auto' }}
      >
        <DialogHeader>
          <div className="flex items-start justify-between gap-6">
            <div className="flex items-center gap-4">
              <div
                className={`w-16 h-16 rounded-lg flex items-center justify-center ${incident.type === 'violence'
                  ? 'bg-red-500/20 text-red-400'
                  : 'bg-orange-500/20 text-orange-400'
                  }`}
              >
                <TypeIcon className="w-8 h-8" />
              </div>
              <div>
                <DialogTitle className={`text-4xl ${vigilClasses.textPrimary} flex items-center gap-3`}>
                  {incident.id}
                  <Badge variant="destructive" className="ml-4 text-sm px-2 py-0.5 animate-pulse">
                    ROLE: {String(userRole || 'undefined')}
                  </Badge>
                </DialogTitle>
                <div className="flex items-center gap-3 mt-2">
                  <Badge className={getSeverityColor(incident.severity) + " text-base px-3 py-1.5"}>
                    {incident.severity.toUpperCase()}
                  </Badge>
                  <Badge
                    className={
                      incident.status === 'active'
                        ? 'bg-red-500/20 text-red-400 border-red-500/30 text-base px-3 py-1.5'
                        : incident.status === 'dispatched'
                          ? 'bg-blue-500/20 text-blue-400 border-blue-500/30 text-base px-3 py-1.5'
                          : incident.status === 'acknowledged'
                            ? 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30 text-base px-3 py-1.5'
                            : 'bg-green-500/20 text-green-400 border-green-500/30 text-base px-3 py-1.5'
                    }
                  >
                    {incident.status === 'active' ? (
                      <>
                        <AlertCircle className="w-5 h-5 mr-1.5 animate-pulse" />
                        ACTIVE
                      </>
                    ) : incident.status === 'dispatched' ? (
                      <>
                        <MapPin className="w-5 h-5 mr-1.5" />
                        DISPATCHED
                      </>
                    ) : incident.status === 'acknowledged' ? (
                      <>
                        <CheckCircle className="w-5 h-5 mr-1.5" />
                        ACKNOWLEDGED
                      </>
                    ) : (
                      <>
                        <CheckCircle className="w-5 h-5 mr-1.5" />
                        RESOLVED
                      </>
                    )}
                  </Badge>
                  <Badge className="bg-cyan-500/20 text-cyan-400 border-cyan-500/30 text-base px-3 py-1.5">
                    {incident.type === 'violence' ? 'VIOLENCE DETECTED' : 'CRASH DETECTED'}
                  </Badge>
                </div>
              </div>
            </div>

            {incident.status !== 'resolved' && (
              <div className="flex gap-3">
                {/* Security Role Actions */}
                {String(userRole).toLowerCase() === 'security' ? (
                  <>
                    {(incident.status === 'active' && !incident.assigned_security) ? (
                      <Button
                        onClick={() => {
                          onDispatch?.(incident.id, userName);
                          onClose();
                        }}
                        className="bg-blue-600 hover:bg-blue-700 px-6 py-3 h-auto"
                      >
                        <MapPin className="w-5 h-5 mr-2" />
                        Acknowledge
                      </Button>
                    ) : (
                      // Only show resolution options if assigned to me or generally active
                      // Assuming security only sees "My Dispatches" here where they are assigned
                      <div className="flex gap-2">
                        <Button
                          onClick={() => {
                            onResolve?.(incident.id, 'resolved' as any); // Passing extra arg, TS might complain if not typed, but JS works. 
                            // Actually onResolve signature in props assumes (id: string). We need to update props interface or cast.
                            // Let's rely on the fact that we updated the hook. But the prop type in this file is (id: string) => void.
                            // We should update the interface too.
                            onClose();
                          }}
                          className="bg-green-600 hover:bg-green-700 px-4 py-3 h-auto"
                        >
                          <CheckCircle className="w-5 h-5 mr-2" />
                          [RESOLVED]
                        </Button>
                        <Button
                          onClick={() => {
                            // @ts-ignore
                            onResolve?.(incident.id, 'not_resolved');
                            onClose();
                          }}
                          className="bg-orange-600 hover:bg-orange-700 px-4 py-3 h-auto"
                        >
                          <AlertCircle className="w-5 h-5 mr-2" />
                          [NOT RESOLVED]
                        </Button>
                      </div>
                    )}
                  </>
                ) : (
                  // Admin / Officer Actions
                  <>
                    {/* Admin can always resolve generally */}
                    <Button
                      onClick={() => onResolve?.(incident.id)}
                      className="bg-green-600 hover:bg-green-700 px-6 py-3 h-auto hidden" // Hiding generic resolve for clarity if using specific flow
                    >
                      Resolve
                    </Button>



                    <Button
                      onClick={() => onDismiss?.(incident.id)}
                      variant="outline"
                      className="border-red-600 text-red-400 hover:bg-red-950/30 px-6 py-3 h-auto"
                    >
                      <XCircle className="w-5 h-5 mr-2" />
                      Dismiss
                    </Button>
                  </>
                )}
              </div>
            )}
          </div>
        </DialogHeader>

        <div className="mt-8">
          <Tabs defaultValue="video" className="w-full">
            <TabsList className="grid w-full grid-cols-4 h-14">
              <TabsTrigger value="video" className="text-lg">Video Evidence</TabsTrigger>
              <TabsTrigger value="details" className="text-lg">Details</TabsTrigger>
              <TabsTrigger value="nearby" className="text-lg">Nearby Cameras</TabsTrigger>
              <TabsTrigger value="activity" className="text-lg">Activity Log</TabsTrigger>
            </TabsList>

            {/* Video Tab */}
            <TabsContent value="video" className="space-y-6 mt-4 md:mt-8">
              {/* Inline Video Player */}
              <div className="relative aspect-video bg-black rounded-lg overflow-hidden border border-gray-800 group">
                {/* Offline State */}
                {!videoUrl ? (
                  <div className="absolute inset-0 flex items-center justify-center bg-gray-900 text-gray-400">
                    <p>Video Unavailable</p>
                  </div>
                ) : (
                  <>
                    <video
                      src={videoUrl}
                      className="w-full h-full object-contain"
                      controls
                      autoPlay={isPlaying}
                      width="100%"
                      height="100%"
                    >
                      Your browser does not support the video tag.
                    </video>
                  </>
                )}
              </div>
              {/* Video Controls and export options removed; VideoModal handles all video UI */}
            </TabsContent>

            {/* Details Tab */}
            <TabsContent value="details" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className={`p-4 rounded-lg ${vigilClasses.card}`}>
                  <h4 className={`text-sm font-medium mb-3 ${vigilClasses.textMuted}`}>
                    Incident Information
                  </h4>
                  <div className="space-y-3">
                    <div>
                      <p className={`text-xs ${vigilClasses.textMuted}`}>Description</p>
                      <p className={vigilClasses.textPrimary}>{incident.description}</p>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <p className={`text-xs ${vigilClasses.textMuted}`}>Type</p>
                        <p className={vigilClasses.textPrimary + " capitalize"}>{incident.type}</p>
                      </div>
                      <div>
                        <p className={`text-xs ${vigilClasses.textMuted}`}>Severity</p>
                        <p className={vigilClasses.textPrimary + " capitalize"}>{incident.severity}</p>
                      </div>
                    </div>
                    <div>
                      <p className={`text-xs ${vigilClasses.textMuted} mb-1`}>AI Confidence</p>
                      <div className="flex items-center gap-2">
                        <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-800 rounded-full overflow-hidden">
                          <div
                            className={`h-full ${incident.confidence > 90
                              ? 'bg-green-500'
                              : incident.confidence > 80
                                ? 'bg-cyan-500'
                                : 'bg-amber-500'
                              }`}
                            style={{ width: `${incident.confidence}%` }}
                          />
                        </div>
                        <span className={vigilClasses.textPrimary}>{incident.confidence}%</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className={`p-4 rounded-lg ${vigilClasses.card}`}>
                  <h4 className={`text-sm font-medium mb-3 ${vigilClasses.textMuted}`}>
                    Location & Time
                  </h4>
                  <div className="space-y-3">
                    <div className="flex items-start gap-2">
                      <MapPin className="w-4 h-4 text-cyan-400 mt-0.5" />
                      <div>
                        <p className={`text-xs ${vigilClasses.textMuted}`}>Location</p>
                        <p className={vigilClasses.textPrimary}>{incident.location}</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-2">
                      <Eye className="w-4 h-4 text-cyan-400 mt-0.5" />
                      <div>
                        <p className={`text-xs ${vigilClasses.textMuted}`}>Camera</p>
                        <p className={vigilClasses.textPrimary}>{incident.cameraId}</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-2">
                      <Clock className="w-4 h-4 text-cyan-400 mt-0.5" />
                      <div>
                        <p className={`text-xs ${vigilClasses.textMuted}`}>Detected</p>
                        <p className={vigilClasses.textPrimary}>{incident.timestamp}</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Notes Section */}
              <div className={`p-4 rounded-lg ${vigilClasses.card}`}>
                <h4 className={`text-sm font-medium mb-3 ${vigilClasses.textMuted}`}>
                  <FileText className="w-4 h-4 inline mr-2" />
                  Officer Notes
                </h4>
                <Textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Add notes about this incident..."
                  className="min-h-[100px]"
                />
                <div className="flex justify-end mt-2">
                  <Button size="sm" className="bg-cyan-600 hover:bg-cyan-700">
                    Save Notes
                  </Button>
                </div>
              </div>
            </TabsContent>

            {/* Nearby Cameras Tab */}
            <TabsContent value="nearby" className="space-y-4">
              <p className={`text-sm ${vigilClasses.textMuted}`}>
                View footage from nearby cameras for additional context
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {nearbyFootage.map((cam) => (
                  <div key={cam.cameraId} className={`p-4 rounded-lg ${vigilClasses.card}`}>
                    <div className="aspect-video bg-gradient-to-br from-gray-950 to-gray-900 rounded-lg overflow-hidden border border-gray-800 relative mb-3">
                      <div className="absolute inset-0 flex items-center justify-center">
                        <Camera className="w-12 h-12 text-cyan-500/20" />
                      </div>
                      <div className="absolute top-2 left-2 bg-black/80 px-2 py-1 rounded text-xs text-white font-mono">
                        {cam.cameraId}
                      </div>
                    </div>
                    <div className="space-y-1">
                      <p className={vigilClasses.textPrimary}>{cam.location}</p>
                      <p className={`text-xs ${vigilClasses.textMuted}`}>
                        Distance: {cam.distance}
                      </p>
                      <Button size="sm" variant="outline" className="w-full mt-2">
                        <Play className="w-3 h-3 mr-1" />
                        View Footage
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </TabsContent>

            {/* Activity Log Tab */}
            <TabsContent value="activity" className="space-y-3">
              {activityLog.map((log, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className={`p-3 rounded-lg ${vigilClasses.card} flex items-start gap-3`}
                >
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center ${log.type === 'system'
                      ? 'bg-cyan-500/20 text-cyan-400'
                      : 'bg-blue-500/20 text-blue-400'
                      }`}
                  >
                    {log.type === 'system' ? (
                      <AlertCircle className="w-4 h-4" />
                    ) : (
                      <User className="w-4 h-4" />
                    )}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <p className={`text-sm ${vigilClasses.textPrimary}`}>{log.action}</p>
                      <span className={`text-xs ${vigilClasses.textMuted}`}>{log.time}</span>
                    </div>
                    <p className={`text-xs ${vigilClasses.textMuted} mt-0.5`}>{log.user}</p>
                  </div>
                </motion.div>
              ))}
            </TabsContent>
          </Tabs>
        </div>
      </DialogContent>
    </Dialog>
  );
}