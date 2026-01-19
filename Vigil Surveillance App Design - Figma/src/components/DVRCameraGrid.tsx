// (Removed custom ImportMetaEnv and ImportMeta declarations to fix Vite env typing)

// Add Vite env typing for TypeScript
/// <reference types="vite/client" />

import { useState } from 'react';
import { motion } from 'motion/react';
import {
  Camera,
  Eye,
  Download,
  AlertCircle,
  Radio,
  Grid2x2,
  Grid3x3,
  Rows3,

  Camera as CameraIcon,
  WifiOff,
  Wifi,
  Swords,
  Car,
  Users,
  Maximize2,
  ChevronLeft,
  ChevronRight,
  RefreshCcw
} from 'lucide-react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { VideoModal } from './VideoModal';
import { downloadVideoClip, captureScreenshot } from '../utils/exportUtils';
import { toast } from 'sonner';
import { useLiveStatus } from '../hooks/useLiveStatus';

interface DVRCameraGridProps {
  role: 'admin' | 'officer' | 'security';
}

export function DVRCameraGrid({ role }: DVRCameraGridProps) {
  const [gridLayout, setGridLayout] = useState<'2x2' | '2x3' | '3x3' | '4x4'>('3x3');
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);

  const [videoModalCamera, setVideoModalCamera] = useState<{
    cameraId: string;
    location: string;
    videoUrl: string;
  } | null>(null);
  const [modalIndex, setModalIndex] = useState<number>(0);
  const [gridKey, setGridKey] = useState<number>(0);

  // Use live status polling hook
  const {
    cameraStatuses,
    systemStatus,
    isPolling,
    togglePolling,
    lastUpdate,
    getCameraStatus,
    offlineMode,
    refreshStatus,
  } = useLiveStatus();

  // Always show all expected cameras, even if backend is down or partial
  const DEFAULT_CAMERAS = [
    'CAM-042', 'CAM-128', 'CAM-089', 'CAM-156', 'CAM-283', 'CAM-074',
    'CAM-195', 'CAM-267', 'CAM-341', 'CAM-412', 'CAM-523', 'CAM-604'
  ];
  const cameras = DEFAULT_CAMERAS.map((cid) => {
    const cam = cameraStatuses.get(cid);
    // Ensure videoUrl always starts with /videos/
    let videoUrl = '';
    if (cam?.video) {
      // Normalize path strictly: replace backslashes, remove existing prefixes
      let rawPath = cam.video.replace(/\\/g, '/');

      // Remove /videos/ prefix if present to re-add it cleanly (prevents double prefixing if backend changes)
      rawPath = rawPath.replace(/^\/?videos\//, '');

      // Remove plain leading slash
      rawPath = rawPath.replace(/^\//, '');

      videoUrl = `/videos/${rawPath}`;

      // Optionally add API URL prefix if needed for cross-origin
      if (import.meta.env.VITE_API_URL) {
        videoUrl = `${(import.meta.env.VITE_API_URL || '').replace(/\/$/, '')}${videoUrl}`;
      }
    }
    return {
      id: cid,
      name: cid,
      location: cam?.location || 'Unknown',
      zone: '',
      status: cam?.status || 'offline',
      hasAlert: cam?.event === 'violence' || cam?.event === 'crash',
      alertType: cam?.event,
      videoUrl,
      confidence: cam?.confidence,
    };
  });

  // DEBUG: Log cameraStatuses and cameras array
  console.log('[DVRCameraGrid] cameraStatuses:', cameraStatuses);
  console.log('[DVRCameraGrid] cameras:', cameras);
  if (cameras.length === 0) {
    console.warn('[DVRCameraGrid] No cameras to display!');
  }

  const gridCols = {
    '2x2': 'grid-cols-2',
    '2x3': 'grid-cols-2 md:grid-cols-3',
    '3x3': 'grid-cols-2 md:grid-cols-3',
    '4x4': 'grid-cols-2 md:grid-cols-4',
  };

  const camerasToShow = {
    '2x2': 4,
    '2x3': 6,
    '3x3': 9,
    '4x4': 16,
  };

  const handleViewCamera = (camera: typeof cameras[0], index?: number) => {
    setVideoModalCamera({
      cameraId: camera.id,
      location: camera.location,
      videoUrl: camera.videoUrl,
    });
    setModalIndex(index !== undefined ? index : cameras.findIndex(c => c.id === camera.id));
  };

  const handleDownloadCamera = async (camera: typeof cameras[0]) => {
    try {
      await downloadVideoClip(camera.videoUrl, camera.id, new Date().toISOString());
      toast.success(`Video from ${camera.id} downloaded`);
    } catch (error) {
      toast.error('Failed to download video');
    }
  };

  const handleScreenshot = async (camera: typeof cameras[0]) => {
    try {
      // Find the card element for this camera - we leverage the data attribute we will add
      const elementId = `camera-feed-${camera.id}`;
      // In the grid map below, we need to add this ID to the motion.div or the Card
      await captureScreenshot(elementId, `vigil_snap_${camera.id}_${Date.now()}.png`);
      toast.success(`Screenshot captured from ${camera.id}`);
    } catch (e) {
      console.error(e);
      toast.error("Failed to capture screenshot");
    }
  };


  return (
    <div className="space-y-2">
      {/* Modern Status Bar */}
      <div className="flex flex-wrap items-center gap-2 mb-2 px-2 py-1 rounded-lg bg-transparent">
        <Badge className="bg-blue-100 dark:bg-cyan-900/30 text-blue-700 dark:text-cyan-400 border-blue-200 dark:border-cyan-800 font-semibold text-xs px-2 py-1 flex items-center gap-1">
          <CameraIcon className="w-4 h-4 mr-1" />
          {cameras.length} Cameras
        </Badge>
        <Badge className={`text-xs px-2 py-1 font-semibold flex items-center gap-1 ${systemStatus === 'operational' ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 border-green-200 dark:border-green-800' : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 border-red-200 dark:border-red-800'}`}>
          <Wifi className="w-4 h-4 mr-1" />
          {systemStatus === 'operational' ? 'System Operational' : 'System Down'}
        </Badge>
        <Badge className={`text-xs px-2 py-1 font-semibold flex items-center gap-1 ${isPolling ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 border-green-200 dark:border-green-800' : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 border-red-200 dark:border-red-800'}`}>
          <Radio className="w-4 h-4 mr-1 animate-pulse" />
          Polling: {isPolling ? 'ON' : 'OFF'}
        </Badge>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            // Force remount of the entire grid by updating a key
            setGridKey(prev => prev + 1);
            refreshStatus();
          }}
          className="h-6 px-2 text-xs text-gray-400 hover:text-white hover:bg-white/10"
          title="Force refresh video feed"
        >
          <RefreshCcw className="w-3.5 h-3.5 mr-1" />
          Refresh Feed
        </Button>
        <span className="text-xs text-gray-500 dark:text-gray-400 ml-2 font-mono whitespace-nowrap">Last Update: {lastUpdate.slice(0, 19).replace('T', ' ')}</span>
      </div>
      {/* Video Modal */}
      {videoModalCamera && (
        <VideoModal
          isOpen={!!videoModalCamera}
          onClose={() => setVideoModalCamera(null)}
          videoUrl={videoModalCamera.videoUrl}
          cameraId={videoModalCamera.cameraId}
          location={videoModalCamera.location}
          timestamp={new Date().toISOString()}
          offlineMode={offlineMode}
          // Pass the counter as a separate prop or part of location? 
          // Actually VideoModal takes 'controls' as ReactNode. 
          // I replaced the whole block. 
          // I will use `children` or modify VideoModal to accept overlay elements if needed, 
          // but for now I put the buttons 'absolute' over the video. 
          // Actually the 'controls' prop is rendered in the footer. 
          // If I make them absolute here, they will look correct overlaying the video? 
          // Wait, 'controls' prop is rendered in the FOOTER of VideoModal (line 231 of VideoModal.tsx).
          // If I make them absolute with `top-1/2` they will be relative to the FOOTER div? 
          // No, VideoModal footer is `absolute bottom-0 ...`.
          // If I want them centered on the VIDEO, I should probably inspect VideoModal structure.
          // VideoModal uses `relative` wrapper. 
          // But I can only pass `controls`. 
          // If I want them floating, I should modify specific CSS.
          // Let's assume `controls` is rendered in a place where `absolute top-1/2` might be relative to the footer or the modal.
          // Actually, VideoModal structure: 
          // <div className="relative ..."> ... footer {controls} ... </div>
          // So if I use `absolute top-1/2` inside `controls`, it might position relative to the footer.
          // To be safe, I will position them firmly in the footer but with negative margin or just fixed position relative to the screen/modal if possible.
          // OR I keep them in the footer but style them aggressively.
          // The USER asked for them to "pop". 
          // The previous edit put them absolute. Let's fix that if it's going to be weird.
          // Actually, let's keep them IN FLOW in the footer for safety, but make them HUGE.
          // Re-doing the content to be safer.
          controls={
            <div className="flex justify-between items-center w-full px-4 -mt-12 pointer-events-none relative z-50">
              {/* Lift them up essentially or just keep them in footer line */}
              {/* Let's try sticking to the footer but making them "Pop" as requested */}
              <div className="pointer-events-auto">
                <Button
                  variant="outline"
                  className="w-16 h-16 rounded-full bg-black border-2 border-cyan-400 text-cyan-400 hover:bg-cyan-600 hover:text-white hover:border-cyan-300 shadow-[0_0_20px_rgba(6,182,212,0.6)] transition-all hover:scale-110 flex items-center justify-center transform -translate-y-4"
                  disabled={modalIndex <= 0}
                  onClick={() => {
                    const prev = cameras[modalIndex - 1];
                    if (prev) handleViewCamera(prev, modalIndex - 1);
                  }}
                >
                  <ChevronLeft className="w-10 h-10" />
                </Button>
              </div>

              <div className="pointer-events-auto px-6 py-2 rounded-full bg-black border-2 border-cyan-500 shadow-[0_0_15px_rgba(6,182,212,0.4)] transform -translate-y-4">
                <span className="text-lg font-bold font-mono text-cyan-400">
                  {modalIndex + 1} <span className="text-gray-400 mx-1">/</span> {cameras.length}
                </span>
              </div>

              <div className="pointer-events-auto">
                <Button
                  variant="outline"
                  className="w-16 h-16 rounded-full bg-black border-2 border-cyan-400 text-cyan-400 hover:bg-cyan-600 hover:text-white hover:border-cyan-300 shadow-[0_0_20px_rgba(6,182,212,0.6)] transition-all hover:scale-110 flex items-center justify-center transform -translate-y-4"
                  disabled={modalIndex >= cameras.length - 1}
                  onClick={() => {
                    const next = cameras[modalIndex + 1];
                    if (next) handleViewCamera(next, modalIndex + 1);
                  }}
                >
                  <ChevronRight className="w-10 h-10" />
                </Button>
              </div>
            </div>
          }
        />
      )}

      {/* Minimal Control Bar */}
      <div className="flex items-center justify-between px-2">
        <div className="flex items-center gap-2">
          <Badge className="bg-green-500/20 text-green-600 dark:text-green-400 border-green-500/30 text-xs">
            <Radio className="w-3 h-3 mr-1 animate-pulse" />
            {cameras.filter((c) => {
              const liveStatus = getCameraStatus(c.id);
              return (liveStatus?.status || c.status) === 'online';
            }).length}/{cameras.length}
          </Badge>

        </div>

        <div className="flex items-center gap-1">
          <div className="flex items-center gap-0.5 border border-gray-700 rounded p-0.5 bg-black/30">
            <Button
              variant={gridLayout === '2x2' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setGridLayout('2x2')}
              className="h-6 w-6 p-0"
            >
              <Grid2x2 className="w-3 h-3" />
            </Button>
            <Button
              variant={gridLayout === '2x3' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setGridLayout('2x3')}
              className="h-6 w-6 p-0"
            >
              <Grid3x3 className="w-3 h-3" />
            </Button>
            <Button
              variant={gridLayout === '3x3' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setGridLayout('3x3')}
              className="h-6 w-6 p-0 hidden md:flex"
            >
              <Grid3x3 className="w-3 h-3" />
            </Button>
            <Button
              variant={gridLayout === '4x4' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setGridLayout('4x4')}
              className="h-6 w-6 p-0 hidden lg:flex"
            >
              <Rows3 className="w-3 h-3" />
            </Button>
          </div>


        </div>
      </div>

      {/* DVR-Style Camera Grid - Edge to Edge, No Gaps */}
      <div key={gridKey} className={`grid ${gridCols[gridLayout]} gap-[2px] bg-black/50 p-[2px] rounded-lg overflow-hidden border-2 border-gray-800`}>
        {cameras.slice(0, camerasToShow[gridLayout]).map((camera, index) => {
          // Get real-time status from polling hook
          const liveStatus = getCameraStatus(camera.id);
          const currentStatus = liveStatus?.status || camera.status;

          return (
            <motion.div
              key={camera.id}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: index * 0.03 }}
              onClick={() => setSelectedCamera(selectedCamera === camera.id ? null : camera.id)}
              id={`camera-feed-${camera.id}`}
              className={`relative cursor-pointer group ${selectedCamera === camera.id ? 'ring-2 ring-cyan-500' : ''
                }`}
            >
              {/* Monitor Screen */}
              <div
                className={`aspect-video bg-gradient-to-br relative overflow-hidden ${camera.hasAlert
                  ? 'from-red-950/60 to-red-900/60 border-2 border-red-500'
                  : 'from-gray-950 to-gray-900 border border-gray-800'
                  }`}
              >
                {/* Scanline Effect */}
                <div className="absolute inset-0 pointer-events-none">
                  <div className="w-full h-full bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:100%_4px] animate-scanline" />
                </div>

                {/* Grid Pattern */}
                <div className="absolute inset-0 opacity-5">
                  <div className="w-full h-full bg-[linear-gradient(rgba(6,182,212,0.3)_1px,transparent_1px),linear-gradient(90deg,rgba(6,182,212,0.3)_1px,transparent_1px)] bg-[size:30px_30px]" />
                </div>

                {/* Video Feed or Camera Icon */}
                {currentStatus === 'online' && !offlineMode && camera.videoUrl ? (
                  <video
                    src={camera.videoUrl}
                    className={`absolute inset-0 w-full h-full object-cover`}
                    autoPlay
                    muted
                    loop
                    playsInline
                    controls
                    onError={e => {
                      // fallback to icon if video fails
                      const target = e.target as HTMLVideoElement;
                      target.style.display = 'none';
                      const fallback = document.createElement('div');
                      fallback.className = 'absolute inset-0 flex items-center justify-center text-red-500 bg-black/80';
                      fallback.innerText = 'Video cannot be played';
                      (target.parentNode as HTMLElement | null)?.appendChild(fallback);
                    }}
                  >
                    Sorry, your browser cannot play this video.
                  </video>
                ) : (
                  <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80">
                    {offlineMode ? (
                      <>
                        <WifiOff className={`text-red-500/50 mb-2 ${gridLayout === '4x4' ? 'w-8 h-8' : gridLayout === '3x3' ? 'w-10 h-10' : 'w-12 h-12'
                          }`} />
                        <span className="text-red-500/70 font-mono text-sm font-bold tracking-widest">OFFLINE</span>
                      </>
                    ) : (
                      <Camera
                        className={`$
                          gridLayout === '4x4' ? 'w-8 h-8' : gridLayout === '3x3' ? 'w-10 h-10' : 'w-12 h-12'
                        } text-cyan-500/20`}
                      />
                    )}
                  </div>
                )}

                {/* Top Overlay - Status and Camera ID */}
                <div className="absolute top-0 left-0 right-0 bg-gradient-to-b from-black/80 to-transparent p-1.5">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5">
                      <div
                        className={`w-1.5 h-1.5 rounded-full ${currentStatus === 'online'
                          ? 'bg-green-500 animate-pulse shadow-lg shadow-green-500/50'
                          : currentStatus === 'offline'
                            ? 'bg-red-500'
                            : 'bg-amber-500'
                          }`}
                      />
                      <span className="text-white text-xs font-mono">{camera.id}</span>
                    </div>
                    {/* Remove event/model tags from grid */}
                  </div>
                </div>


                {/* Bottom Overlay - Location and Time */}
                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-white/80 truncate text-xs font-medium">
                      {camera.location}
                    </span>
                    <span className="text-cyan-400 font-mono text-xs ml-1">
                      {new Date().toLocaleTimeString('en-US', {
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                        hour12: false,
                      })}
                    </span>
                  </div>
                </div>

                {/* Center Hover Action */}
                <div className="absolute inset-0 bg-black/70 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center justify-center gap-2 p-2"
                  onClick={(e) => e.stopPropagation()}
                >
                  <Button
                    size="sm"
                    className="bg-blue-600 hover:bg-blue-700 text-white shadow-lg text-xs h-7"
                    onClick={() => handleViewCamera(camera)}
                  >
                    <Maximize2 className="w-3 h-3 mr-1" />
                    Full Screen
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="border-cyan-500/50 text-cyan-400 hover:bg-cyan-500/10 shadow-lg text-xs h-7"
                    onClick={() => handleScreenshot(camera)}
                  >
                    <CameraIcon className="w-3 h-3 mr-1" />
                    Screenshot
                  </Button>
                  <Button
                    size="sm"
                    className="bg-green-600 hover:bg-green-700 text-white shadow-lg text-xs h-7"
                    onClick={() => handleDownloadCamera(camera)}
                  >
                    <Download className="w-3 h-3 mr-1" />
                    Download
                  </Button>
                </div>

                {/* Recording Indicator */}
                {currentStatus === 'online' && (
                  <div className="absolute top-1.5 right-1.5">
                    <div className="flex items-center gap-1 bg-red-600/90 px-1.5 py-0.5 rounded">
                      <div className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
                      <span className="text-white text-xs">REC</span>
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Bottom Info Bar */}
      <div className="flex items-center justify-between px-2 py-1 bg-black/30 border border-gray-800 rounded text-xs">
        <div className="flex items-center gap-3">
          <span className="text-gray-500">
            Displaying {camerasToShow[gridLayout]} of {cameras.length} cameras
          </span>
          {selectedCamera && (
            <>
              <span className="text-gray-600">|</span>
              <span className="text-cyan-400">Selected: {selectedCamera}</span>
            </>
          )}
        </div>
        <div className="text-gray-500 font-mono">
          {new Date().toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric',
          })}
        </div>
      </div>
    </div >
  );
}