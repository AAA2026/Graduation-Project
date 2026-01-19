import { useState } from 'react';
import { useLiveStatus } from '../hooks/useLiveStatus';
import { Card } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { vigilClasses } from './vigil-theme';
import { CameraWall } from './CameraWall';
import { Dialog } from './ui/dialog';
import {
  Camera,
  Radio,
  AlertCircle,
  MapPin,
  Maximize2,
  Eye,
  Settings,
  RefreshCw,
  Grid3x3,
  Grid2x2,
  Rows3,
  Play,
  Pause,
  Expand,
} from 'lucide-react';
import { motion } from 'motion/react';

interface LiveCameraGridProps {
  role: 'admin' | 'officer' | 'security-authority';
  dvrMode?: boolean;
}

export function LiveCameraGrid({ role, dvrMode = false }: LiveCameraGridProps) {
  const [gridLayout, setGridLayout] = useState<'2x2' | '3x2' | '3x3' | '4x4'>(
    dvrMode ? '3x2' : '2x2'
  );
  const [isPaused, setIsPaused] = useState(false);
  const [showCameraWall, setShowCameraWall] = useState(false);
  const [fullscreenCamera, setFullscreenCamera] = useState<null | (typeof cameras)[0]>(null);
  // Use real backend data via useLiveStatus
  const {
    cameraStatuses,
    systemStatus,
    lastUpdate,
    isPolling,
    getCameraStatus,
    togglePolling,
    refreshStatus,
    offlineMode,
    toggleOfflineMode,
  } = useLiveStatus();

  // Map backend camera data to frontend format
  const DEFAULT_CAMERAS = [
    'CAM-042', 'CAM-128', 'CAM-089', 'CAM-156',
    'CAM-283', 'CAM-074', 'CAM-195', 'CAM-267',
    'CAM-341', 'CAM-412', 'CAM-523', 'CAM-604'
  ];

  // Helper to build video URL
  function buildVideoUrl(video: string | undefined) {
    if (!video) return '';
    let url = video.startsWith('/videos/') ? video : `/videos/${video.replace(/^\/+/,'')}`;
    if (import.meta.env.VITE_API_URL) {
      url = `${(import.meta.env.VITE_API_URL || '').replace(/\/$/, '')}${url}`;
    }
    return url;
  }

  // Compose camera objects for grid
  const cameras = DEFAULT_CAMERAS.map((cid) => {
    const cam = cameraStatuses.get(cid);
    return {
      id: cid,
      name: cam?.location || cid,
      location: cam?.location || '',
      zone: '',
      status: cam?.status || 'offline',
      hasAlert: cam?.event === 'violence' || cam?.event === 'crash',
      alertType: cam?.event,
      lastActivity: cam?.last_update ? `${Math.round((Date.now()/1000 - cam.last_update)/60)} min ago` : '',
      videoUrl: buildVideoUrl(cam?.video),
    };
  });

  const gridCols = {
    '2x2': 'grid-cols-1 sm:grid-cols-2',
    '3x2': 'grid-cols-2 lg:grid-cols-3',
    '3x3': 'grid-cols-2 lg:grid-cols-3',
    '4x4': 'grid-cols-2 md:grid-cols-3 lg:grid-cols-4',
  };

  const camerasToShow = {
    '2x2': 4,
    '3x2': 6,
    '3x3': 9,
    '4x4': 16,
  };

  return (
    <div className="space-y-3 sm:space-y-4">
      {/* Control Bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 sm:gap-4">
        <div className="flex items-center gap-2 sm:gap-3">
          <div className="flex items-center gap-2">
            <Eye className="w-5 h-5 sm:w-6 sm:h-6 text-cyan-500" />
            <h2 className={`text-lg sm:text-xl ${vigilClasses.textPrimary}`}>
              Live Camera Feed
            </h2>
          </div>
          <Badge className="bg-green-500/20 text-green-600 dark:text-green-400 border-green-500/30">
            <Radio className="w-3 h-3 mr-1 animate-pulse" />
            {cameras.filter((c) => c.status === 'online').length} Live
          </Badge>
          {cameras.some((c) => c.hasAlert) && (
            <Badge className="bg-red-500/20 text-red-600 dark:text-red-400 border-red-500/30 animate-pulse">
              <AlertCircle className="w-3 h-3 mr-1" />
              {cameras.filter((c) => c.hasAlert).length} Active Alerts
            </Badge>
          )}
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowCameraWall(true)}
            className={`bg-cyan-600 hover:bg-cyan-700 text-white border-0 hidden lg:flex`}
          >
            <Expand className="w-4 h-4 mr-1" />
            Camera Wall
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsPaused(!isPaused)}
            className={`flex-1 sm:flex-none`}
          >
            {isPaused ? (
              <>
                <Play className="w-4 h-4 mr-1" />
                Resume
              </>
            ) : (
              <>
                <Pause className="w-4 h-4 mr-1" />
                Pause
              </>
            )}
          </Button>

          <div className="flex items-center gap-1 border rounded-lg p-1 bg-gray-50 dark:bg-[#0a0e1a]">
            <Button
              variant={gridLayout === '2x2' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setGridLayout('2x2')}
              className="h-8 w-8 p-0"
            >
              <Grid2x2 className="w-4 h-4" />
            </Button>
            <Button
              variant={gridLayout === '3x2' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setGridLayout('3x2')}
              className="h-8 w-8 p-0 hidden sm:flex"
            >
              <Grid3x3 className="w-4 h-4" />
            </Button>
            <Button
              variant={gridLayout === '3x3' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setGridLayout('3x3')}
              className="h-8 w-8 p-0 hidden sm:flex"
            >
              <Grid3x3 className="w-4 h-4" />
            </Button>
            <Button
              variant={gridLayout === '4x4' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setGridLayout('4x4')}
              className="h-8 w-8 p-0 hidden md:flex"
            >
              <Rows3 className="w-4 h-4" />
            </Button>
          </div>

          <Button
            variant="outline"
            size="sm"
            className={`hidden sm:flex`}
          >
            <RefreshCw className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Camera Grid */}
      <div className={`grid ${gridCols[gridLayout]} gap-2 sm:gap-3 lg:gap-4`}>
        {cameras.slice(0, camerasToShow[gridLayout]).map((camera, index) => (
          <motion.div
            key={camera.id}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.05 }}
          >
            <Card
              className={`group relative overflow-hidden ${
                camera.hasAlert
                  ? 'border-red-500/50 shadow-lg shadow-red-500/20'
                  : vigilClasses.card
              } hover:border-cyan-500/50 transition-all duration-300 cursor-pointer`}
            >
              {/* Video Feed */}
              <div
                className={`aspect-video bg-black relative overflow-hidden`}
              >
                {/* Real video feed if available, else fallback icon */}
                {camera.videoUrl ? (
                  <video
                    src={camera.videoUrl}
                    autoPlay
                    muted
                    loop
                    className="absolute inset-0 w-full h-full object-cover bg-black"
                  />
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Camera
                      className={`w-12 h-12 sm:w-16 sm:h-16 ${
                        camera.status === 'online'
                          ? 'text-cyan-500/30'
                          : 'text-gray-600/30'
                      }`}
                    />
                  </div>
                )}

                {/* Status Indicator */}
                <div className="absolute top-2 left-2 flex items-center gap-2">
                  <div
                    className={`w-2 h-2 rounded-full ${
                      camera.status === 'online'
                        ? 'bg-green-500 animate-pulse'
                        : 'bg-amber-500'
                    }`}
                  />
                  <span className="text-xs text-white/90 backdrop-blur-sm bg-black/30 px-2 py-0.5 rounded">
                    {camera.status === 'online' ? 'LIVE' : 'MAINTENANCE'}
                  </span>
                </div>

                {/* Alert Badge */}
                {camera.hasAlert && (
                  <div className="absolute top-2 right-2">
                    <Badge className="bg-red-500 text-white border-0 animate-pulse">
                      <AlertCircle className="w-3 h-3 mr-1" />
                      {camera.alertType === 'violence' ? 'Violence' : 'Crash'}
                    </Badge>
                  </div>
                )}

                {/* Timestamp Overlay */}
                <div className="absolute bottom-2 left-2 text-xs text-white/70 backdrop-blur-sm bg-black/30 px-2 py-0.5 rounded font-mono">
                  {new Date().toLocaleTimeString()}
                </div>

                {/* Hover Actions: Only Fullscreen and Settings, no View Live */}
                <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center gap-2">
                  <Button
                    size="sm"
                    className="bg-cyan-600 hover:bg-cyan-700 text-white"
                    onClick={() => setFullscreenCamera(camera)}
                  >
                    <Maximize2 className="w-4 h-4 mr-1" />
                    Fullscreen
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="bg-white/10 hover:bg-white/20 text-white border-white/20"
                  >
                    <Settings className="w-4 h-4" />
                  </Button>
                </div>
              </div>

              {/* Camera Info */}
              <div className="p-2 sm:p-3 space-y-1">
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    <h3
                      className={`text-xs sm:text-sm ${vigilClasses.textPrimary} truncate`}
                    >
                      {camera.id}
                    </h3>
                    <p
                      className={`text-xs ${vigilClasses.textSecondary} truncate`}
                    >
                      {camera.name}
                    </p>
                  </div>
                  <Badge
                    variant="outline"
                    className="text-xs flex-shrink-0 bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border-cyan-500/30"
                  >
                    {camera.zone}
                  </Badge>
                </div>

                <div className="flex items-center gap-1 text-xs text-gray-600 dark:text-gray-400">
                  <MapPin className="w-3 h-3 flex-shrink-0" />
                  <span className="truncate">{camera.location}</span>
                </div>
              </div>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Camera Wall */}
      {showCameraWall && (
        <CameraWall onClose={() => setShowCameraWall(false)} />
      )}

      {/* Fullscreen Video Modal */}
      <Dialog open={!!fullscreenCamera} onOpenChange={() => setFullscreenCamera(null)}>
        {fullscreenCamera && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/90">
            <div className="relative w-full max-w-3xl">
              <video
                src={fullscreenCamera.videoUrl}
                autoPlay
                controls
                className="w-full h-[60vh] object-contain bg-black rounded-lg"
              >
                Your browser does not support the video tag.
              </video>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setFullscreenCamera(null)}
                className="absolute top-4 right-4 bg-white/10 hover:bg-white/20 text-white border-white/20"
              >
                Close
              </Button>
              <div className="absolute bottom-4 left-4 text-white">
                <h3 className="text-lg font-bold">{fullscreenCamera.name}</h3>
                <p className="text-sm opacity-80">{fullscreenCamera.location}</p>
              </div>
            </div>
          </div>
        )}
      </Dialog>
    </div>
  );
}