import { useState } from 'react';
import { Card } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { vigilClasses } from './vigil-theme';
import {
  Camera,
  Radio,
  AlertCircle,
  MapPin,
  Maximize2,
  X,
  Settings,
  Play,
  Pause,
  Grid3x3,
  Rows4,
  LayoutGrid,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

interface CameraWallProps {
  onClose?: () => void;
}

export function CameraWall({ onClose }: CameraWallProps) {
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);
  const [gridSize, setGridSize] = useState<'small' | 'medium' | 'large'>('medium');
  const [isPaused, setIsPaused] = useState(false);

  const cameras = [
    {
      id: 'CAM-042',
      name: 'Main Street Camera 1',
      location: 'Main St & 5th Ave',
      zone: 'Downtown',
      status: 'online',
      hasAlert: true,
      alertType: 'violence',
    },
    {
      id: 'CAM-128',
      name: 'Highway Monitor 23',
      location: 'Highway 101, Exit 23',
      zone: 'Highway',
      status: 'online',
      hasAlert: false,
    },
    {
      id: 'CAM-089',
      name: 'Park Plaza Surveillance',
      location: 'Park Plaza',
      zone: 'Downtown',
      status: 'online',
      hasAlert: true,
      alertType: 'crash',
    },
    {
      id: 'CAM-156',
      name: 'Downtown Bridge Monitor',
      location: 'Downtown Bridge',
      zone: 'Downtown',
      status: 'online',
      hasAlert: false,
    },
    {
      id: 'CAM-203',
      name: 'Shopping District Cam',
      location: 'Shopping District',
      zone: 'Residential',
      status: 'online',
      hasAlert: false,
    },
    {
      id: 'CAM-074',
      name: 'Riverside Parkway',
      location: 'Riverside Parkway',
      zone: 'Residential',
      status: 'maintenance',
      hasAlert: false,
    },
    {
      id: 'CAM-112',
      name: 'Industrial Park A',
      location: 'Industrial Park A',
      zone: 'Industrial',
      status: 'online',
      hasAlert: false,
    },
    {
      id: 'CAM-198',
      name: 'North Terminal',
      location: 'North Terminal',
      zone: 'Highway',
      status: 'online',
      hasAlert: false,
    },
    {
      id: 'CAM-215',
      name: 'Central Station',
      location: 'Central Station',
      zone: 'Downtown',
      status: 'online',
      hasAlert: false,
    },
    {
      id: 'CAM-234',
      name: 'West Gate Entrance',
      location: 'West Gate',
      zone: 'Industrial',
      status: 'online',
      hasAlert: false,
    },
    {
      id: 'CAM-245',
      name: 'East Market Plaza',
      location: 'East Market',
      zone: 'Residential',
      status: 'online',
      hasAlert: false,
    },
    {
      id: 'CAM-267',
      name: 'South Parkway',
      location: 'South Parkway',
      zone: 'Highway',
      status: 'online',
      hasAlert: false,
    },
  ];

  const gridClasses = {
    small: 'grid-cols-3 md:grid-cols-4 lg:grid-cols-6',
    medium: 'grid-cols-2 md:grid-cols-3 lg:grid-cols-4',
    large: 'grid-cols-2 md:grid-cols-3',
  };

  return (
    <div className="fixed inset-0 bg-gray-950 dark:bg-black z-50 overflow-hidden">
      {/* Top Control Bar */}
      <div className="absolute top-0 left-0 right-0 bg-gradient-to-b from-black/80 to-transparent z-10 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <Camera className="w-6 h-6 text-cyan-500" />
              <h2 className="text-gray-900 dark:text-white text-xl">Camera Wall</h2>
            </div>
            <Badge className="bg-green-500/20 text-green-400 border-green-500/30">
              <Radio className="w-3 h-3 mr-1 animate-pulse" />
              {cameras.filter((c) => c.status === 'online').length} Live
            </Badge>
            {cameras.some((c) => c.hasAlert) && (
              <Badge className="bg-red-500/20 text-red-400 border-red-500/30 animate-pulse">
                <AlertCircle className="w-3 h-3 mr-1" />
                {cameras.filter((c) => c.hasAlert).length} Alerts
              </Badge>
            )}
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsPaused(!isPaused)}
              className="bg-white/10 hover:bg-white/20 text-gray-900 dark:text-white border-white/20"
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

            <div className="flex items-center gap-1 border border-white/20 rounded-lg p-1 bg-white/10">
              <Button
                variant={gridSize === 'large' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setGridSize('large')}
                className="h-8 w-8 p-0"
              >
                <LayoutGrid className="w-4 h-4" />
              </Button>
              <Button
                variant={gridSize === 'medium' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setGridSize('medium')}
                className="h-8 w-8 p-0"
              >
                <Grid3x3 className="w-4 h-4" />
              </Button>
              <Button
                variant={gridSize === 'small' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setGridSize('small')}
                className="h-8 w-8 p-0"
              >
                <Rows4 className="w-4 h-4" />
              </Button>
            </div>

            {onClose && (
              <Button
                variant="outline"
                size="sm"
                onClick={onClose}
                className="bg-white/10 hover:bg-white/20 text-gray-900 dark:text-white border-white/20"
              >
                <X className="w-4 h-4 mr-1" />
                Exit
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* Camera Grid */}
      <div className="h-full overflow-auto pt-20 pb-4 px-4">
        <div className={`grid ${gridClasses[gridSize]} gap-2`}>
          {cameras.map((camera, index) => (
            <motion.div
              key={camera.id}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.02 }}
              onClick={() => setSelectedCamera(camera.id)}
              className="cursor-pointer"
            >
              <Card
                className={`group relative overflow-hidden ${
                  camera.hasAlert
                    ? 'border-red-500/50 shadow-lg shadow-red-500/20'
                    : 'border-gray-700/50'
                } hover:border-cyan-500/50 transition-all duration-300 bg-gray-900/50 backdrop-blur`}
              >
                {/* Video Feed Placeholder */}
                <div
                  className={`aspect-video bg-gradient-to-br ${
                    camera.hasAlert
                      ? 'from-red-950/60 to-red-900/60'
                      : 'from-gray-900 to-gray-800'
                  } relative overflow-hidden`}
                >
                  {/* Grid Pattern */}
                  <div className="absolute inset-0 opacity-10">
                    <div className="w-full h-full bg-[linear-gradient(rgba(6,182,212,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(6,182,212,0.1)_1px,transparent_1px)] bg-[size:20px_20px]" />
                  </div>

                  {/* Camera Icon */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Camera
                      className={`${
                        gridSize === 'small' ? 'w-8 h-8' : 'w-12 h-12'
                      } ${
                        camera.status === 'online'
                          ? 'text-cyan-500/30'
                          : 'text-gray-600/30'
                      }`}
                    />
                  </div>

                  {/* Status Indicator */}
                  <div className="absolute top-2 left-2 flex items-center gap-1">
                    <div
                      className={`w-1.5 h-1.5 rounded-full ${
                        camera.status === 'online'
                          ? 'bg-green-500 animate-pulse'
                          : 'bg-amber-500'
                      }`}
                    />
                    <span className="text-xs text-gray-900/90 dark:text-white/90 backdrop-blur-sm bg-black/30 px-1.5 py-0.5 rounded">
                      {camera.status === 'online' ? 'LIVE' : 'MAINT'}
                    </span>
                  </div>

                  {/* Alert Badge */}
                  {camera.hasAlert && (
                    <div className="absolute top-2 right-2">
                      <Badge className="bg-red-500 text-gray-900 dark:text-white border-0 animate-pulse text-xs">
                        <AlertCircle className="w-2.5 h-2.5 mr-1" />
                        {camera.alertType === 'violence' ? 'Violence' : 'Crash'}
                      </Badge>
                    </div>
                  )}

                  {/* Timestamp */}
                  <div className="absolute bottom-2 left-2 text-xs text-gray-900/70 dark:text-white/70 backdrop-blur-sm bg-black/30 px-1.5 py-0.5 rounded font-mono">
                    {new Date().toLocaleTimeString()}
                  </div>

                  {/* Hover Actions */}
                  <div className="absolute bottom-2 right-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button
                      size="sm"
                      className="bg-cyan-600 hover:bg-cyan-700 text-gray-900 dark:text-white text-xs h-7"
                    >
                      <Maximize2 className="w-3 h-3 mr-1" />
                      View
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="bg-white/10 hover:bg-white/20 text-white border-white/20 text-xs h-7"
                    >
                      <Settings className="w-3 h-3" />
                    </Button>
                  </div>
                </div>

                {/* Camera Info */}
                <div className="p-2 space-y-0.5 bg-gray-900/80">
                  <div className="flex items-start justify-between gap-1">
                    <div className="min-w-0 flex-1">
                      <h3 className="text-xs text-white truncate">{camera.id}</h3>
                      {gridSize !== 'small' && (
                        <p className="text-xs text-gray-400 truncate">{camera.name}</p>
                      )}
                    </div>
                    <Badge
                      variant="outline"
                      className="text-xs flex-shrink-0 bg-cyan-500/10 text-cyan-400 border-cyan-500/30"
                    >
                      {camera.zone}
                    </Badge>
                  </div>

                  {gridSize !== 'small' && (
                    <div className="flex items-center gap-1 text-xs text-gray-500">
                      <MapPin className="w-2.5 h-2.5 flex-shrink-0" />
                      <span className="truncate">{camera.location}</span>
                    </div>
                  )}
                </div>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Fullscreen Camera Modal */}
      <AnimatePresence>
        {selectedCamera && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/95 z-50 flex items-center justify-center p-4"
            onClick={() => setSelectedCamera(null)}
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              className="w-full max-w-6xl"
              onClick={(e) => e.stopPropagation()}
            >
              <Card className={`${vigilClasses.card} overflow-hidden`}>
                <div className="aspect-video bg-black relative flex items-center justify-center">
                  {/* Replace with real video feed if available, else fallback */}
                  <video
                    src={cameras.find((c) => c.id === selectedCamera)?.videoUrl || ''}
                    autoPlay
                    controls
                    className="w-full h-full object-contain bg-black"
                  >
                    Your browser does not support the video tag.
                  </video>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setSelectedCamera(null)}
                    className="absolute top-4 right-4 bg-white/10 hover:bg-white/20 text-white border-white/20"
                  >
                    <X className="w-4 h-4 mr-1" />
                    Close
                  </Button>
                </div>
                <div className="p-4 bg-gray-900/50">
                  <h3 className="text-xl text-white mb-2">{selectedCamera}</h3>
                  <p className="text-gray-400">
                    {cameras.find((c) => c.id === selectedCamera)?.name}
                  </p>
                </div>
              </Card>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
