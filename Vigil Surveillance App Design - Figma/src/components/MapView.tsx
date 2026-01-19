import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { ScrollArea } from './ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import {
  MapPin,
  Camera,
  AlertCircle,
  Eye,
  Radio,
  Navigation,
  ZoomIn,
  ZoomOut,
  Maximize,
} from 'lucide-react';
import { useTheme } from './ThemeProvider';

export function MapView() {
  const [selectedCamera, setSelectedCamera] = useState<number | null>(null);
  const { theme } = useTheme();

  const cameras = [
    { id: 1, name: 'CAM-042', location: 'Main St & 5th Ave', zone: 'Downtown', status: 'online', hasIncident: true, lat: 35, lng: 45 },
    { id: 2, name: 'CAM-128', location: 'Highway 101, Exit 23', zone: 'Highway', status: 'online', hasIncident: true, lat: 42, lng: 38 },
    { id: 3, name: 'CAM-089', location: 'Park Plaza', zone: 'Downtown', status: 'online', hasIncident: false, lat: 28, lng: 55 },
    { id: 4, name: 'CAM-156', location: 'Downtown Bridge', zone: 'Downtown', status: 'online', hasIncident: false, lat: 65, lng: 32 },
    { id: 5, name: 'CAM-203', location: 'Shopping District', zone: 'Residential', status: 'online', hasIncident: false, lat: 48, lng: 68 },
    { id: 6, name: 'CAM-074', location: 'Riverside Parkway', zone: 'Residential', status: 'offline', hasIncident: false, lat: 72, lng: 58 },
    { id: 7, name: 'CAM-112', location: 'Industrial Park A', zone: 'Industrial', status: 'online', hasIncident: false, lat: 20, lng: 25 },
    { id: 8, name: 'CAM-198', location: 'North Terminal', zone: 'Highway', status: 'online', hasIncident: false, lat: 55, lng: 75 },
  ];

  const selectedCameraData = cameras.find((cam) => cam.id === selectedCamera);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl sm:text-2xl font-bold text-foreground mb-1">Map View</h2>
        <p className="text-sm sm:text-base text-muted-foreground">View camera locations and incidents on the map.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6">
        {/* Map */}
        <div className="lg:col-span-2">
          <Card className="bg-card/80 backdrop-blur border-border overflow-hidden">
            <CardHeader className="border-b border-border p-3 sm:p-6">
              <div className="flex items-center justify-between gap-2">
                <CardTitle className="flex items-center gap-2 text-card-foreground text-base sm:text-lg">
                  <Navigation className="w-4 h-4 sm:w-5 sm:h-5 text-blue-600 dark:text-blue-300" />
                  <span className="hidden sm:inline">Camera Network Map</span>
                  <span className="sm:hidden">Network Map</span>
                </CardTitle>
                <div className="flex items-center gap-1 sm:gap-2">
                  <Button
                    size="icon"
                    variant="ghost"
                    className="w-7 h-7 sm:w-8 sm:h-8 text-muted-foreground hover:text-foreground"
                  >
                    <ZoomIn className="w-3 h-3 sm:w-4 sm:h-4" />
                  </Button>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="w-7 h-7 sm:w-8 sm:h-8 text-muted-foreground hover:text-foreground"
                  >
                    <ZoomOut className="w-3 h-3 sm:w-4 sm:h-4" />
                  </Button>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="w-7 h-7 sm:w-8 sm:h-8 text-muted-foreground hover:text-foreground"
                  >
                    <Maximize className="w-3 h-3 sm:w-4 sm:h-4" />
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <div className="relative w-full h-[600px] bg-gradient-to-br from-muted/50 via-background to-muted/50">
                {/* Grid Lines */}
                <div className="absolute inset-0 opacity-10">
                  {[...Array(20)].map((_, i) => (
                    <div
                      key={`v-${i}`}
                      className="absolute top-0 bottom-0 border-l border-primary"
                      style={{ left: `${(i + 1) * 5}%` }}
                    />
                  ))}
                  {[...Array(12)].map((_, i) => (
                    <div
                      key={`h-${i}`}
                      className="absolute left-0 right-0 border-t border-primary"
                      style={{ top: `${(i + 1) * 8.33}%` }}
                    />
                  ))}
                </div>

                {/* Zone Labels */}
                <div className="absolute top-4 left-4 space-y-2">
                  <Badge className="bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-500/20 dark:text-blue-400 dark:border-blue-500/30">
                    Downtown
                  </Badge>
                </div>
                <div className="absolute top-4 right-4 space-y-2">
                  <Badge className="bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-500/20 dark:text-amber-400 dark:border-amber-500/30">
                    Highway
                  </Badge>
                </div>
                <div className="absolute bottom-4 left-4 space-y-2">
                  <Badge className="bg-purple-100 text-purple-800 border-purple-200 dark:bg-purple-500/20 dark:text-purple-400 dark:border-purple-500/30">
                    Industrial
                  </Badge>
                </div>
                <div className="absolute bottom-4 right-4 space-y-2">
                  <Badge className="bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30">
                    Residential
                  </Badge>
                </div>

                {/* Camera Pins */}
                {cameras.map((camera) => (
                  <button
                    key={camera.id}
                    onClick={() => setSelectedCamera(camera.id)}
                    className={`absolute group cursor-pointer transform -translate-x-1/2 -translate-y-1/2 transition-all ${
                      selectedCamera === camera.id ? 'scale-125 z-20' : 'z-10'
                    }`}
                    style={{ left: `${camera.lng}%`, top: `${camera.lat}%` }}
                  >
                    {/* Pulse animation for incidents */}
                    {camera.hasIncident && (
                      <div className="absolute inset-0 w-10 h-10 -translate-x-1/2 -translate-y-1/2 left-1/2 top-1/2">
                        <div className="absolute inset-0 bg-red-500 rounded-full animate-ping opacity-75" />
                      </div>
                    )}

                    {/* Camera Pin */}
<div
  className={`relative w-10 h-10 rounded-full flex items-center justify-center shadow-lg transition-all ${
    camera.hasIncident
      ? 'bg-red-500 shadow-red-500/50'
      : camera.status === 'online'
      ? 'bg-blue-500 shadow-blue-500/50'
      : 'bg-gray-500 shadow-gray-500/50'
  } ${
    selectedCamera === camera.id
      ? 'ring-4 ring-white/30'
      : 'group-hover:ring-2 ring-white/20'
  }`}
>
                      {camera.hasIncident ? (
                        <AlertCircle className="w-5 h-5 text-white" />
                      ) : (
                        <Camera className="w-5 h-5 text-white" />
                      )}
                    </div>

                    {/* Camera Label */}
                    <div
              className={`absolute top-full mt-2 left-1/2 -translate-x-1/2 whitespace-nowrap px-2 py-1 rounded bg-blue-100 text-blue-800 border border-blue-200 dark:bg-blue-900 dark:text-blue-100 dark:border-blue-700 text-xs transition-opacity ${
              selectedCamera === camera.id ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
              }`}
                >
                {camera.name}
                      </div>
                  </button>
                ))}

                {/* Legend */}
                <div className="absolute bottom-6 left-1/2 -translate-x-1/2 bg-background/90 backdrop-blur border border-border rounded-lg p-3 flex items-center gap-4 text-xs">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-primary" />
                    <span className="text-foreground">Online</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-red-500" />
                    <span className="text-foreground">Incident</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-muted-foreground" />
                    <span className="text-foreground">Offline</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Camera Details / List */}
        <div>
          <Card className="bg-card/80 backdrop-blur border-border h-[664px] flex flex-col">
            <CardHeader className="border-b border-border">
              <CardTitle className="flex items-center gap-2 text-card-foreground">
                <Eye className="w-5 h-5 text-blue-600 dark:text-blue-300" />
                {selectedCameraData ? 'Camera Details' : 'All Cameras'}
              </CardTitle>
            </CardHeader>
            <CardContent className="flex-1 p-0">
              {selectedCameraData ? (
                <div className="p-4 space-y-4">
                  <div className="space-y-3">
                    <div>
                      <p className="text-sm text-muted-foreground mb-1">Camera ID</p>
                      <p className="text-foreground">{selectedCameraData.name}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground mb-1">Location</p>
                      <p className="text-foreground">{selectedCameraData.location}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground mb-1">Zone</p>
                      <Badge variant="outline" className="border-border text-foreground">
                        {selectedCameraData.zone}
                      </Badge>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground mb-1">Status</p>
                      <Badge
                        className={
                          selectedCameraData.status === 'online'
                            ? 'bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30'
                            : 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-500/20 dark:text-gray-400 dark:border-gray-500/30'
                        }
                      >
                        <Radio className="w-3 h-3 mr-1" />
                        {selectedCameraData.status}
                      </Badge>
                    </div>
                    {selectedCameraData.hasIncident && (
                      <div>
                        <Badge variant="destructive" className="w-full justify-center">
                          <AlertCircle className="w-3 h-3 mr-1" />
                          Active Incident
                        </Badge>
                      </div>
                    )}
                  </div>

                  {/* Mock Video Feed */}
                  <div className="mt-4">
                    <p className="text-sm text-muted-foreground mb-2">Live Feed</p>
                    <div className="aspect-video bg-muted rounded-lg flex items-center justify-center border border-border">
                      <div className="text-center">
                        <Eye className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
                        <p className="text-muted-foreground text-sm">Live video stream</p>
                      </div>
                    </div>
                  </div>

                  <Button
                    variant="outline"
                    className="w-full border-border text-foreground hover:bg-muted"
                    onClick={() => setSelectedCamera(null)}
                  >
                    Back to Camera List
                  </Button>
                </div>
              ) : (
                <ScrollArea className="h-full">
                  <div className="p-4 space-y-2">
                    {cameras.map((camera) => (
                      <button
                        key={camera.id}
                        onClick={() => setSelectedCamera(camera.id)}
                        className="w-full p-3 rounded-lg bg-muted/50 border border-border hover:border-primary/50 transition-colors text-left"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-foreground">{camera.name}</span>
                          <Badge
                            className={
                              camera.hasIncident
                                ? 'bg-red-100 text-red-800 border-red-200 dark:bg-red-500/20 dark:text-red-400 dark:border-red-500/30'
                                : camera.status === 'online'
                                ? 'bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30'
                                : 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-500/20 dark:text-gray-400 dark:border-gray-500/30'
                            }
                          >
                            {camera.status}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-2 text-sm text-muted-foreground">
                          <MapPin className="w-3 h-3" />
                          <span className="truncate">{camera.location}</span>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">{camera.zone}</p>
                      </button>
                    ))}
                  </div>
                </ScrollArea>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}