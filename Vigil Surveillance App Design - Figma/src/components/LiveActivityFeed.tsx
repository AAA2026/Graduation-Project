import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { vigilClasses } from './vigil-theme';
import { ScrollArea } from './ui/scroll-area';
import {
  AlertCircle,
  Car,
  Users,
  Clock,
  MapPin,
  Eye,
  Radio,
  TrendingUp,
  Activity,
  Camera,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

interface ActivityItem {
  id: string;
  type: 'incident' | 'camera' | 'system' | 'user';
  subtype?: 'violence' | 'crash' | 'online' | 'offline';
  title: string;
  description: string;
  timestamp: Date;
  severity?: 'critical' | 'high' | 'medium' | 'low' | 'info';
  location?: string;
  cameraId?: string;
}

interface LiveActivityFeedProps {
  onIncidentClick?: (id: string) => void;
}

export function LiveActivityFeed({ onIncidentClick }: LiveActivityFeedProps) {
  const [activities, setActivities] = useState<ActivityItem[]>([
    {
      id: '1',
      type: 'incident',
      subtype: 'violence',
      title: 'Violence Detected',
      description: 'Physical altercation detected at Main St',
      timestamp: new Date(Date.now() - 2 * 60 * 1000),
      severity: 'critical',
      location: 'Main St & 5th Ave',
      cameraId: 'CAM-042',
    },
    {
      id: '2',
      type: 'incident',
      subtype: 'crash',
      title: 'Vehicle Collision',
      description: 'Traffic accident on Highway 101',
      timestamp: new Date(Date.now() - 8 * 60 * 1000),
      severity: 'high',
      location: 'Highway 101, Exit 23',
      cameraId: 'CAM-128',
    },
    {
      id: '3',
      type: 'camera',
      subtype: 'offline',
      title: 'Camera Offline',
      description: 'CAM-074 lost connection',
      timestamp: new Date(Date.now() - 15 * 60 * 1000),
      severity: 'medium',
      cameraId: 'CAM-074',
    },
    {
      id: '4',
      type: 'system',
      title: 'AI Model Updated',
      description: 'Violence detection model v2.4 deployed',
      timestamp: new Date(Date.now() - 30 * 60 * 1000),
      severity: 'info',
    },
  ]);

  const [stats, setStats] = useState({
    totalIncidents: 2,
    activeCameras: 147,
    systemStatus: 'operational',
    avgResponseTime: '2.3m',
  });

  // Simulate live activity updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Random chance to add new activity
      if (Math.random() > 0.7) {
        const newActivity = generateRandomActivity();
        setActivities((prev) => [newActivity, ...prev].slice(0, 20)); // Keep last 20
      }
    }, 15000); // Check every 15 seconds

    return () => clearInterval(interval);
  }, []);

  const generateRandomActivity = (): ActivityItem => {
    const types = ['incident', 'camera', 'system', 'user'];
    const type = types[Math.floor(Math.random() * types.length)] as ActivityItem['type'];

    const cameras = ['CAM-042', 'CAM-128', 'CAM-089', 'CAM-156', 'CAM-203', 'CAM-074'];
    const locations = [
      'Main St & 5th Ave',
      'Highway 101',
      'Park Plaza',
      'Downtown Bridge',
      'Shopping District',
    ];

    switch (type) {
      case 'incident':
        const isViolence = Math.random() > 0.5;
        return {
          id: `activity-${Date.now()}`,
          type: 'incident',
          subtype: isViolence ? 'violence' : 'crash',
          title: isViolence ? 'Violence Detected' : 'Vehicle Collision',
          description: isViolence
            ? 'Suspicious activity detected'
            : 'Traffic accident reported',
          timestamp: new Date(),
          severity: Math.random() > 0.5 ? 'critical' : 'high',
          location: locations[Math.floor(Math.random() * locations.length)],
          cameraId: cameras[Math.floor(Math.random() * cameras.length)],
        };
      case 'camera':
        const isOnline = Math.random() > 0.3;
        return {
          id: `activity-${Date.now()}`,
          type: 'camera',
          subtype: isOnline ? 'online' : 'offline',
          title: isOnline ? 'Camera Online' : 'Camera Offline',
          description: `${cameras[Math.floor(Math.random() * cameras.length)]} ${
            isOnline ? 'reconnected' : 'lost connection'
          }`,
          timestamp: new Date(),
          severity: isOnline ? 'info' : 'medium',
          cameraId: cameras[Math.floor(Math.random() * cameras.length)],
        };
      case 'system':
        return {
          id: `activity-${Date.now()}`,
          type: 'system',
          title: 'System Update',
          description: 'Performance optimization completed',
          timestamp: new Date(),
          severity: 'info',
        };
      default:
        return {
          id: `activity-${Date.now()}`,
          type: 'user',
          title: 'User Action',
          description: 'Officer reviewed incident',
          timestamp: new Date(),
          severity: 'info',
        };
    }
  };

  const getActivityIcon = (activity: ActivityItem) => {
    if (activity.type === 'incident') {
      return activity.subtype === 'violence' ? Users : Car;
    }
    if (activity.type === 'camera') {
      return Camera;
    }
    if (activity.type === 'system') {
      return Activity;
    }
    return AlertCircle;
  };

  const getSeverityColor = (severity?: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-500/20 text-red-400 border-red-500/30';
      case 'high':
        return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
      case 'medium':
        return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
      case 'low':
        return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
      default:
        return 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30';
    }
  };

  const formatTimestamp = (timestamp: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - timestamp.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return timestamp.toLocaleDateString();
  };

  return (
    <div className="space-y-4">
      {/* Stats Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Card className={vigilClasses.card}>
          <CardContent className="p-3">
            <div className="flex items-center justify-between">
              <div>
                <p className={`text-xs ${vigilClasses.textMuted}`}>Active Incidents</p>
                <p className="text-2xl text-red-400">{stats.totalIncidents}</p>
              </div>
              <AlertCircle className="w-8 h-8 text-red-400" />
            </div>
          </CardContent>
        </Card>

        <Card className={vigilClasses.card}>
          <CardContent className="p-3">
            <div className="flex items-center justify-between">
              <div>
                <p className={`text-xs ${vigilClasses.textMuted}`}>Cameras Online</p>
                <p className="text-2xl text-green-400">{stats.activeCameras}</p>
              </div>
              <Radio className="w-8 h-8 text-green-400 animate-pulse" />
            </div>
          </CardContent>
        </Card>

        <Card className={vigilClasses.card}>
          <CardContent className="p-3">
            <div className="flex items-center justify-between">
              <div>
                <p className={`text-xs ${vigilClasses.textMuted}`}>Response Time</p>
                <p className="text-2xl text-cyan-400 dark:text-cyan-400">{stats.avgResponseTime}</p>
              </div>
              <Clock className="w-8 h-8 text-cyan-400" />
            </div>
          </CardContent>
        </Card>

        <Card className={vigilClasses.card}>
          <CardContent className="p-3">
            <div className="flex items-center justify-between">
              <div>
                <p className={`text-xs ${vigilClasses.textMuted}`}>System Status</p>
                <p className="text-sm text-green-400 capitalize mt-1">{stats.systemStatus}</p>
              </div>
              <TrendingUp className="w-8 h-8 text-green-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Activity Feed */}
      <Card className={vigilClasses.card}>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Radio className="w-5 h-5 text-green-500 animate-pulse" />
              Live Activity Feed
            </CardTitle>
            <Badge className="bg-green-500/20 text-green-400 border-green-500/30">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse mr-2" />
              LIVE
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[500px] pr-4">
            <AnimatePresence mode="popLayout">
              {activities.map((activity, index) => {
                const Icon = getActivityIcon(activity);
                return (
                  <motion.div
                    key={activity.id}
                    initial={{ opacity: 0, x: -20, height: 0 }}
                    animate={{ opacity: 1, x: 0, height: 'auto' }}
                    exit={{ opacity: 0, x: 20, height: 0 }}
                    transition={{ duration: 0.3 }}
                    className={`mb-3 p-3 rounded-lg ${vigilClasses.card} border-l-4 ${
                      activity.type === 'incident'
                        ? 'border-l-red-500 bg-red-950/10'
                        : activity.type === 'camera'
                        ? 'border-l-cyan-500 bg-cyan-950/10'
                        : 'border-l-blue-500 bg-blue-950/10'
                    } cursor-pointer hover:scale-[1.02] transition-transform`}
                    onClick={() => {
                      if (activity.type === 'incident' && onIncidentClick) {
                        onIncidentClick(activity.id);
                      }
                    }}
                  >
                    <div className="flex items-start gap-3">
                      <div
                        className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${
                          activity.type === 'incident' && activity.subtype === 'violence'
                            ? 'bg-red-500/20 text-red-400'
                            : activity.type === 'incident' && activity.subtype === 'crash'
                            ? 'bg-orange-500/20 text-orange-400'
                            : activity.type === 'camera'
                            ? 'bg-cyan-500/20 text-cyan-400'
                            : 'bg-blue-500/20 text-blue-400'
                        }`}
                      >
                        <Icon className="w-5 h-5" />
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between gap-2 mb-1">
                          <h4 className={`font-medium ${vigilClasses.textPrimary}`}>
                            {activity.title}
                          </h4>
                          <Badge className={getSeverityColor(activity.severity)}>
                            {activity.severity?.toUpperCase() || 'INFO'}
                          </Badge>
                        </div>

                        <p className={`text-sm ${vigilClasses.textMuted} mb-2`}>
                          {activity.description}
                        </p>

                        <div className={`flex items-center gap-3 text-xs ${vigilClasses.textMuted} flex-wrap`}>
                          <div className="flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            <span>{formatTimestamp(activity.timestamp)}</span>
                          </div>
                          {activity.location && (
                            <div className="flex items-center gap-1">
                              <MapPin className="w-3 h-3" />
                              <span>{activity.location}</span>
                            </div>
                          )}
                          {activity.cameraId && (
                            <div className="flex items-center gap-1">
                              <Eye className="w-3 h-3" />
                              <span>{activity.cameraId}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}
