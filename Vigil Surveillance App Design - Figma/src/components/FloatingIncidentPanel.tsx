import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { ScrollArea } from './ui/scroll-area';
import { vigilClasses } from './vigil-theme';
import {
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Clock,
  MapPin,
  Eye,
  CheckCircle,
  XCircle,
  Navigation,
  Phone,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

interface FloatingIncidentPanelProps {
  role: 'admin' | 'officer' | 'security-authority';
}

export function FloatingIncidentPanel({ role }: FloatingIncidentPanelProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [isMinimized, setIsMinimized] = useState(false);
  // Use real incidents from backend
  // Import the hook at the top: import { useRealtimeIncidents } from '../hooks/useRealtimeIncidents';
  // (Assume import is added if not present)
  // Use the hook:
  const { incidents } = require('../hooks/useRealtimeIncidents').useRealtimeIncidents();
  // If incidents have no 'time' field, compute a relative time from timestamp
  const getRelativeTime = (timestamp: string) => {
    const now = Date.now();
    const ts = Date.parse(timestamp);
    const diff = Math.floor((now - ts) / 1000);
    if (isNaN(ts)) return '';
    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return new Date(ts).toLocaleString();
  };

  const activeIncidents = incidents.filter((i) => i.status === 'active');

  if (isMinimized) {
    return (
      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="fixed bottom-4 right-4 z-50"
      >
        <Button
          onClick={() => setIsMinimized(false)}
          className="bg-red-600 hover:bg-red-700 text-white shadow-lg shadow-red-600/50 relative"
          size="lg"
        >
          <AlertCircle className="w-5 h-5 mr-2" />
          {activeIncidents.length} Active Alerts
          {activeIncidents.length > 0 && (
            <span className="absolute -top-1 -right-1 w-3 h-3 bg-amber-500 rounded-full animate-ping" />
          )}
        </Button>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="space-y-3 sm:space-y-4"
    >
      <Card className={`${vigilClasses.card} border-red-500/30 shadow-lg`}>
        <CardHeader className="pb-3 cursor-pointer" onClick={() => setIsExpanded(!isExpanded)}>
          <div className="flex items-center justify-between">
            <CardTitle className={`flex items-center gap-2 text-base sm:text-lg ${vigilClasses.textPrimary}`}>
              <AlertCircle className="w-5 h-5 text-red-500 animate-pulse" />
              Active Incidents
              <Badge className="bg-red-500/20 text-red-600 dark:text-red-400 border-red-500/30">
                {activeIncidents.length}
              </Badge>
            </CardTitle>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  setIsMinimized(true);
                }}
                className="h-8 w-8 p-0"
              >
                <XCircle className="w-4 h-4" />
              </Button>
              <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                {isExpanded ? (
                  <ChevronUp className="w-4 h-4" />
                ) : (
                  <ChevronDown className="w-4 h-4" />
                )}
              </Button>
            </div>
          </div>
        </CardHeader>

        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <CardContent className="pt-0">
                <ScrollArea className="h-[300px] sm:h-[400px] pr-4">
                  <div className="space-y-3">
                    {incidents.map((incident) => (
                      <motion.div
                        key={incident.id}
                        initial={{ x: -20, opacity: 0 }}
                        animate={{ x: 0, opacity: 1 }}
                        className={`p-3 rounded-lg border ${
                          incident.status === 'active'
                            ? 'bg-red-500/5 border-red-500/30 dark:bg-red-950/20'
                            : incident.status === 'investigating'
                            ? 'bg-amber-500/5 border-amber-500/30 dark:bg-amber-950/20'
                            : 'bg-green-500/5 border-green-500/30 dark:bg-green-950/20'
                        } hover:shadow-md transition-all duration-200`}
                      >
                        {/* Header */}
                        <div className="flex items-start justify-between gap-2 mb-2">
                          <div className="flex items-center gap-2">
                            <div
                              className={`w-2 h-2 rounded-full flex-shrink-0 ${
                                incident.severity === 'critical'
                                  ? 'bg-red-500 animate-pulse'
                                  : incident.severity === 'high'
                                  ? 'bg-red-400'
                                  : 'bg-amber-400'
                              }`}
                            />
                            <span className={`text-sm ${vigilClasses.textPrimary}`}>
                              {incident.id}
                            </span>
                          </div>
                          <Badge
                            variant={
                              incident.status === 'active'
                                ? 'destructive'
                                : incident.status === 'investigating'
                                ? 'default'
                                : 'secondary'
                            }
                            className="text-xs"
                          >
                            {incident.status}
                          </Badge>
                        </div>

                        {/* Type and Confidence */}
                        <div className="flex items-center gap-2 mb-2">
                          <Badge
                            className={`${
                              incident.type === 'Violence'
                                ? 'bg-red-500/20 text-red-600 dark:text-red-400 border-red-500/30'
                                : 'bg-amber-500/20 text-amber-600 dark:text-amber-400 border-amber-500/30'
                            }`}
                          >
                            {incident.type}
                          </Badge>
                          <span className={`text-xs ${vigilClasses.textMuted}`}>
                            {incident.confidence}% confidence
                          </span>
                        </div>

                        {/* Description */}
                        <p className={`text-sm ${vigilClasses.textSecondary} mb-2`}>
                          {incident.description}
                        </p>

                        {/* Location and Time */}
                        <div className="space-y-1 mb-3">
                          <div className="flex items-center gap-1 text-xs text-gray-600 dark:text-gray-400">
                            <MapPin className="w-3 h-3 flex-shrink-0" />
                            <span>{incident.location}</span>
                            <span className="text-cyan-500">({incident.cameraId})</span>
                          </div>
                          <div className="flex items-center gap-1 text-xs text-gray-600 dark:text-gray-400">
                            <Clock className="w-3 h-3 flex-shrink-0" />
                            <span>{getRelativeTime(incident.timestamp)}</span>
                          </div>
                        </div>

                        {/* Actions */}
                        {incident.status === 'active' && (
                          <div className="flex gap-2">
                            <Button
                              size="sm"
                              className="flex-1 bg-cyan-600 hover:bg-cyan-700 text-white h-8 text-xs"
                            >
                              <Eye className="w-3 h-3 mr-1" />
                              View Feed
                            </Button>
                            {role === 'officer' && (
                              <>
                                <Button
                                  size="sm"
                                  variant="outline"
                                  className="h-8 text-xs"
                                >
                                  <Navigation className="w-3 h-3 mr-1" />
                                  Navigate
                                </Button>
                                <Button
                                  size="sm"
                                  variant="outline"
                                  className="h-8 text-xs"
                                >
                                  <Phone className="w-3 h-3 mr-1" />
                                  Dispatch
                                </Button>
                              </>
                            )}
                            {role === 'admin' && (
                              <Button
                                size="sm"
                                variant="outline"
                                className="h-8 text-xs"
                              >
                                <CheckCircle className="w-3 h-3 mr-1" />
                                Resolve
                              </Button>
                            )}
                          </div>
                        )}
                      </motion.div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>
    </motion.div>
  );
}
