import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { vigilClasses } from './vigil-theme';
import { DVRCameraGrid } from './DVRCameraGrid';
import { FloatingIncidentPanel } from './FloatingIncidentPanel';
import { CompactStatsBar } from './CompactStatsBar';
import { useState } from 'react';
import {
  Shield,
  Eye,
  MapPin,
  Navigation,
  Phone,
  Radio,
  Users,
  Clock,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

export function SecurityAuthorityDashboard() {
  const [showStats, setShowStats] = useState(false);
  const [showIncidents, setShowIncidents] = useState(false);

  const quickActions = [
    {
      label: 'View All Zones',
      icon: MapPin,
      color: 'bg-cyan-600 hover:bg-cyan-700',
      count: null,
    },
    {
      label: 'Emergency Contacts',
      icon: Phone,
      color: 'bg-red-600 hover:bg-red-700',
      count: null,
    },
    {
      label: 'Dispatch Units',
      icon: Navigation,
      color: 'bg-amber-600 hover:bg-amber-700',
      count: 3,
    },
    {
      label: 'Active Teams',
      icon: Users,
      color: 'bg-blue-600 hover:bg-blue-700',
      count: 8,
    },
  ];

  return (
    <div className="space-y-3">
      {/* Minimal Top Bar */}
      <motion.div
        initial={{ y: -10, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="flex items-center justify-between bg-gradient-to-r from-cyan-600/10 to-purple-600/10 border border-cyan-500/20 rounded-lg p-2 sm:p-3"
      >
        <div className="flex items-center gap-2 sm:gap-3">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span className={`text-sm sm:text-base ${vigilClasses.textPrimary}`}>
              SECURITY COMMAND
            </span>
          </div>
          <Badge className="bg-green-500/20 text-green-600 dark:text-green-400 border-green-500/30 text-xs">
            <Radio className="w-3 h-3 mr-1 animate-pulse" />
            ALL SYSTEMS ACTIVE
          </Badge>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowStats(!showStats)}
            className={`${vigilClasses.button} h-7 sm:h-8 text-xs`}
          >
            {showStats ? <ChevronUp className="w-3 h-3 mr-1" /> : <ChevronDown className="w-3 h-3 mr-1" />}
            Stats
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowIncidents(!showIncidents)}
            className={`${vigilClasses.button} h-7 sm:h-8 text-xs`}
          >
            {showIncidents ? <ChevronUp className="w-3 h-3 mr-1" /> : <ChevronDown className="w-3 h-3 mr-1" />}
            Incidents
          </Button>
        </div>
      </motion.div>

      {/* Collapsible Stats */}
      <AnimatePresence>
        {showStats && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <CompactStatsBar role="security-authority" />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Quick Action Buttons - Compact */}
      <motion.div
        initial={{ y: 10, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.1 }}
        className="grid grid-cols-4 gap-2"
      >
        {quickActions.map((action, index) => {
          const Icon = action.icon;
          return (
            <Button
              key={action.label}
              className={`${action.color} text-white h-auto py-2 flex flex-col items-center gap-1 relative text-xs`}
            >
              <Icon className="w-4 h-4" />
              <span className="hidden sm:inline text-xs">{action.label}</span>
              {action.count !== null && (
                <Badge className="absolute -top-1 -right-1 bg-white text-gray-900 border-0 h-4 w-4 p-0 flex items-center justify-center text-xs">
                  {action.count}
                </Badge>
              )}
            </Button>
          );
        })}
      </motion.div>

      {/* DVR Camera Grid */}
      <div className={showIncidents ? 'grid grid-cols-1 xl:grid-cols-3 gap-3' : ''}>
        <div className={showIncidents ? 'xl:col-span-2' : ''}>
          <DVRCameraGrid role="security-authority" />
        </div>

        {/* Collapsible Incidents Panel */}
        <AnimatePresence>
          {showIncidents && (
            <motion.div
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: 20, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="xl:col-span-1"
            >
              <FloatingIncidentPanel role="security-authority" />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}