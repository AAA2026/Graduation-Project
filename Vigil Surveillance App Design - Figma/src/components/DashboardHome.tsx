import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { vigilClasses } from './vigil-theme';
import { DVRCameraGrid } from './DVRCameraGrid';
import { FloatingIncidentPanel } from './FloatingIncidentPanel';
import { CompactStatsBar } from './CompactStatsBar';
import { useState } from 'react';
import {
  Camera,
  AlertCircle,
  Bell,
  Clock,
  TrendingUp,
  TrendingDown,
  Eye,
  CheckCircle,
  Users,
  Activity,
  UserCheck,
  Shield,
  Zap,
  ChevronDown,
  ChevronUp,
  Radio,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { Button } from './ui/button';

interface DashboardHomeProps {
  role: 'admin' | 'officer';
}

export function DashboardHome({ role }: DashboardHomeProps) {
  const [showStats, setShowStats] = useState(false);
  const [showIncidents, setShowIncidents] = useState(false);

  return (
    <div className="space-y-3">
      {/* Minimal Top Bar - Just Status */}
      <motion.div
        initial={{ y: -10, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="flex items-center justify-between bg-gradient-to-r from-cyan-600/10 to-purple-600/10 border border-cyan-500/20 rounded-lg p-2 sm:p-3"
      >
        <div className="flex items-center gap-2 sm:gap-3">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span className={`text-sm sm:text-base ${vigilClasses.textPrimary}`}>
              {role === 'admin' ? 'COMMAND CENTER' : 'SURVEILLANCE CONTROL'}
            </span>
          </div>
          <Badge className="bg-green-500/20 text-green-600 dark:text-green-400 border-green-500/30 text-xs">
            <Radio className="w-3 h-3 mr-1 animate-pulse" />
            SYSTEM ACTIVE
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
          <span className={`text-xs ${vigilClasses.textSecondary} hidden sm:inline`}>
            {new Date().toLocaleTimeString('en-US', {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </span>
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
            <CompactStatsBar role={role} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* DVR-Style Camera Grid - FULL SCREEN */}
      <div className={showIncidents ? 'grid grid-cols-1 xl:grid-cols-3 gap-3' : ''}>
        <div className={showIncidents ? 'xl:col-span-2' : ''}>
          <DVRCameraGrid role={role} />
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
              <FloatingIncidentPanel role={role} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}