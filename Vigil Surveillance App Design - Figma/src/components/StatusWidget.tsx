import { useState, useEffect } from 'react';
import { Activity, Database, Wifi, Video, ChevronUp, ChevronDown } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

interface SystemStatus {
  apiStatus: 'healthy' | 'degraded' | 'down';
  apiResponseTime: number;
  databaseStatus: 'connected' | 'disconnected';
  systemUptime: string;
  activeCameras: number;
  totalCameras: number;
}

export default function StatusWidget() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [status, setStatus] = useState<SystemStatus>({
    apiStatus: 'healthy',
    apiResponseTime: 0,
    databaseStatus: 'connected',
    systemUptime: '0h 0m',
    activeCameras: 0,
    totalCameras: 6,
  });

  // Simulate system status (no backend)
  useEffect(() => {
    const startTime = Date.now();

    const updateStatus = () => {
      const uptimeMs = Date.now() - startTime;
      const hours = Math.floor(uptimeMs / (1000 * 60 * 60));
      const minutes = Math.floor((uptimeMs % (1000 * 60 * 60)) / (1000 * 60));
      
      // Simulate varying response times
      const responseTime = Math.floor(Math.random() * 300) + 100; // 100-400ms
      
      // Simulate camera count
      const activeCameras = Math.floor(Math.random() * 2) + 5; // 5-6 active cameras

      setStatus({
        apiStatus: responseTime < 500 ? 'healthy' : responseTime < 1500 ? 'degraded' : 'down',
        apiResponseTime: responseTime,
        databaseStatus: 'connected',
        systemUptime: `${hours}h ${minutes}m`,
        activeCameras,
        totalCameras: 6,
      });
    };

    // Initial update
    updateStatus();

    // Update every 5 seconds
    const interval = setInterval(updateStatus, 5000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = () => {
    switch (status.apiStatus) {
      case 'healthy':
        return 'text-green-500';
      case 'degraded':
        return 'text-yellow-500';
      case 'down':
        return 'text-red-500';
      default:
        return 'text-gray-500';
    }
  };

  const getStatusDot = () => {
    switch (status.apiStatus) {
      case 'healthy':
        return 'bg-green-500';
      case 'degraded':
        return 'bg-yellow-500';
      case 'down':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="fixed bottom-6 right-6 z-50"
    >
      <div className="bg-[var(--vigil-card-bg)] border border-[var(--vigil-border)] rounded-lg shadow-lg overflow-hidden backdrop-blur-sm bg-opacity-95">
        {/* Header - Always Visible */}
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full px-4 py-3 flex items-center justify-between hover:bg-[var(--vigil-border)] transition-colors"
        >
          <div className="flex items-center space-x-3">
            <Activity className={`w-5 h-5 ${getStatusColor()}`} />
            <div className="flex items-center space-x-2">
              <span className="text-sm font-medium text-gray-900 dark:text-white">System Status</span>
              <div className={`w-2 h-2 rounded-full ${getStatusDot()} animate-pulse`}></div>
            </div>
          </div>
          {isExpanded ? (
            <ChevronDown className="w-4 h-4 text-[var(--vigil-text-secondary)]" />
          ) : (
            <ChevronUp className="w-4 h-4 text-[var(--vigil-text-secondary)]" />
          )}
        </button>

        {/* Expanded Details */}
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <div className="px-4 py-3 space-y-3 border-t border-[var(--vigil-border)]">
                {/* API Status */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Wifi className="w-4 h-4 text-[var(--vigil-text-secondary)]" />
                    <span className="text-sm text-[var(--vigil-text-secondary)]">API</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`text-sm font-medium ${getStatusColor()}`}>
                      {status.apiStatus}
                    </span>
                    <span className="text-xs text-[var(--vigil-text-muted)]">
                      {status.apiResponseTime}ms
                    </span>
                  </div>
                </div>

                {/* Database Status */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Database className="w-4 h-4 text-[var(--vigil-text-secondary)]" />
                    <span className="text-sm text-[var(--vigil-text-secondary)]">Database</span>
                  </div>
                  <span className={`text-sm font-medium ${status.databaseStatus === 'connected' ? 'text-green-500' : 'text-red-500'}`}>
                    {status.databaseStatus}
                  </span>
                </div>

                {/* Active Cameras */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Video className="w-4 h-4 text-[var(--vigil-text-secondary)]" />
                    <span className="text-sm text-[var(--vigil-text-secondary)]">Cameras</span>
                  </div>
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {status.activeCameras}/{status.totalCameras}
                  </span>
                </div>

                {/* System Uptime */}
                <div className="flex items-center justify-between pt-2 border-t border-[var(--vigil-border)]">
                  <span className="text-xs text-[var(--vigil-text-muted)]">Uptime</span>
                  <span className="text-xs font-medium text-[var(--vigil-text-secondary)]">
                    {status.systemUptime}
                  </span>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}

// Also export as named export for flexibility
export { StatusWidget };