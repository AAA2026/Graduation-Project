import { useEffect, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Card } from './ui/card';
import { AnimatedBackgroundBeams } from './AnimatedBackgroundBeams';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import {
  Camera,
  AlertCircle,
  MapPin,
  BarChart3,
  Settings,
  Users,
  Activity,
  Cpu,
  Brain,
  Bell,
  FileText,
  Radio,
  LogOut,
  Volume2,
  VolumeX,
  Sun,
  Moon,
  ChevronDown,
  Wifi,
  WifiOff,
  CalendarDays,
  UserCheck,
  Shield
} from 'lucide-react';
import { useTheme } from './ThemeProvider';
import { useRealtimeIncidents } from '../hooks/useRealtimeIncidents';
import { useLiveStatus } from '../hooks/useLiveStatus';
import { vigilClasses } from './vigil-theme';
import { VigilLogo } from './VigilLogo';
import { DVRCameraGrid } from './DVRCameraGrid';
import { IncidentsView } from './IncidentsView';
import { DispatchesView } from './DispatchesView';
import { AnalyticsView } from './AnalyticsView';
import { MapView } from './MapView';
import { CameraManagement } from './CameraManagement';
import { UserManagement } from './UserManagement';
import { SystemHealthView } from './SystemHealthView';
import { AIModelManagement } from './AIModelManagement';
import { AIFeedbackSection } from './AIFeedbackSection';
import { NotificationCenter } from './NotificationCenter';
import { OfficerReports } from './OfficerReports';
import { AdminReports } from './AdminReports';

import { DemoRequestsTab } from './DemoRequestsTab';

import { IncidentDetailModal } from './IncidentDetailModal';
import { StatusWidget } from './StatusWidget';

interface ModernSecurityLayoutProps {
  role: 'admin' | 'officer' | 'security';
  onLogout: () => void;
  onNavigate: (view: string) => void;
  currentView: string;
  userName: string;
}

export function ModernSecurityLayout({
  role,
  onLogout,
  onNavigate,
  currentView,
  userName,
}: ModernSecurityLayoutProps) {
  const [showQuickPanel, setShowQuickPanel] = useState(false);
  const { theme, toggleTheme } = useTheme();

  console.log("ModernSecurityLayout role:", role);

  // Use live status polling hook (only for admin and officer)
  // Only Admin can see/toggle Live Status, Officer sees read-only via effectiveOffline logic
  const shouldShowLiveStatus = role === 'admin';

  const {
    systemStatus,
    isPolling,
    togglePolling,
    lastUpdate,
    cameraStatuses,
    offlineMode,
    toggleOfflineMode,
    refreshStatus,
  } = useLiveStatus();

  // Use realtime incidents hook
  const [pendingIncidentId, setPendingIncidentId] = useState<string | null>(null);

  const handleIncidentReceived = useCallback((incidentId: string) => {
    setPendingIncidentId(incidentId);
    onNavigate('incidents');
  }, [onNavigate]);

  const {
    activeIncidents,
    incidents,
    resolveIncident,
    dismissIncident,
    soundEnabled,
    toggleSound,
    dismissAllIncidents,
    confirmIncident,
    rejectIncident,
    dispatchIncident,
  } = useRealtimeIncidents(handleIncidentReceived, offlineMode, userName);

  // When pendingIncidentId is set and incidents are loaded, open modal for that incident
  useEffect(() => {
    if (pendingIncidentId && incidents.length > 0) {
      setSelectedIncidentId(pendingIncidentId);
      setIsModalOpen(true);
      setPendingIncidentId(null);
    }
  }, [pendingIncidentId, incidents]);

  // Loading state for offline toggle
  const [offlineLoading, setOfflineLoading] = useState(false);
  // Toast for error
  const [offlineError, setOfflineError] = useState<string | null>(null);

  // Patch toggleOfflineMode to show loading and error, and always force refresh
  const handleToggleOffline = async () => {
    if (offlineLoading) return;
    setOfflineLoading(true);
    setOfflineError(null);
    try {
      await toggleOfflineMode();
      // Wait a bit for backend state to propagate
      setTimeout(async () => {
        await refreshStatus();
      }, 500);
    } catch (e) {
      console.error("Toggle offline failed:", e);
      setOfflineError('Failed to toggle offline mode. Backend unreachable.');
    } finally {
      setOfflineLoading(false);
    }
  }

  // Calculate online cameras count
  const onlineCamerasCount = Array.from(cameraStatuses.values()).filter((c: any) => c.status === 'online').length;
  const totalCameras = cameraStatuses.size || 12;

  // Backend status logic
  const backendOnline = systemStatus === 'operational';

  // EFFECTIVE OFFLINE CALCULATION
  // If backend is down, we are effectively offline.
  // If backend is up, use the reported offlineMode state.
  const effectiveOffline = !backendOnline || offlineMode;

  // Show LIVE only if backend is fully operational and NOT in offline mode
  const showLive = backendOnline && !offlineMode;

  // Sync offlineMode UI if backend goes down
  useEffect(() => {
    if (!backendOnline && !offlineMode) {
      // If backend unreachable, force offlineMode true for UI
      setTimeout(() => {
        if (!offlineMode) {
          // Only update if not already offline
          // This is UI only, does not POST to backend
        }
      }, 500);
    }
  }, [backendOnline, offlineMode]);

  const [selectedIncidentId, setSelectedIncidentId] = useState<string | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Only count truly "active" (unhandled) incidents for alerts
  const unhandledIncidents = activeIncidents.filter((i: any) => i.status === 'active');
  const activeAlerts = unhandledIncidents.length;

  // Filter incidents for "My Dispatches" tab
  // Split into Active (assigned, acknowledged, confirmed) vs Resolved (resolved by me)
  const myActiveDispatches = incidents.filter((inc: any) =>
    (inc.status === 'dispatched' || inc.status === 'acknowledged' || inc.status === 'confirmed') &&
    (inc.dispatchedOfficerName === userName ||
      inc.assigned_security === userName ||
      inc.ack_by === userName ||
      (inc.dispatched_to && inc.dispatched_to.includes(userName)))
  );

  const myResolvedDispatches = incidents.filter((inc: any) =>
    inc.status === 'resolved' &&
    (inc.dispatchedOfficerName === userName ||
      inc.assigned_security === userName ||
      inc.ack_by === userName)
  );

  // Filter incidents for "Active Dispatches" tab (Admin/Officer)
  // Show ALL incidents that are dispatched, acknowledged OR confirmed (being worked on)
  const activeDispatches = incidents.filter((inc: any) =>
    inc.status === 'dispatched' || inc.status === 'acknowledged' || inc.status === 'confirmed'
  );

  const handleSelfDispatch = (id: string) => {
    dispatchIncident(id, userName);
  };

  const handleIncidentClick = (id: string) => {
    setSelectedIncidentId(id);
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setSelectedIncidentId(null);
  };

  const handleResolve = (id: string) => {
    resolveIncident(id);
    handleCloseModal();
  };

  const handleDismiss = (id: string) => {
    dismissIncident(id);
    handleCloseModal();
  };

  const selectedIncident = incidents.find((inc: any) => inc.id === selectedIncidentId) || null;

  // Navigation items based on role
  const getNavItems = () => {
    const adminItems = [
      { id: 'live', label: 'Live Monitor', icon: Camera, badge: null },
      { id: 'incidents', label: 'Incidents', icon: AlertCircle, badge: activeAlerts },
      { id: 'active-dispatches', label: 'Active Dispatches', icon: Shield, badge: activeDispatches.length },
      { id: 'map', label: 'Map View', icon: MapPin, badge: null },
      { id: 'analytics', label: 'Analytics', icon: BarChart3, badge: null },
      { id: 'cameras', label: 'Cameras', icon: Settings, badge: null },
      { id: 'users', label: 'Users', icon: Users, badge: null },
      { id: 'demos', label: 'Demo Requests', icon: CalendarDays, badge: null },
      { id: 'reports', label: 'Reports', icon: FileText, badge: null },
      { id: 'health', label: 'System Health', icon: Activity, badge: null },
      { id: 'models', label: 'AI Models', icon: Cpu, badge: null },
      { id: 'feedback', label: 'AI Feedback', icon: Brain, badge: null },
    ];

    const officerItems = [
      { id: 'live', label: 'Live Monitor', icon: Camera, badge: null },
      { id: 'incidents', label: 'Active Incidents', icon: AlertCircle, badge: activeAlerts },
      { id: 'active-dispatches', label: 'Active Dispatches', icon: Shield, badge: activeDispatches.length },
      { id: 'map', label: 'Map View', icon: MapPin, badge: null },
      { id: 'analytics', label: 'Analytics', icon: BarChart3, badge: null },
      { id: 'reports', label: 'Reports', icon: FileText, badge: null },
    ];

    const authorityItems = [
      { id: 'live', label: 'Live Monitor', icon: Camera, badge: null },
      { id: 'incidents', label: 'Incidents', icon: AlertCircle, badge: activeAlerts },
      { id: 'my-dispatches', label: 'My Dispatches', icon: Shield, badge: myActiveDispatches.length },
      { id: 'map', label: 'Map View', icon: MapPin, badge: null },
    ];

    if (role === 'admin') return adminItems;
    if (role === 'officer') return officerItems;
    return authorityItems;
  };

  const navItems = getNavItems();

  const getRoleTitle = () => {
    if (role === 'admin') return 'Command Center';
    if (role === 'officer') return 'Officer Station';
    return 'Security Command';
  };

  const stats = [
    { label: 'Cameras Online', value: `${onlineCamerasCount}/${totalCameras}`, status: 'good' },
    { label: 'Active Alerts', value: activeAlerts.toString(), status: activeAlerts > 0 ? 'alert' : 'good' },
    { label: 'Response Time', value: '2.3m', status: 'good' },
    { label: 'AI Detection', value: '98.7%', status: 'good' },
  ];

  return (
    <AnimatedBackgroundBeams mode={theme === 'dark' ? 'dark' : 'light'}>
      <div className={`min-h-screen bg-transparent relative overflow-hidden`}>
        {/* Content Layer - needs z-index to be above animated background */}
        <div className="relative z-10">
          {/* Top Header - Professional Command Center Style */}
          <header className={`border-b ${vigilClasses.border} bg-white/80 dark:bg-black/40 backdrop-blur-xl sticky top-0 z-50`}>
            <div className="px-4 sm:px-6 py-3">
              <div className="flex items-center justify-between">
                {/* Left: Logo and Title */}
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-3">
                    <VigilLogo width={120} height={40} theme={theme} />
                    <div className="hidden sm:block border-l border-gray-300 dark:border-gray-700 pl-3">
                      <p className={`text-xs ${vigilClasses.textMuted}`}>{getRoleTitle()}</p>
                    </div>
                  </div>

                  {/* System Status */}
                  {shouldShowLiveStatus ? (
                    <div className="hidden md:flex items-center gap-3 ml-6">
                      {/* Live/Offline Toggle Button - Connected to Backend */}
                      <Button
                        onClick={backendOnline && !offlineLoading ? handleToggleOffline : undefined}
                        size="sm"
                        variant={showLive ? 'default' : 'outline'}
                        className={`h-8 px-3 text-xs ${showLive
                          ? 'bg-green-600 hover:bg-green-700 text-gray-900 dark:text-white shadow-lg'
                          : 'border-gray-500/50 text-gray-400 hover:bg-gray-700/50'
                          } ${offlineLoading ? 'opacity-60 cursor-wait' : ''}`}
                        title={showLive
                          ? `LIVE - Backend is active`
                          : backendOnline
                            ? 'OFFLINE - Backend is paused'
                            : 'OFFLINE - Backend not reachable'}
                        disabled={!backendOnline || offlineLoading}
                        aria-pressed={showLive}
                        aria-label={showLive ? 'Set system OFFLINE' : 'Set system LIVE'}
                      >
                        {offlineLoading ? (
                          <span className="flex items-center"><span className="loader mr-2 w-3 h-3 border-2 border-t-2 border-gray-400 rounded-full animate-spin" />Toggling...</span>
                        ) : !effectiveOffline ? (
                          <>
                            <Wifi className="w-3.5 h-3.5 mr-1.5 animate-pulse" />
                            LIVE
                          </>
                        ) : (
                          <>
                            <WifiOff className="w-3.5 h-3.5 mr-1.5" />
                            OFFLINE
                          </>
                        )}
                      </Button>
                      {/* Error toast for offline toggle */}
                      {offlineError && (
                        <div className="ml-2 text-xs text-red-600 bg-red-100 rounded px-2 py-1 shadow">
                          {offlineError}
                        </div>
                      )}
                      <div className="flex flex-col gap-0.5">
                        <span className="text-xs font-semibold text-gray-900 dark:text-white">
                          {onlineCamerasCount}/{totalCameras} Cameras Online
                        </span>
                        {/* System: Operational/Offline removed, toggle already shows LIVE/OFFLINE */}
                        <span className="text-xs text-gray-700 dark:text-gray-400">
                          Last Update: <span className="font-mono">{lastUpdate.slice(11, 19)}</span>
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div className="hidden md:flex items-center gap-2 ml-6">
                      {effectiveOffline ? (
                        <>
                          <div className="w-2 h-2 bg-red-500 rounded-full" />
                          <span className="text-sm text-red-600 dark:text-red-400 font-semibold uppercase tracking-wider">System Offline</span>
                        </>
                      ) : (
                        <>
                          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                          <span className="text-sm text-green-700 dark:text-green-400 font-semibold">All Systems Active</span>
                        </>
                      )}
                    </div>
                  )}
                </div>

                {/* Right: Quick Actions and User */}
                <div className="flex items-center gap-3">
                  {/* Sound Toggle */}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={toggleSound}
                    className="relative"
                    title={soundEnabled ? 'Mute alert sounds' : 'Enable alert sounds'}
                  >
                    {soundEnabled ? (
                      <Volume2 className="w-5 h-5 text-green-400" />
                    ) : (
                      <VolumeX className="w-5 h-5 text-gray-400" />
                    )}
                  </Button>

                  {/* Theme Toggle */}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={toggleTheme}
                    className="relative"
                    title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
                  >
                    {theme === 'dark' ? (
                      <Sun className="w-5 h-5 text-amber-400" />
                    ) : (
                      <Moon className="w-5 h-5 text-slate-600" />
                    )}
                  </Button>

                  {/* Alerts */}
                  <Button
                    variant="ghost"
                    size="sm"
                    className="relative"
                    onClick={() => onNavigate('incidents')}
                  >
                    <Bell className="w-5 h-5" />
                    {activeAlerts > 0 && (
                      <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center animate-pulse">
                        {activeAlerts}
                      </span>
                    )}
                  </Button>

                  {/* Quick Panel Toggle */}
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowQuickPanel(!showQuickPanel)}
                  >
                    <Activity className="w-4 h-4 mr-2" />
                    Quick Stats
                    <ChevronDown className="w-3 h-3 ml-1" />
                  </Button>

                  {/* User Menu */}
                  <div className={`flex items-center gap-2 border-l ${vigilClasses.border} pl-3`}>
                    <div className="text-right hidden sm:block">
                      <p className={`text-sm ${vigilClasses.textPrimary}`}>{userName}</p>
                      <p className={`text-xs ${vigilClasses.textMuted} capitalize`}>{role.replace('-', ' ')}</p>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={onLogout}
                      className="text-red-400 hover:text-red-300 hover:bg-red-950/30 dark:hover:bg-red-950/30"
                    >
                      <LogOut className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </div>

              {/* Quick Stats Panel - Collapsible */}
              <AnimatePresence>
                {showQuickPanel && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="overflow-hidden mt-4"
                  >
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pb-2">
                      <div className="bg-white dark:bg-[#13182b] border border-white dark:border-gray-800 rounded-lg shadow-lg p-4 flex flex-col gap-2">
                        <p className="text-xs font-extrabold text-black dark:text-white mb-1">System Status</p>
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-extrabold text-black dark:text-cyan-300">API</span>
                          <span className="text-xs px-2 py-0.5 rounded bg-green-400 text-black dark:bg-green-700/30 dark:text-green-300 font-extrabold">healthy</span>
                          <span className="text-xs text-black dark:text-gray-300 ml-2 font-extrabold">{backendOnline ? '177ms' : '--'}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-extrabold text-black dark:text-cyan-300">Database</span>
                          <span className="text-xs px-2 py-0.5 rounded bg-green-400 text-black dark:bg-green-700/30 dark:text-green-300 font-extrabold">connected</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-extrabold text-black dark:text-cyan-300">Cameras</span>
                          <span className="text-xs px-2 py-0.5 rounded bg-blue-400 text-black dark:bg-blue-700/30 dark:text-blue-300 font-extrabold">{onlineCamerasCount}/{totalCameras}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-extrabold text-black dark:text-cyan-300">Uptime</span>
                          <span className="text-xs px-2 py-0.5 rounded bg-gray-400 text-black dark:bg-gray-700/30 dark:text-gray-200 font-extrabold">0h 12m</span>
                        </div>
                      </div>
                      <div className={`${vigilClasses.card} p-4 flex flex-col gap-2`}>
                        <p className="text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">Polling</p>
                        <span className={`text-lg font-bold ${isPolling ? 'text-green-600 dark:text-green-400' : 'text-red-500'}`}>{isPolling ? 'Enabled' : 'Paused'}</span>
                        <p className="text-xs text-gray-500 dark:text-gray-400">System automatically refreshes every 3s</p>
                        <p className="text-xs text-gray-500 dark:text-gray-400">Last Update: <span className="font-mono">{lastUpdate.slice(11, 19)}</span></p>
                      </div>
                      <div className={`${vigilClasses.card} p-4 flex flex-col gap-2`}>
                        <p className="text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">Offline Mode</p>
                        <span className={`text-lg font-bold ${effectiveOffline ? 'text-red-500' : 'text-green-600 dark:text-green-400'}`}>{effectiveOffline ? 'ON' : 'OFF'}</span>
                        <p className="text-xs text-gray-500 dark:text-gray-400">When ON, no incidents or notifications are created.</p>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Horizontal Navigation Tabs */}
            <div className={`border-t ${vigilClasses.border} bg-gray-100/50 dark:bg-black/20`}>
              <div className="px-4 sm:px-6">
                <div className="flex items-center gap-1 overflow-x-auto scrollbar-hide py-2">
                  {navItems.map((item) => {
                    const Icon = item.icon;
                    const isActive = currentView === item.id;
                    return (
                      <button
                        key={item.id}
                        onClick={() => onNavigate(item.id)}
                        className={`relative flex items-center gap-2 px-4 py-2 rounded-lg whitespace-nowrap transition-all ${isActive
                          ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-gray-900 dark:text-white shadow-lg shadow-cyan-500/30'
                          : `${vigilClasses.textSecondary} ${vigilClasses.bgCardHover}`
                          }`}
                      >
                        <Icon className="w-4 h-4" />
                        <span className="text-sm font-medium">{item.label}</span>
                        {item.badge !== null && item.badge > 0 && (
                          <Badge className="bg-red-500 text-white border-0 h-5 px-1.5 ml-1 animate-pulse">
                            {item.badge}
                          </Badge>
                        )}
                        {isActive && (
                          <motion.div
                            layoutId="activeTab"
                            className="absolute bottom-0 left-0 right-0 h-0.5 bg-cyan-400"
                          />
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          </header>

          {/* Main Content Area */}
          <main className="p-4 sm:p-6 overflow-y-auto max-h-[calc(100vh-140px)]">
            {currentView === 'live' && <DVRCameraGrid role={role} />}
            {currentView === 'incidents' && (
              <IncidentsView
                role={role}
                incidents={incidents}
                onIncidentClick={handleIncidentClick}
                onResolve={resolveIncident}
                onDismiss={dismissIncident}
                soundEnabled={soundEnabled}
                onToggleSound={toggleSound}
                onSelfDispatch={handleSelfDispatch}
                onDispatch={dispatchIncident}
                onDismissAll={dismissAllIncidents}
              />
            )}
            {currentView === 'active-dispatches' && (
              <DispatchesView
                role={role}
                title="Active Missions"
                incidents={activeDispatches}
                onResolve={resolveIncident}
                onDismiss={dismissIncident}
                onConfirm={confirmIncident}
                onReject={rejectIncident}
                onIncidentClick={handleIncidentClick}
                onDismissAll={dismissAllIncidents}
              />
            )}
            {currentView === 'my-dispatches' && (
              <DispatchesView
                role={role}
                title="My Active Missions"
                incidents={myActiveDispatches}
                resolvedIncidents={myResolvedDispatches}
                onResolve={resolveIncident}
                onDismiss={dismissIncident}
                onConfirm={confirmIncident}
                onReject={rejectIncident}
                onIncidentClick={handleIncidentClick}
              />
            )}
            {currentView === 'analytics' && <AnalyticsView />}
            {currentView === 'users' && <UserManagement />}
            {currentView === 'cameras' && <CameraManagement />}
            {currentView === 'models' && <AIModelManagement />}
            {currentView === 'reports' && (role === 'admin' ? <AdminReports /> : <OfficerReports />)}
            {currentView === 'demos' && role === 'admin' && <DemoRequestsTab />}
            {currentView === 'map' && <MapView />}
            {currentView === 'health' && <SystemHealthView />}
            {currentView === 'feedback' && <AIFeedbackSection />}
          </main>

          {/* Incident Detail Modal */}
          {
            isModalOpen && selectedIncident && (
              <IncidentDetailModal
                incident={selectedIncident}
                isOpen={isModalOpen}
                onClose={handleCloseModal}
                onResolve={handleResolve}
                onDismiss={handleDismiss}
                onConfirm={confirmIncident}
                onReject={rejectIncident}
                onDispatch={dispatchIncident}
                userRole={role}
                userName={userName}
              />
            )
          }

          {/* Status Widget - Fixed Bottom Right */}
          <StatusWidget />
        </div >
      </div >
    </AnimatedBackgroundBeams >
  );
}