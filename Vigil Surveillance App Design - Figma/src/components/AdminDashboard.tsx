import { useState } from 'react';
import { DashboardLayout } from './DashboardLayout';
import { DashboardHome } from './DashboardHome';
import { IncidentCenter } from './IncidentCenter';
import { MapView } from './MapView';
import { AnalyticsView } from './AnalyticsView';
import { CameraManagement } from './CameraManagement';
import { AIFeedbackSection } from './AIFeedbackSection';
// ...existing code...
import { UserManagement } from './UserManagement';
import { SystemHealthView } from './SystemHealthView';
import { AIModelManagement } from './AIModelManagement';
import {
  LayoutDashboard,
  AlertCircle,
  Map,
  BarChart3,
  Camera,
  Brain,
  Bell,
  Users,
  Activity,
  Cpu,
} from 'lucide-react';

interface AdminDashboardProps {
  onLogout: () => void;
}

export function AdminDashboard({ onLogout }: AdminDashboardProps) {
  const [activeView, setActiveView] = useState('dashboard');

  const navigationItems = [
    { id: 'dashboard', label: 'Live Control', icon: LayoutDashboard },
    { id: 'incidents', label: 'Incidents', icon: AlertCircle },
    { id: 'map', label: 'Map View', icon: Map },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'cameras', label: 'Cameras', icon: Camera },
    { id: 'users', label: 'Users', icon: Users },
    { id: 'system-health', label: 'System Health', icon: Activity },
    { id: 'ai-models', label: 'AI Models', icon: Cpu },
    { id: 'ai-feedback', label: 'AI Feedback', icon: Brain },
    { id: 'notifications', label: 'Notifications', icon: Bell },
  ];

  const renderContent = () => {
    switch (activeView) {
      case 'dashboard':
        return <DashboardHome role="admin" />;
      case 'incidents':
        return <IncidentCenter />;
      case 'map':
        return <MapView />;
      case 'analytics':
        return <AnalyticsView />;
      case 'cameras':
        return <CameraManagement />;
      case 'users':
        return <UserManagement />;
      case 'system-health':
        return <SystemHealthView />;
      case 'ai-models':
        return <AIModelManagement />;
      case 'ai-feedback':
        return <AIFeedbackSection />;
      // Notifications tab merged into incidents
      default:
        return <DashboardHome role="admin" />;
    }
  };

  return (
    <DashboardLayout
      role="Admin"
      navigationItems={navigationItems}
      activeView={activeView}
      onNavigate={setActiveView}
      onLogout={onLogout}
    >
      {renderContent()}
    </DashboardLayout>
  );
}