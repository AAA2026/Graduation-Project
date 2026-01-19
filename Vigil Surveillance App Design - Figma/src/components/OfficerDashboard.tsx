import { useState } from 'react';
import { DashboardLayout } from './DashboardLayout';
import { DashboardHome } from './DashboardHome';
import { IncidentFeed } from './IncidentFeed';
import { MapView } from './MapView';
import { NotificationCenter } from './NotificationCenter';
import { EventConfirmation } from './EventConfirmation';
import { SecurityMonitoring } from './dashboard/SecurityMonitoring';
import { LayoutDashboard, AlertCircle, Map, Bell, CheckSquare, Users, Eye } from 'lucide-react';

interface OfficerDashboardProps {
  onLogout: () => void;
}

export function OfficerDashboard({ onLogout }: OfficerDashboardProps) {
  const [activeView, setActiveView] = useState('dashboard');

  const navigationItems = [
    { id: 'dashboard', label: 'Live Control', icon: LayoutDashboard },
    { id: 'monitoring', label: 'Security Oversight', icon: Eye },
    { id: 'incidents', label: 'Live Alerts', icon: AlertCircle },
    { id: 'confirmation', label: 'Event Review', icon: CheckSquare },
    { id: 'map', label: 'Map View', icon: Map },
    { id: 'notifications', label: 'Notifications', icon: Bell },
  ];

  const renderContent = () => {
    switch (activeView) {
      case 'dashboard':
        return <DashboardHome role="officer" />;
      case 'monitoring':
        return <SecurityMonitoring />;
      case 'incidents':
        return <IncidentFeed role="officer" />;
      case 'confirmation':
        return <EventConfirmation />;
      case 'map':
        return <MapView />;
      case 'notifications':
        return <NotificationCenter />;
      default:
        return <DashboardHome role="officer" />;
    }
  };

  return (
    <DashboardLayout
      role="Officer"
      navigationItems={navigationItems}
      activeView={activeView}
      onNavigate={setActiveView}
      onLogout={onLogout}
    >
      {renderContent()}
    </DashboardLayout>
  );
}