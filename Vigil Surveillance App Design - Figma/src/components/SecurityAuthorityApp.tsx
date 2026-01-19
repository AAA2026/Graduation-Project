import { useState } from 'react';
import { DashboardLayout } from './DashboardLayout';
import { SecurityAuthorityDashboard } from './SecurityAuthorityDashboard';
import { IncidentFeed } from './IncidentFeed';
import { MapView } from './MapView';
import { NotificationCenter } from './NotificationCenter';
import { 
  AlertCircle, 
  MapPin, 
  Bell,
  Shield,
  Eye
} from 'lucide-react';

interface SecurityAuthorityAppProps {
  onLogout: () => void;
}

export function SecurityAuthorityApp({ onLogout }: SecurityAuthorityAppProps) {
  const [activeView, setActiveView] = useState('dashboard');

  const navigationItems = [
    { id: 'dashboard', label: 'Live Control', icon: Eye },
    { id: 'incidents', label: 'Emergency Alerts', icon: AlertCircle },
    { id: 'map', label: 'Map View', icon: MapPin },
    { id: 'notifications', label: 'Notifications', icon: Bell },
  ];

  const renderContent = () => {
    switch (activeView) {
      case 'dashboard':
        return <SecurityAuthorityDashboard />;
      case 'incidents':
        return <IncidentFeed role="security" />;
      case 'map':
        return <MapView />;
      case 'notifications':
        return <NotificationCenter />;
      default:
        return <SecurityAuthorityDashboard />;
    }
  };

  return (
    <DashboardLayout
      role="Security Authority"
      navigationItems={navigationItems}
      activeView={activeView}
      onNavigate={setActiveView}
      onLogout={onLogout}
    >
      {renderContent()}
    </DashboardLayout>
  );
}