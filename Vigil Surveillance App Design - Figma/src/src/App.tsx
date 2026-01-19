import { useState } from "react";
import { LoginScreen } from "./components/LoginScreen";
import { ModernSecurityLayout } from "./components/ModernSecurityLayout";
import { Toaster } from "./components/ui/sonner";
import { ThemeProvider } from "./components/ThemeProvider";

type UserRole = "admin" | "officer" | "security-authority" | null;

export default function App() {
  const [currentRole, setCurrentRole] = useState<UserRole>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentView, setCurrentView] = useState('live');
  const [userName, setUserName] = useState('');

  const handleLogin = (role: UserRole) => {
    // Map old role names to new ones
    if (role === 'security') {
      setCurrentRole('security-authority');
    } else {
      setCurrentRole(role);
    }
    setIsAuthenticated(true);
    setCurrentView('live');
    
    // Set user name based on role
    if (role === 'admin') setUserName('Admin User');
    else if (role === 'officer') setUserName('Officer Smith');
    else setUserName('Chief Martinez');
  };

  const handleLogout = () => {
    setCurrentRole(null);
    setIsAuthenticated(false);
    setCurrentView('live');
    setUserName('');
  };

  return (
    <ThemeProvider>
      {!isAuthenticated || !currentRole ? (
        <>
          <LoginScreen onLogin={handleLogin} />
          <Toaster />
        </>
      ) : (
        <>
          <ModernSecurityLayout
            role={currentRole}
            onLogout={handleLogout}
            onNavigate={setCurrentView}
            currentView={currentView}
            userName={userName}
          />
          <Toaster />
        </>
      )}
    </ThemeProvider>
  );
}
