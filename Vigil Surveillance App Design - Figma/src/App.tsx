import { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { LoginScreen } from "./components/LoginScreen";
import { ModernSecurityLayout } from "./components/ModernSecurityLayout";
import { Toaster } from "./components/ui/sonner";
import { ThemeProvider } from "./components/ThemeProvider";
import { LandingPage } from "./components/landing/LandingPage";

type UserRole = "admin" | "officer" | "security" | null;

function SecurityApp() {
  const [currentRole, setCurrentRole] = useState<UserRole>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentView, setCurrentView] = useState('live');
  const [userName, setUserName] = useState('');

  // Check for existing session/settings (simplified)
  useEffect(() => {
    // Optional: check localStorage for persistence if you want
  }, []);

  const handleLogin = (role: UserRole) => {
    setCurrentRole(role);
    setIsAuthenticated(true);
    setCurrentView('live');
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

  if (!isAuthenticated || !currentRole) {
    return (
      <>
        <LoginScreen onLogin={handleLogin} />
        <Toaster position="top-right" closeButton />
      </>
    );
  }

  return (
    <>
      <ModernSecurityLayout
        role={currentRole}
        onLogout={handleLogout}
        onNavigate={setCurrentView}
        currentView={currentView}
        userName={userName}
      />
      <Toaster position="top-right" className="mt-16" closeButton />
    </>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/app" element={<SecurityApp />} />
          {/* Fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}