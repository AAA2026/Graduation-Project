import { ReactNode, useState } from 'react';
import { Button } from './ui/button';
import { ScrollArea } from './ui/scroll-area';
import { Separator } from './ui/separator';
import { Badge } from './ui/badge';
import { Eye, LogOut, Menu, X, Bell } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { VigilLogo } from './VigilLogo';
import { ThemeToggle } from './ThemeToggle';
import { useTheme } from './ThemeProvider';

interface NavigationItem {
  id: string;
  label: string;
  icon: LucideIcon;
}

interface DashboardLayoutProps {
  role: string;
  navigationItems: NavigationItem[];
  activeView: string;
  onNavigate: (viewId: string) => void;
  onLogout: () => void;
  children: ReactNode;
}

export function DashboardLayout({
  role,
  navigationItems,
  activeView,
  onNavigate,
  onLogout,
  children,
}: DashboardLayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { theme } = useTheme();

  return (
    <div className={`min-h-screen ${theme === 'dark' ? 'bg-[#1c2746]' : 'bg-background'} text-foreground`}>
      {/* Top Bar */}
      <div className="sticky top-0 z-50 bg-card/95 backdrop-blur-lg border-b border-border">
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="icon"
              className="lg:hidden text-muted-foreground hover:text-foreground"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </Button>
            <div className="flex items-center gap-3">
              <VigilLogo width={120} height={40} theme={theme} />
              <div className="hidden sm:block border-l border-border pl-3 ml-1">
                <p className="text-xs text-muted-foreground">{role} Dashboard</p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              className="relative text-muted-foreground hover:text-foreground"
            >
              <Bell className="w-5 h-5" />
              <Badge className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-red-500 text-white p-0 flex items-center justify-center text-xs">
                3
              </Badge>
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={onLogout}
              className="text-muted-foreground hover:text-foreground"
            >
              <LogOut className="w-5 h-5" />
            </Button>
            <ThemeToggle />
          </div>
        </div>
      </div>

      <div className="flex">
        {/* Sidebar */}
        <aside
          className={`fixed lg:sticky top-[57px] left-0 z-40 h-[calc(100vh-57px)] w-64 bg-card/80 backdrop-blur-lg border-r border-border transition-transform lg:translate-x-0 ${
            sidebarOpen ? 'translate-x-0' : '-translate-x-full'
          }`}
        >
          <ScrollArea className="h-full">
            <nav className="p-4 space-y-1">
              {navigationItems.map((item) => {
                const Icon = item.icon;
                const isActive = activeView === item.id;
                return (
                  <Button
                    key={item.id}
                    variant={isActive ? 'default' : 'ghost'}
                    className={`w-full justify-start ${
                      isActive
                        ? 'bg-blue-600 hover:bg-blue-700 text-white'
                        : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                    }`}
                    onClick={() => {
                      onNavigate(item.id);
                      setSidebarOpen(false);
                    }}
                  >
                    <Icon className="w-5 h-5 mr-3" />
                    {item.label}
                  </Button>
                );
              })}
            </nav>
          </ScrollArea>
        </aside>

        {/* Overlay for mobile */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 z-30 bg-black/50 lg:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Main Content */}
        <main className="flex-1 min-h-[calc(100vh-57px)]">
          <ScrollArea className="h-[calc(100vh-57px)]">
            <div className="p-4 lg:p-6">{children}</div>
          </ScrollArea>
        </main>
      </div>
    </div>
  );
}