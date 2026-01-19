
import { Sun, Moon } from 'lucide-react';
import { Button } from './ui/button';
import { useTheme } from './ThemeProvider';

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();

  return (
    <div
      style={{
        position: 'fixed',
        top: 24,
        right: 32,
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
      }}
    >
      <Button
        variant="outline"
        size="lg"
        onClick={toggleTheme}
        className="theme-toggle-btn flex items-center gap-2 px-5 py-2 text-base font-semibold border-2 border-primary/60 bg-white/80 dark:bg-[#18181b]/80 dark:text-white text-gray-900 shadow-lg rounded-2xl hover:bg-primary/10 hover:border-primary focus-visible:ring-2 focus-visible:ring-primary/60 transition-all duration-200"
        style={{
          minWidth: 160,
          boxShadow: theme === 'dark'
            ? '0 4px 24px 0 #06b6d455, 0 1.5px 8px 0 #23272f33'
            : '0 4px 24px 0 #6366f155, 0 1.5px 8px 0 #c7d2fe33',
          borderRadius: 18,
          transform: 'translateZ(0)',
        }}
        onMouseOver={e => (e.currentTarget.style.transform = 'scale(1.06)')}
        onMouseOut={e => (e.currentTarget.style.transform = 'scale(1)')}
      >
        {theme === 'light' ? (
          <>
            {/* Blue-filled moon for light mode (switch to dark) */}
            <Moon className="w-6 h-6" fill="#60a5fa" stroke="#60a5fa" />
            <span>Dark Mode</span>
          </>
        ) : (
          <>
            {/* Yellow-filled sun for dark mode (switch to light) */}
            <Sun className="w-6 h-6" fill="#facc15" stroke="#facc15" />
            <span>Light Mode</span>
          </>
        )}
      </Button>
    </div>
  );
}
