import { useState, useEffect } from 'react';
import { Moon, Sun, LogIn } from 'lucide-react';
import { motion } from 'motion/react';

interface NavigationProps {
  theme: 'light' | 'dark';
  toggleTheme: () => void;
  onLogin: () => void;
}

export function Navigation({ theme, toggleTheme, onLogin }: NavigationProps) {
  const [scrolled, setScrolled] = useState(false);
  const [isOverHero, setIsOverHero] = useState(true);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
      // Consider we're over hero if scroll position is less than ~80vh (roughly the hero height)
      setIsOverHero(window.scrollY < window.innerHeight * 0.8);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  // Determine logo colors based on position and theme
  const logoTextColor = isOverHero ? '#ffffff' : (theme === 'dark' ? '#ffffff' : '#0a0a0a');
  const logoEyeFill = isOverHero ? '#1a1a1a' : '#0a0a0a';

  return (
    <motion.nav 
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled 
          ? theme === 'dark'
            ? 'bg-[#0a0a0a]/95 backdrop-blur-lg border-b border-[#06b6d4]/20 shadow-lg'
            : 'bg-white/95 backdrop-blur-lg border-b border-gray-200 shadow-lg'
          : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          {/* Logo */}
          <motion.div 
            whileHover={{ scale: 1.05 }}
            className="flex items-center gap-3 cursor-pointer"
          >
            <svg width="180" height="40" viewBox="0 0 240 80" fill="none" xmlns="http://www.w3.org/2000/svg" className="transition-all duration-300">
              {/* Eye Icon */}
              <g transform="translate(15, 20)">
                <path 
                  d="M 5 20 Q 5 8, 25 5 Q 45 8, 45 20 Q 45 32, 25 35 Q 5 32, 5 20 Z" 
                  fill={logoEyeFill}
                  stroke="#06b6d4" 
                  strokeWidth="2"
                  className="transition-all duration-300"
                />
                <path d="M 8 20 Q 8 11, 25 8 Q 42 11, 42 20" stroke="#06b6d4" strokeWidth="1.5" fill="none" opacity="0.5"/>
                <circle cx="25" cy="20" r="9" fill="url(#irisGradient)" stroke="#06b6d4" strokeWidth="1.5"/>
                <g opacity="0.4">
                  {[0, 45, 90, 135, 180, 225, 270, 315].map((angle, i) => {
                    const rad = (angle * Math.PI) / 180;
                    const x2 = 25 + Math.cos(rad) * 8;
                    const y2 = 20 + Math.sin(rad) * 8;
                    return <line key={i} x1="25" y1="20" x2={x2} y2={y2} stroke="#06b6d4" strokeWidth="0.8" />;
                  })}
                </g>
                <circle cx="25" cy="20" r="5" fill="#000000"/>
                <circle cx="27" cy="18" r="2" fill="#06b6d4" opacity="0.9"/>
                <circle cx="28" cy="17" r="1" fill="#f8fafc"/>
                <g opacity="0.7">
                  <path d="M 10 12 L 10 10 L 12 10" stroke="#06b6d4" strokeWidth="1.5" strokeLinecap="round"/>
                  <path d="M 40 10 L 38 10 L 38 12" stroke="#06b6d4" strokeWidth="1.5" strokeLinecap="round"/>
                  <path d="M 10 28 L 10 30 L 12 30" stroke="#06b6d4" strokeWidth="1.5" strokeLinecap="round"/>
                  <path d="M 40 30 L 38 30 L 38 28" stroke="#06b6d4" strokeWidth="1.5" strokeLinecap="round"/>
                </g>
                <line x1="10" y1="20" x2="40" y2="20" stroke="#06b6d4" strokeWidth="0.5" opacity="0.3" strokeDasharray="2,2"/>
              </g>
              
              <defs>
                <radialGradient id="irisGradient" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.3"/>
                  <stop offset="50%" stopColor="#06b6d4" stopOpacity="0.5"/>
                  <stop offset="100%" stopColor="#0891b2" stopOpacity="0.7"/>
                </radialGradient>
              </defs>
              
              {/* VIGIL Text - Color changes based on theme */}
              <g transform="translate(75, 18)" fill={logoTextColor}>
                <path d="M 0 0 L 9 40 L 16 40 L 25 0 L 18 0 L 12.5 30 L 7 0 Z"/>
                <rect x="35" y="0" width="7" height="40"/>
                <rect x="52" y="0" width="14" height="7"/>
                <rect x="52" y="0" width="7" height="40"/>
                <rect x="52" y="33" width="20" height="7"/>
                <rect x="62" y="18" width="10" height="7"/>
                <rect x="65" y="18" width="7" height="22"/>
                <rect x="82" y="0" width="7" height="40"/>
                <rect x="99" y="0" width="7" height="40"/>
                <rect x="99" y="33" width="20" height="7"/>
              </g>
              
              {/* Pulsing status indicator */}
              <circle cx="228" cy="48" r="3" fill="#06b6d4">
                <animate attributeName="opacity" values="1;0.3;1" dur="2s" repeatCount="indefinite"/>
              </circle>
            </svg>
          </motion.div>

          {/* Navigation Links */}
          <div className="hidden md:flex items-center space-x-8">
            <motion.button 
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => scrollToSection('features')}
              className={`transition-colors relative group ${
                theme === 'dark' ? 'text-[#cbd5e1] hover:text-[#06b6d4]' : 'text-gray-600 hover:text-[#06b6d4]'
              }`}
            >
              Features
              <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9] group-hover:w-full transition-all duration-300" />
            </motion.button>
            <motion.button 
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => scrollToSection('use-cases')}
              className={`transition-colors relative group ${
                theme === 'dark' ? 'text-[#cbd5e1] hover:text-[#06b6d4]' : 'text-gray-600 hover:text-[#06b6d4]'
              }`}
            >
              Use Cases
              <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9] group-hover:w-full transition-all duration-300" />
            </motion.button>
            <motion.button 
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => scrollToSection('technology')}
              className={`transition-colors relative group ${
                theme === 'dark' ? 'text-[#cbd5e1] hover:text-[#06b6d4]' : 'text-gray-600 hover:text-[#06b6d4]'
              }`}
            >
              Technology
              <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9] group-hover:w-full transition-all duration-300" />
            </motion.button>
            <motion.button 
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => scrollToSection('team')}
              className={`transition-colors relative group ${
                theme === 'dark' ? 'text-[#cbd5e1] hover:text-[#06b6d4]' : 'text-gray-600 hover:text-[#06b6d4]'
              }`}
            >
              Technology & Expertise
              <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9] group-hover:w-full transition-all duration-300" />
            </motion.button>
          </div>

          {/* Actions */}
          <div className="flex items-center space-x-4">
            <motion.button
              whileHover={{ scale: 1.1, rotate: 180 }}
              whileTap={{ scale: 0.9 }}
              onClick={toggleTheme}
              className={`p-2 rounded-lg transition-colors ${
                theme === 'dark' ? 'hover:bg-[#06b6d4]/10' : 'hover:bg-[#06b6d4]/10'
              }`}
              aria-label="Toggle theme"
            >
              {theme === 'dark' ? (
                <Sun className="w-5 h-5 text-[#06b6d4]" />
              ) : (
                <Moon className="w-5 h-5 text-[#06b6d4]" />
              )}
            </motion.button>
            
            <motion.button
              whileHover={{ scale: 1.05, boxShadow: '0 10px 30px rgba(6, 182, 212, 0.5)' }}
              whileTap={{ scale: 0.95 }}
              onClick={onLogin}
              className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9] rounded-lg transition-all text-white font-medium"
            >
              <span>Login</span>
              <LogIn className="w-4 h-4" />
            </motion.button>
          </div>
        </div>
      </div>
    </motion.nav>
  );
}
