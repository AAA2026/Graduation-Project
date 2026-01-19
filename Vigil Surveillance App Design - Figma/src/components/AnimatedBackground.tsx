import { motion } from 'motion/react';
import { useTheme } from './ThemeProvider';

interface AnimatedBackgroundProps {
  variant?: 'modal' | 'dashboard' | 'card';
  intensity?: 'subtle' | 'medium' | 'strong';
  className?: string;
}

export function AnimatedBackground({ 
  variant = 'dashboard', 
  intensity = 'subtle',
  className = '' 
}: AnimatedBackgroundProps) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  // Animation variants for floating shapes
  const floatingVariants = {
    initial: { 
      x: 0, 
      y: 0, 
      scale: 1,
      rotate: 0,
    },
    animate1: {
      x: [0, 30, -20, 0],
      y: [0, -40, 30, 0],
      scale: [1, 1.1, 0.9, 1],
      rotate: [0, 5, -5, 0],
      transition: {
        duration: 20,
        repeat: Infinity,
        ease: "easeInOut",
      }
    },
    animate2: {
      x: [0, -40, 20, 0],
      y: [0, 30, -30, 0],
      scale: [1, 0.9, 1.1, 1],
      rotate: [0, -8, 8, 0],
      transition: {
        duration: 25,
        repeat: Infinity,
        ease: "easeInOut",
      }
    },
    animate3: {
      x: [0, 20, -30, 0],
      y: [0, -20, 40, 0],
      scale: [1, 1.15, 0.85, 1],
      rotate: [0, 10, -10, 0],
      transition: {
        duration: 30,
        repeat: Infinity,
        ease: "easeInOut",
      }
    }
  };

  // Intensity settings
  const intensitySettings = {
    subtle: { opacity: isDark ? 0.15 : 0.10, blur: 80 },
    medium: { opacity: isDark ? 0.25 : 0.15, blur: 60 },
    strong: { opacity: isDark ? 0.35 : 0.20, blur: 40 },
  };

  const { opacity, blur } = intensitySettings[intensity];

  // Color schemes for dark and light modes
  const darkModeColors = {
    base1: 'from-cyan-500/40 to-blue-600/40',
    base2: 'from-blue-500/30 to-purple-600/30',
    base3: 'from-purple-500/25 to-cyan-500/25',
    accent1: 'bg-cyan-400',
    accent2: 'bg-blue-500',
    accent3: 'bg-purple-500',
  };

  const lightModeColors = {
    base1: 'from-blue-200/60 to-cyan-200/60',
    base2: 'from-purple-200/50 to-blue-200/50',
    base3: 'from-yellow-200/40 to-blue-200/40',
    accent1: 'bg-blue-300',
    accent2: 'bg-cyan-300',
    accent3: 'bg-purple-300',
  };

  const colors = isDark ? darkModeColors : lightModeColors;

  return (
    <div className={`absolute inset-0 overflow-hidden pointer-events-none ${className}`}>
      {/* Base Gradient Layer */}
      <div className={`absolute inset-0 ${
        isDark 
          ? 'bg-gradient-to-br from-black via-gray-950 to-gray-900'
          : 'bg-gradient-to-br from-white via-gray-50 to-gray-100'
      }`} />

      {/* Accent Gradient Overlays */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: opacity * 0.8 }}
        transition={{ duration: 2 }}
        className={`absolute inset-0 bg-gradient-to-br ${colors.base1}`}
        style={{ filter: `blur(${blur}px)` }}
      />

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: opacity * 0.6 }}
        transition={{ duration: 2.5, delay: 0.3 }}
        className={`absolute inset-0 bg-gradient-to-tl ${colors.base2}`}
        style={{ filter: `blur(${blur * 1.2}px)` }}
      />

      {/* Animated Floating Shapes */}
      {/* Shape 1 - Top Left */}
      <motion.div
        variants={floatingVariants}
        initial="initial"
        animate="animate1"
        className={`absolute -top-20 -left-20 w-96 h-96 ${colors.accent1} rounded-full`}
        style={{ 
          opacity: opacity * 0.5,
          filter: `blur(${blur}px)`,
        }}
      />

      {/* Shape 2 - Top Right */}
      <motion.div
        variants={floatingVariants}
        initial="initial"
        animate="animate2"
        className={`absolute -top-32 -right-32 w-[500px] h-[500px] ${colors.accent2} rounded-full`}
        style={{ 
          opacity: opacity * 0.4,
          filter: `blur(${blur * 1.1}px)`,
        }}
      />

      {/* Shape 3 - Bottom Left */}
      <motion.div
        variants={floatingVariants}
        initial="initial"
        animate="animate3"
        className={`absolute -bottom-24 -left-24 w-[450px] h-[450px] ${colors.accent3} rounded-full`}
        style={{ 
          opacity: opacity * 0.45,
          filter: `blur(${blur * 0.9}px)`,
        }}
      />

      {/* Shape 4 - Bottom Right */}
      <motion.div
        variants={floatingVariants}
        initial="initial"
        animate="animate1"
        className={`absolute -bottom-40 -right-40 w-[550px] h-[550px] ${colors.accent1} rounded-full`}
        style={{ 
          opacity: opacity * 0.35,
          filter: `blur(${blur * 1.3}px)`,
        }}
      />

      {/* Shape 5 - Center Float */}
      <motion.div
        variants={floatingVariants}
        initial="initial"
        animate="animate2"
        className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] ${colors.accent2} rounded-full`}
        style={{ 
          opacity: opacity * 0.3,
          filter: `blur(${blur * 1.5}px)`,
        }}
      />

      {/* Noise/Grain Overlay for Texture */}
      <div 
        className={`absolute inset-0 ${isDark ? 'opacity-[0.015]' : 'opacity-[0.02]'}`}
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
        }}
      />

      {/* Subtle Border Glow */}
      {variant === 'modal' && (
        <div className={`absolute inset-0 rounded-lg border ${
          isDark 
            ? 'border-cyan-500/20' 
            : 'border-blue-400/20'
        }`} />
      )}

      {/* Radial Vignette Effect */}
      <div className={`absolute inset-0 ${
        isDark
          ? 'bg-[radial-gradient(circle_at_center,transparent_0%,rgba(0,0,0,0.3)_100%)]'
          : 'bg-[radial-gradient(circle_at_center,transparent_0%,rgba(255,255,255,0.4)_100%)]'
      }`} />
    </div>
  );
}
