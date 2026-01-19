import { motion, useScroll, useTransform } from 'motion/react';
import { Play, ArrowRight } from 'lucide-react';
import { ImageWithFallback } from './figma/ImageWithFallback';

interface HeroProps {
  theme: 'light' | 'dark';
  onBookDemo: () => void;
}

export function Hero({ theme, onBookDemo }: HeroProps) {
  // Parallax scrolling effect - Extended for longer scroll distance
  const { scrollY } = useScroll();
  const y = useTransform(scrollY, [0, 2000], [0, 600]);
  const opacity = useTransform(scrollY, [0, 1200], [1, 0]);
  const backgroundOpacity = useTransform(scrollY, [0, 1500], [0.6, 0]);

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section className={`relative min-h-[200vh] flex items-start justify-center px-4 sm:px-6 lg:px-8 pt-24 pb-16 overflow-hidden ${
      theme === 'dark' ? 'bg-[#050505]' : 'bg-[#0a0a0a]'
    }`}>
      {/* Parallax Background Image - Apple-style - Extended height */}
      <motion.div 
        style={{ y, opacity: backgroundOpacity }}
        className="fixed inset-0 z-0 w-full h-screen"
      >
        <div className="relative w-full h-full">
          <ImageWithFallback
            src="https://images.unsplash.com/photo-1708807472445-d33589e6b090?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxzdXJ2ZWlsbGFuY2UlMjBjb250cm9sJTIwcm9vbSUyMG1vbml0b3JzfGVufDF8fHx8MTc2ODQwMzIxOHww&ixlib=rb-4.1.0&q=80&w=1080"
            alt="Modern Surveillance System"
            className="w-full h-full object-cover scale-110"
          />
          
          {/* Gradient overlays - More visible background */}
          <div className={`absolute inset-0 ${
            theme === 'dark' 
              ? 'bg-gradient-to-b from-[#050505]/60 via-[#050505]/70 to-[#050505]/95' 
              : 'bg-gradient-to-b from-[#0a0a0a]/60 via-[#0a0a0a]/70 to-[#0a0a0a]/95'
          }`} />
          
          {/* Brand color accent overlay - more visible */}
          <div className="absolute inset-0 bg-gradient-to-br from-[#06b6d4]/20 via-transparent to-[#0ea5e9]/20" />
          
          {/* Radial gradient for spotlight effect */}
          <div className="absolute inset-0 bg-radial-gradient from-transparent via-transparent to-[#050505]/80" />
        </div>
      </motion.div>

      {/* Main Content */}
      <motion.div 
        style={{ opacity }}
        className="max-w-7xl mx-auto relative z-10 pt-12"
      >
        <div className="text-center">
          {/* Large Centered Logo */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="mb-12 flex justify-center"
          >
            <div className="relative">
              {/* Logo SVG - Using existing VIGIL logo design */}
              <svg width="420" height="120" viewBox="0 0 420 120" fill="none" xmlns="http://www.w3.org/2000/svg">
                {/* Eye Icon - Always using dark mode design */}
                <g transform="translate(20, 20)">
                  <path 
                    d="M 10 40 Q 10 16, 50 10 Q 90 16, 90 40 Q 90 64, 50 70 Q 10 64, 10 40 Z" 
                    fill="#0a0a0a"
                    stroke="url(#gradient1)" 
                    strokeWidth="3"
                  />
                  <path d="M 16 40 Q 16 22, 50 16 Q 84 22, 84 40" stroke="url(#gradient1)" strokeWidth="2.5" fill="none" opacity="0.5"/>
                  <circle cx="50" cy="40" r="18" fill="url(#irisGradient)" stroke="url(#gradient1)" strokeWidth="2.5"/>
                  <g opacity="0.4">
                    {[0, 45, 90, 135, 180, 225, 270, 315].map((angle, i) => {
                      const rad = (angle * Math.PI) / 180;
                      const x2 = 50 + Math.cos(rad) * 16;
                      const y2 = 40 + Math.sin(rad) * 16;
                      return <line key={i} x1="50" y1="40" x2={x2} y2={y2} stroke="#06b6d4" strokeWidth="1.5" />;
                    })}
                  </g>
                  <circle cx="50" cy="40" r="10" fill="#000000"/>
                  <circle cx="54" cy="36" r="4" fill="#06b6d4" opacity="0.9"/>
                  <circle cx="56" cy="34" r="2" fill="#f8fafc"/>
                  <g opacity="0.7">
                    <path d="M 20 24 L 20 20 L 24 20" stroke="url(#gradient1)" strokeWidth="2.5" strokeLinecap="round"/>
                    <path d="M 80 20 L 76 20 L 76 24" stroke="url(#gradient1)" strokeWidth="2.5" strokeLinecap="round"/>
                    <path d="M 20 56 L 20 60 L 24 60" stroke="url(#gradient1)" strokeWidth="2.5" strokeLinecap="round"/>
                    <path d="M 80 60 L 76 60 L 76 56" stroke="url(#gradient1)" strokeWidth="2.5" strokeLinecap="round"/>
                  </g>
                  <line x1="20" y1="40" x2="80" y2="40" stroke="#06b6d4" strokeWidth="1" opacity="0.3" strokeDasharray="4,4"/>
                </g>
                
                {/* VIGIL Text - Always white for visibility */}
                <g transform="translate(140, 20)" fill="#ffffff">
                  <path d="M 0 0 L 18 80 L 32 80 L 50 0 L 36 0 L 25 60 L 14 0 Z"/>
                  <rect x="70" y="0" width="14" height="80"/>
                  <rect x="104" y="0" width="28" height="14"/>
                  <rect x="104" y="0" width="14" height="80"/>
                  <rect x="104" y="66" width="40" height="14"/>
                  <rect x="124" y="36" width="20" height="14"/>
                  <rect x="130" y="36" width="14" height="44"/>
                  <rect x="164" y="0" width="14" height="80"/>
                  <rect x="198" y="0" width="14" height="80"/>
                  <rect x="198" y="66" width="40" height="14"/>
                </g>
                
                {/* Gradients */}
                <defs>
                  <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#06b6d4"/>
                    <stop offset="100%" stopColor="#0ea5e9"/>
                  </linearGradient>
                  <radialGradient id="irisGradient" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.3"/>
                    <stop offset="50%" stopColor="#06b6d4" stopOpacity="0.5"/>
                    <stop offset="100%" stopColor="#0891b2" stopOpacity="0.7"/>
                  </radialGradient>
                </defs>
              </svg>
              
              {/* Glow Effect */}
              <div className="absolute inset-0 blur-3xl opacity-30 bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9]" />
            </div>
          </motion.div>

          {/* Headline */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.8 }}
            className="max-w-4xl mx-auto mb-8"
          >
            <h1 className="text-5xl sm:text-6xl lg:text-7xl mb-6 leading-tight text-white drop-shadow-2xl">
              Intelligent Surveillance
              <br />
              <span className="bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9] bg-clip-text text-transparent drop-shadow-lg">
                That Works for You
              </span>
            </h1>
            <p className="text-xl sm:text-2xl max-w-3xl mx-auto text-gray-200 drop-shadow-lg">
              Transform your existing cameras into powerful AI-driven safety systems. 
              Detect accidents, prevent violence, and optimize operations in real-time.
            </p>
          </motion.div>

          {/* Key Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6, duration: 0.8 }}
            className="flex flex-wrap justify-center gap-8 mb-12"
          >
            {[
              { value: '95%+', label: 'Detection Accuracy' },
              { value: '<1s', label: 'Response Time' },
              { value: '24/7', label: 'Real-Time Monitoring' }
            ].map((stat, index) => (
              <motion.div 
                key={index} 
                className="text-center"
                whileHover={{ scale: 1.05 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <div className="text-3xl sm:text-4xl bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9] bg-clip-text text-transparent font-bold drop-shadow-lg">
                  {stat.value}
                </div>
                <div className="text-sm mt-1 text-gray-300 drop-shadow-md">
                  {stat.label}
                </div>
              </motion.div>
            ))}
          </motion.div>

          {/* CTA Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8, duration: 0.8 }}
            className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12"
          >
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onBookDemo}
              className="group flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9] text-white rounded-lg hover:shadow-lg hover:shadow-[#06b6d4]/50 transition-all text-lg font-medium"
            >
              <span>Book a Demo</span>
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => scrollToSection('features')}
              className="flex items-center gap-2 px-8 py-4 border-2 border-white/30 text-white hover:bg-white/10 hover:border-white/50 backdrop-blur-sm rounded-lg transition-all text-lg font-medium"
            >
              <Play className="w-5 h-5" />
              <span>See It in Action</span>
            </motion.button>
          </motion.div>

          {/* Trust Badges */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1, duration: 0.8 }}
            className="flex flex-wrap items-center justify-center gap-6 pt-8"
          >
            <div className="flex items-center gap-2 text-sm text-gray-300 drop-shadow-md">
              <div className="w-2 h-2 rounded-full bg-green-500" />
              <span>SOC 2 Certified</span>
            </div>
            <div className="flex items-center gap-2 text-sm text-gray-300 drop-shadow-md">
              <div className="w-2 h-2 rounded-full bg-green-500" />
              <span>ISO 27001</span>
            </div>
            <div className="flex items-center gap-2 text-sm text-gray-300 drop-shadow-md">
              <div className="w-2 h-2 rounded-full bg-green-500" />
              <span>GDPR Compliant</span>
            </div>
            <div className="flex items-center gap-2 text-sm text-gray-300 drop-shadow-md">
              <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
              <span>99.9% Uptime</span>
            </div>
          </motion.div>
        </div>
      </motion.div>
    </section>
  );
}
