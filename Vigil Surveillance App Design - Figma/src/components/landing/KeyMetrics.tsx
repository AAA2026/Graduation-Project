import { motion } from 'motion/react';
import { useEffect, useState, useRef } from 'react';
import { TrendingUp, Activity, Shield } from 'lucide-react';

interface KeyMetricsProps {
  theme: 'light' | 'dark';
}

function AnimatedCounter({ target, suffix = '' }: { target: number; suffix?: string }) {
  const [count, setCount] = useState(0);
  const [hasAnimated, setHasAnimated] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && !hasAnimated) {
          setHasAnimated(true);
          const duration = 2000;
          const steps = 60;
          const increment = target / steps;
          let current = 0;

          const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
              setCount(target);
              clearInterval(timer);
            } else {
              setCount(Math.floor(current));
            }
          }, duration / steps);

          return () => clearInterval(timer);
        }
      },
      { threshold: 0.5 }
    );

    if (ref.current) {
      observer.observe(ref.current);
    }

    return () => observer.disconnect();
  }, [target, hasAnimated]);

  return (
    <div ref={ref}>
      {count}
      {suffix}
    </div>
  );
}

export function KeyMetrics({ theme }: KeyMetricsProps) {
  return (
    <section className={`py-20 px-4 sm:px-6 lg:px-8 ${
      theme === 'dark' ? 'bg-[#0a0a0a]' : 'bg-white'
    }`}>
      <div className="max-w-6xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Metric 1 */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className={`p-8 rounded-2xl border relative overflow-hidden group hover:scale-105 transition-transform ${
              theme === 'dark'
                ? 'bg-gradient-to-br from-[#0a0a0a] to-[#06b6d4]/5 border-[#06b6d4]/20 hover:border-[#06b6d4]/40'
                : 'bg-gradient-to-br from-white to-[#0ea5e9]/5 border-gray-200 hover:border-[#0ea5e9]/40'
            }`}
          >
            <div className={`w-12 h-12 rounded-lg flex items-center justify-center mb-4 ${
              theme === 'dark'
                ? 'bg-[#06b6d4]/10 border border-[#06b6d4]/20'
                : 'bg-[#0ea5e9]/10 border border-[#0ea5e9]/20'
            }`}>
              <TrendingUp className="w-6 h-6 text-[#06b6d4]" />
            </div>
            <div className="text-4xl text-[#06b6d4] mb-2">
              <AnimatedCounter target={847} suffix="+" />
            </div>
            <div className={`mb-1 ${theme === 'dark' ? 'text-[#f8fafc]' : 'text-[#0f172a]'}`}>
              Incidents Detected
            </div>
            <div className={`text-sm ${theme === 'dark' ? 'text-[#cbd5e1]' : 'text-[#475569]'}`}>
              Real-time accuracy: 94%
            </div>
          </motion.div>

          {/* Metric 2 */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className={`p-8 rounded-2xl border relative overflow-hidden group hover:scale-105 transition-transform ${
              theme === 'dark'
                ? 'bg-gradient-to-br from-[#0a0a0a] to-[#06b6d4]/5 border-[#06b6d4]/20 hover:border-[#06b6d4]/40'
                : 'bg-gradient-to-br from-white to-[#0ea5e9]/5 border-gray-200 hover:border-[#0ea5e9]/40'
            }`}
          >
            <div className={`w-12 h-12 rounded-lg flex items-center justify-center mb-4 ${
              theme === 'dark'
                ? 'bg-[#06b6d4]/10 border border-[#06b6d4]/20'
                : 'bg-[#0ea5e9]/10 border border-[#0ea5e9]/20'
            }`}>
              <Activity className="w-6 h-6 text-[#06b6d4]" />
            </div>
            <div className="text-4xl text-[#06b6d4] mb-2">
              <AnimatedCounter target={12} />
            </div>
            <div className={`mb-1 ${theme === 'dark' ? 'text-[#f8fafc]' : 'text-[#0f172a]'}`}>
              Live Deployment Sites
            </div>
            <div className={`text-sm ${theme === 'dark' ? 'text-[#cbd5e1]' : 'text-[#475569]'}`}>
              24/7 Monitoring
            </div>
          </motion.div>

          {/* Metric 3 */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className={`p-8 rounded-2xl border relative overflow-hidden group hover:scale-105 transition-transform ${
              theme === 'dark'
                ? 'bg-gradient-to-br from-[#0a0a0a] to-[#06b6d4]/5 border-[#06b6d4]/20 hover:border-[#06b6d4]/40'
                : 'bg-gradient-to-br from-white to-[#0ea5e9]/5 border-gray-200 hover:border-[#0ea5e9]/40'
            }`}
          >
            <div className={`w-12 h-12 rounded-lg flex items-center justify-center mb-4 ${
              theme === 'dark'
                ? 'bg-[#06b6d4]/10 border border-[#06b6d4]/20'
                : 'bg-[#0ea5e9]/10 border border-[#0ea5e9]/20'
            }`}>
              <Shield className="w-6 h-6 text-[#06b6d4]" />
            </div>
            <div className="text-4xl text-[#06b6d4] mb-2">99.9%</div>
            <div className={`mb-1 ${theme === 'dark' ? 'text-[#f8fafc]' : 'text-[#0f172a]'}`}>
              System Uptime
            </div>
            <div className={`text-sm ${theme === 'dark' ? 'text-[#cbd5e1]' : 'text-[#475569]'}`}>
              Enterprise SLA
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
