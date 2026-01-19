import { motion } from 'motion/react';
import { useEffect, useRef, useState } from 'react';

interface StatsShowcaseProps {
  theme: 'light' | 'dark';
}

function AnimatedNumber({ value, suffix = '' }: { value: string; suffix?: string }) {
  const [displayValue, setDisplayValue] = useState('0');
  const elementRef = useRef<HTMLDivElement>(null);
  const [hasAnimated, setHasAnimated] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && !hasAnimated) {
          setHasAnimated(true);
          
          const numericValue = parseFloat(value.replace(/[^0-9.]/g, ''));
          if (!isNaN(numericValue)) {
            const duration = 2000;
            const steps = 60;
            const stepValue = numericValue / steps;
            let current = 0;
            
            const interval = setInterval(() => {
              current += stepValue;
              if (current >= numericValue) {
                setDisplayValue(value);
                clearInterval(interval);
              } else {
                setDisplayValue(Math.floor(current).toString());
              }
            }, duration / steps);
          } else {
            setDisplayValue(value);
          }
        }
      },
      { threshold: 0.5 }
    );

    if (elementRef.current) {
      observer.observe(elementRef.current);
    }

    return () => observer.disconnect();
  }, [value, hasAnimated]);

  return (
    <div ref={elementRef} className="text-5xl bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9] bg-clip-text text-transparent font-bold">
      {displayValue}{suffix}
    </div>
  );
}

export function StatsShowcase({ theme }: StatsShowcaseProps) {
  const stats = [
    { value: '40', suffix: '%', label: 'Faster Response Times', sublabel: 'Average incident detection' },
    { value: '60', suffix: '%', label: 'Cost Reduction', sublabel: 'In security operations' },
    { value: '99.9', suffix: '%', label: 'System Uptime', sublabel: 'Enterprise reliability' },
    { value: '24', suffix: '/7', label: 'Real-Time Monitoring', sublabel: 'Always watching' }
  ];

  return (
    <section className={`relative py-20 px-4 sm:px-6 lg:px-8 ${
      theme === 'dark' ? 'bg-[#0a0a0a]/50' : 'bg-white'
    }`}>
      <div className="max-w-7xl mx-auto relative z-10">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className={`text-4xl sm:text-5xl mb-4 ${
            theme === 'dark' ? 'text-white' : 'text-gray-900'
          }`}>
            Measurable Business Impact
          </h2>
          <p className={`text-xl max-w-2xl mx-auto ${
            theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
          }`}>
            See the difference VIGIL makes to your bottom line
          </p>
        </motion.div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
          {stats.map((stat, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.05 }}
              className={`text-center p-8 rounded-2xl transition-all ${
                theme === 'dark'
                  ? 'bg-[#06b6d4]/5 border border-[#06b6d4]/20 hover:border-[#06b6d4]/50 hover:bg-[#06b6d4]/10'
                  : 'bg-gradient-to-br from-[#06b6d4]/10 to-[#0ea5e9]/5 border border-[#06b6d4]/20 hover:border-[#06b6d4]/40 hover:shadow-lg'
              }`}
            >
              <AnimatedNumber value={stat.value} suffix={stat.suffix} />
              <div className={`mt-3 text-lg ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                {stat.label}
              </div>
              <div className={`mt-1 text-sm ${theme === 'dark' ? 'text-gray-500' : 'text-gray-500'}`}>
                {stat.sublabel}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
