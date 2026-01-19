import { motion } from 'motion/react';
import { Shield, Zap, DollarSign } from 'lucide-react';

interface ROISectionProps {
  theme: 'light' | 'dark';
}

const benefits = [
  {
    icon: Shield,
    title: 'Risk Reduction',
    description: 'Reduce security incidents by up to 60% with proactive detection',
    stat: 'Average 40-second response time improvement',
  },
  {
    icon: Zap,
    title: 'Operational Efficiency',
    description: 'Eliminate manual monitoring fatigue and enhance team productivity',
    stat: '4x faster incident escalation',
  },
  {
    icon: DollarSign,
    title: 'Cost Optimization',
    description: 'ROI achieved within 18 months through incident prevention and insurance savings',
    stat: 'Average 35% operational cost reduction',
  },
];

export function ROISection({ theme }: ROISectionProps) {
  return (
    <section id="roi" className={`py-20 px-4 sm:px-6 lg:px-8 ${
      theme === 'dark' ? 'bg-[#0a0a0a]' : 'bg-white'
    }`}>
      <div className="max-w-6xl mx-auto">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className={`mb-4 ${theme === 'dark' ? 'text-[#f8fafc]' : 'text-[#0f172a]'}`}>
            Proven Enterprise Benefits
          </h2>
          <p className={`max-w-2xl mx-auto ${
            theme === 'dark' ? 'text-[#cbd5e1]' : 'text-[#475569]'
          }`}>
            Security teams report significant improvements in response times and incident prevention
          </p>
        </motion.div>

        {/* Benefits Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {benefits.map((benefit, index) => (
            <motion.div
              key={benefit.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              className={`p-8 rounded-2xl border relative overflow-hidden group ${
                theme === 'dark'
                  ? 'bg-[#050505] border-[#06b6d4]/20 hover:border-[#06b6d4]/40'
                  : 'bg-[#f8fafc] border-gray-200 hover:border-[#0ea5e9]/40'
              }`}
            >
              {/* Icon */}
              <motion.div
                whileHover={{ rotate: [0, -10, 10, -10, 0] }}
                transition={{ duration: 0.5 }}
                className={`w-14 h-14 rounded-xl flex items-center justify-center mb-6 ${
                  theme === 'dark'
                    ? 'bg-gradient-to-br from-[#06b6d4] to-[#0ea5e9]'
                    : 'bg-gradient-to-br from-[#0ea5e9] to-[#06b6d4]'
                } shadow-lg`}
              >
                <benefit.icon className="w-7 h-7 text-white" />
              </motion.div>

              {/* Content */}
              <h3 className={`mb-3 ${theme === 'dark' ? 'text-[#f8fafc]' : 'text-[#0f172a]'}`}>
                {benefit.title}
              </h3>
              <p className={`mb-4 ${theme === 'dark' ? 'text-[#cbd5e1]' : 'text-[#475569]'}`}>
                {benefit.description}
              </p>

              {/* Stat */}
              <div className={`text-sm ${
                theme === 'dark' ? 'text-[#06b6d4]' : 'text-[#0ea5e9]'
              }`}>
                {benefit.stat}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
