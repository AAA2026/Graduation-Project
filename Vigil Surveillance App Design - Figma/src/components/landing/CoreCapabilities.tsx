import { motion } from 'motion/react';
import { ShieldCheck, Activity, Eye, Network } from 'lucide-react';

interface CoreCapabilitiesProps {
  theme: 'light' | 'dark';
}

const capabilities = [
  {
    icon: ShieldCheck,
    title: 'Violence & Aggression Detection',
    description: 'Detects physical confrontations and aggressive behavior with 94% accuracy',
    metric: '< 100ms Detection Latency',
  },
  {
    icon: Activity,
    title: 'Accident & Collision Monitoring',
    description: 'Real-time detection of traffic incidents and workplace accidents',
    metric: 'Instant Alert Dispatch',
  },
  {
    icon: Eye,
    title: 'Behavioral Anomaly Detection',
    description: 'Identifies unusual patterns and potential security threats',
    metric: 'ML-Powered Analysis',
  },
  {
    icon: Network,
    title: 'Integration & Automation',
    description: 'Seamless integration with existing security infrastructure',
    metric: 'API-First Architecture',
  },
];

export function CoreCapabilities({ theme }: CoreCapabilitiesProps) {
  return (
    <section id="features" className={`py-20 px-4 sm:px-6 lg:px-8 ${
      theme === 'dark' ? 'bg-[#050505]' : 'bg-[#f8fafc]'
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
            Intelligent Security Solutions
          </h2>
          <p className={`max-w-2xl mx-auto ${
            theme === 'dark' ? 'text-[#cbd5e1]' : 'text-[#475569]'
          }`}>
            Advanced AI models trained on real-world security scenarios
          </p>
        </motion.div>

        {/* Capabilities Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {capabilities.map((capability, index) => (
            <motion.div
              key={capability.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              className={`p-8 rounded-2xl border relative overflow-hidden group hover:scale-105 transition-all cursor-pointer ${
                theme === 'dark'
                  ? 'bg-[#0a0a0a] border-[#06b6d4]/20 hover:border-[#06b6d4]/60 hover:shadow-xl hover:shadow-[#06b6d4]/10'
                  : 'bg-white border-gray-200 hover:border-[#0ea5e9]/60 hover:shadow-xl hover:shadow-[#0ea5e9]/10'
              }`}
            >
              {/* Gradient Background on Hover */}
              <div className={`absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity ${
                theme === 'dark'
                  ? 'bg-gradient-to-br from-[#06b6d4]/5 to-transparent'
                  : 'bg-gradient-to-br from-[#0ea5e9]/5 to-transparent'
              }`} />
              
              <div className="relative z-10">
                {/* Icon */}
                <div className={`w-14 h-14 rounded-xl flex items-center justify-center mb-6 transition-all group-hover:scale-110 ${
                  theme === 'dark'
                    ? 'bg-gradient-to-br from-[#06b6d4]/20 to-[#0ea5e9]/20 border border-[#06b6d4]/30'
                    : 'bg-gradient-to-br from-[#0ea5e9]/20 to-[#06b6d4]/20 border border-[#0ea5e9]/30'
                }`}>
                  <capability.icon className="w-7 h-7 text-[#06b6d4]" />
                </div>

                {/* Content */}
                <h3 className={`mb-3 ${theme === 'dark' ? 'text-[#f8fafc]' : 'text-[#0f172a]'}`}>
                  {capability.title}
                </h3>
                <p className={`mb-4 ${theme === 'dark' ? 'text-[#cbd5e1]' : 'text-[#475569]'}`}>
                  {capability.description}
                </p>

                {/* Metric Badge */}
                <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm ${
                  theme === 'dark'
                    ? 'bg-[#06b6d4]/10 text-[#06b6d4] border border-[#06b6d4]/20'
                    : 'bg-[#0ea5e9]/10 text-[#0ea5e9] border border-[#0ea5e9]/20'
                }`}>
                  {capability.metric}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
