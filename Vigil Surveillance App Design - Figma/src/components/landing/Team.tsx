import { motion } from 'motion/react';
import { Code, Database, Brain, Shield, Zap, Cpu } from 'lucide-react';

interface TeamProps {
  theme: 'light' | 'dark';
}

export function Team({ theme }: TeamProps) {
  const expertiseAreas = [
    {
      icon: Brain,
      title: 'AI & Machine Learning',
      description: 'State-of-the-art deep learning architectures and computer vision models',
      color: 'from-purple-500 to-pink-500',
      stats: '99.2% Accuracy'
    },
    {
      icon: Cpu,
      title: 'Real-time Processing',
      description: 'High-performance inference engines with sub-second response times',
      color: 'from-blue-500 to-cyan-500',
      stats: '<100ms Latency'
    },
    {
      icon: Database,
      title: 'Scalable Infrastructure',
      description: 'Enterprise-grade cloud architecture supporting millions of events daily',
      color: 'from-green-500 to-emerald-500',
      stats: '10M+ Events/Day'
    },
    {
      icon: Shield,
      title: 'Security & Compliance',
      description: 'Military-grade encryption with full regulatory compliance standards',
      color: 'from-red-500 to-orange-500',
      stats: 'ISO 27001 Certified'
    },
    {
      icon: Zap,
      title: 'Edge Computing',
      description: 'Optimized for edge deployment with minimal hardware requirements',
      color: 'from-yellow-500 to-amber-500',
      stats: 'Low Resource'
    },
    {
      icon: Code,
      title: 'API Integration',
      description: 'RESTful APIs with comprehensive documentation and SDK support',
      color: 'from-indigo-500 to-purple-500',
      stats: '99.99% Uptime'
    }
  ];

  return (
    <section 
      id="team" 
      className={`relative py-24 px-4 sm:px-6 lg:px-8 transition-colors duration-300 ${ 
        theme === 'dark' 
          ? 'bg-gradient-to-b from-[#050505] via-[#06b6d4]/5 to-[#050505]' 
          : 'bg-gradient-to-b from-white via-[#06b6d4]/5 to-white'
      }`}
    >
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
            Technology & Expertise
          </h2>
          <p className={`text-lg max-w-2xl mx-auto ${
            theme === 'dark' ? 'text-[#cbd5e1]' : 'text-gray-600'
          }`}>
            Built on cutting-edge technology with enterprise-grade reliability
          </p>
        </motion.div>

        {/* Expertise Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {expertiseAreas.map((area, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ y: -8, scale: 1.02 }}
              className={`group relative p-8 rounded-2xl border transition-all duration-300 ${
                theme === 'dark'
                  ? 'bg-[#06b6d4]/5 border-[#06b6d4]/20 hover:border-[#06b6d4]/60 hover:shadow-2xl hover:shadow-[#06b6d4]/30'
                  : 'bg-white border-[#06b6d4]/30 hover:border-[#06b6d4]/60 hover:shadow-2xl hover:shadow-[#06b6d4]/20'
              }`}
            >
              {/* Animated gradient overlay on hover */}
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-[#06b6d4]/0 to-[#0ea5e9]/0 group-hover:from-[#06b6d4]/10 group-hover:to-[#0ea5e9]/5 transition-all duration-300" />
              
              <div className="relative z-10">
                {/* Icon */}
                <motion.div 
                  whileHover={{ rotate: 360, scale: 1.1 }}
                  transition={{ duration: 0.6 }}
                  className={`w-16 h-16 rounded-xl bg-gradient-to-br ${area.color} flex items-center justify-center mb-6 shadow-lg`}
                >
                  <area.icon className="w-8 h-8 text-white" />
                </motion.div>

                {/* Title */}
                <h3 className={`text-xl font-semibold mb-3 ${
                  theme === 'dark' ? 'text-white' : 'text-gray-900'
                }`}>
                  {area.title}
                </h3>

                {/* Description */}
                <p className={`text-sm mb-4 ${
                  theme === 'dark' ? 'text-[#cbd5e1]' : 'text-gray-600'
                }`}>
                  {area.description}
                </p>

                {/* Stats Badge */}
                <div className="inline-block px-3 py-1 rounded-full bg-gradient-to-r from-[#06b6d4]/20 to-[#0ea5e9]/20 border border-[#06b6d4]/30">
                  <span className="text-xs font-semibold bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9] bg-clip-text text-transparent">
                    {area.stats}
                  </span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Enterprise Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-6"
        >
          {[
            { value: '50K+', label: 'Training Hours' },
            { value: '99.9%', label: 'System Uptime' },
            { value: '24/7', label: 'Live Monitoring' },
            { value: 'SOC 2', label: 'Compliant' }
          ].map((stat, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.05, y: -4 }}
              className={`text-center p-6 rounded-xl border cursor-default transition-all duration-300 ${
                theme === 'dark'
                  ? 'bg-gradient-to-br from-[#06b6d4]/10 to-[#0ea5e9]/5 border-[#06b6d4]/30 hover:border-[#06b6d4]/60 hover:shadow-lg hover:shadow-[#06b6d4]/20'
                  : 'bg-gradient-to-br from-[#06b6d4]/5 to-[#0ea5e9]/5 border-[#06b6d4]/30 hover:border-[#06b6d4]/60 hover:shadow-lg hover:shadow-[#06b6d4]/20'
              }`}
            >
              <motion.div 
                className="text-3xl font-bold bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9] bg-clip-text text-transparent mb-2"
                whileHover={{ scale: 1.1 }}
              >
                {stat.value}
              </motion.div>
              <div className={`text-sm ${
                theme === 'dark' ? 'text-[#cbd5e1]' : 'text-gray-600'
              }`}>{stat.label}</div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
