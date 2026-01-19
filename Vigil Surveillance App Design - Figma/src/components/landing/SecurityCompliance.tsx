import { motion } from 'motion/react';
import { Shield, Lock, Database, Eye, FileCheck, Server } from 'lucide-react';

interface SecurityComplianceProps {
  theme: 'light' | 'dark';
}

export function SecurityCompliance({ theme }: SecurityComplianceProps) {
  const features = [
    {
      icon: Shield,
      title: 'Enterprise Security',
      description: 'End-to-end encryption for all data transmission and storage'
    },
    {
      icon: Lock,
      title: 'Access Control',
      description: 'Role-based permissions with multi-factor authentication'
    },
    {
      icon: Database,
      title: 'Data Privacy',
      description: 'GDPR and CCPA compliant with automatic data anonymization'
    },
    {
      icon: Eye,
      title: 'Audit Logging',
      description: 'Complete audit trails for all system activities'
    },
    {
      icon: FileCheck,
      title: 'Compliance Ready',
      description: 'SOC 2, ISO 27001 certified infrastructure'
    },
    {
      icon: Server,
      title: 'Flexible Deployment',
      description: 'Cloud or on-premise options to meet your needs'
    }
  ];

  const certifications = [
    { name: 'SOC 2', status: 'Certified' },
    { name: 'ISO 27001', status: 'Certified' },
    { name: 'GDPR', status: 'Compliant' },
    { name: 'CCPA', status: 'Compliant' }
  ];

  return (
    <section 
      className={`relative py-24 px-4 sm:px-6 lg:px-8 ${
        theme === 'dark' ? 'bg-[#050505]' : 'bg-gray-50'
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
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#06b6d4]/10 border border-[#06b6d4]/30 mb-6">
            <Shield className="w-4 h-4 text-[#06b6d4]" />
            <span className="text-sm text-[#06b6d4] font-medium">Enterprise-Grade Security</span>
          </div>
          <h2 className={`text-4xl sm:text-5xl mb-4 ${
            theme === 'dark' ? 'text-white' : 'text-gray-900'
          }`}>
            Your Security is Our Priority
          </h2>
          <p className={`text-xl max-w-2xl mx-auto ${
            theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
          }`}>
            Built with industry-leading security standards and compliance frameworks
          </p>
        </motion.div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className={`p-6 rounded-xl transition-all ${
                theme === 'dark'
                  ? 'bg-[#06b6d4]/5 border border-[#06b6d4]/20 hover:border-[#06b6d4]/50'
                  : 'bg-white border border-gray-200 hover:border-[#06b6d4]/50 hover:shadow-lg'
              }`}
            >
              <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-[#06b6d4] to-[#0ea5e9] flex items-center justify-center mb-4">
                <feature.icon className="w-6 h-6 text-white" />
              </div>
              <h3 className={`text-lg mb-2 ${
                theme === 'dark' ? 'text-white' : 'text-gray-900'
              }`}>
                {feature.title}
              </h3>
              <p className={`text-sm ${
                theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
              }`}>
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>

        {/* Certifications */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className={`p-8 rounded-2xl ${
            theme === 'dark'
              ? 'bg-gradient-to-br from-[#06b6d4]/10 to-[#0ea5e9]/5 border border-[#06b6d4]/30'
              : 'bg-gradient-to-br from-[#06b6d4]/10 to-[#0ea5e9]/5 border border-[#06b6d4]/30'
          }`}
        >
          <h3 className={`text-2xl mb-8 text-center ${
            theme === 'dark' ? 'text-white' : 'text-gray-900'
          }`}>
            Certifications & Compliance
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {certifications.map((cert, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.05 }}
                className={`text-center p-4 rounded-lg ${
                  theme === 'dark'
                    ? 'bg-[#0a0a0a]/50 border border-[#06b6d4]/20'
                    : 'bg-white border border-[#06b6d4]/20'
                }`}
              >
                <div className={`text-lg mb-1 ${
                  theme === 'dark' ? 'text-white' : 'text-gray-900'
                }`}>
                  {cert.name}
                </div>
                <div className="inline-flex items-center gap-1 px-3 py-1 rounded-full bg-[#06b6d4]/20 border border-[#06b6d4]/30">
                  <div className="w-2 h-2 rounded-full bg-[#06b6d4]" />
                  <span className="text-xs text-[#06b6d4]">{cert.status}</span>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
