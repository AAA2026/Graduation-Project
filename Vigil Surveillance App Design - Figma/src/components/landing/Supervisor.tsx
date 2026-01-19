import { motion } from 'motion/react';
import { Award, Shield, Zap, CheckCircle } from 'lucide-react';

interface SupervisorProps {
  theme: 'light' | 'dark';
}

export function Supervisor({ theme }: SupervisorProps) {
  const certifications = [
    { icon: Shield, text: 'Enterprise Security Compliance' },
    { icon: Award, text: 'ISO 27001 Standards' },
    { icon: Zap, text: 'Real-Time Performance Validated' },
    { icon: CheckCircle, text: 'Industry Best Practices' }
  ];

  return (
    <section 
      className={`relative py-16 px-4 sm:px-6 lg:px-8 border-y ${
        theme === 'dark' 
          ? 'bg-gradient-to-r from-[#06b6d4]/10 via-[#0ea5e9]/5 to-[#06b6d4]/10 border-[#06b6d4]/20' 
          : 'bg-gradient-to-r from-[#06b6d4]/20 via-[#0ea5e9]/10 to-[#06b6d4]/20 border-[#06b6d4]/30'
      }`}
    >
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-6 py-3 rounded-full bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9] mb-6">
            <Shield className="w-5 h-5 text-white" />
            <span className="text-white">Enterprise-Grade Platform</span>
          </div>

          {/* Title */}
          <h3 className="text-3xl sm:text-4xl mb-4 text-white">
            Built for Mission-Critical Operations
          </h3>

          {/* Subtitle */}
          <p className="text-lg text-[#cbd5e1] max-w-3xl mx-auto">
            Trusted by security professionals worldwide for reliability, accuracy, and performance
          </p>
        </motion.div>

        {/* Certifications Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {certifications.map((cert, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="flex items-center gap-4 p-4 rounded-xl bg-[#0a0a0a]/50 border border-[#06b6d4]/30 backdrop-blur-sm"
            >
              <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-[#06b6d4] to-[#0ea5e9] flex items-center justify-center flex-shrink-0">
                <cert.icon className="w-6 h-6 text-white" />
              </div>
              <span className="text-sm text-[#cbd5e1]">{cert.text}</span>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
