import { motion, useScroll, useTransform } from 'motion/react';
import { Building2, GraduationCap, ShoppingBag, Train } from 'lucide-react';
import { ImageWithFallback } from './figma/ImageWithFallback';

interface UseCasesProps {
  theme: 'light' | 'dark';
}

export function UseCases({ theme }: UseCasesProps) {
  // Parallax scrolling effect for background
  const { scrollY } = useScroll();
  const y = useTransform(scrollY, [0, 6000], [0, -300]);
  const backgroundOpacity = useTransform(scrollY, [3500, 4200, 6000], [0, 0.6, 0]);

  const useCases = [
    {
      icon: Building2,
      title: 'Corporate Facilities',
      description: 'Enhance workplace safety and optimize building operations with intelligent monitoring.',
      benefits: [
        'Reduce workplace incidents',
        'Optimize security staffing',
        'Monitor occupancy compliance'
      ]
    },
    {
      icon: GraduationCap,
      title: 'Educational Campuses',
      description: 'Create safer learning environments with proactive threat detection and crowd management.',
      benefits: [
        'Protect students and staff',
        'Faster emergency response',
        'Track campus traffic patterns'
      ]
    },
    {
      icon: ShoppingBag,
      title: 'Retail & Hospitality',
      description: 'Improve customer experience and operational efficiency through advanced analytics.',
      benefits: [
        'Optimize staff scheduling',
        'Understand customer flow',
        'Enhance loss prevention'
      ]
    },
    {
      icon: Train,
      title: 'Transportation Hubs',
      description: 'Secure high-traffic areas and manage crowds effectively in real-time.',
      benefits: [
        'Detect accidents instantly',
        'Manage crowd density',
        'Improve passenger safety'
      ]
    }
  ];

  return (
    <section 
      className={`relative py-24 px-4 sm:px-6 lg:px-8 overflow-hidden ${
        theme === 'dark' ? 'bg-[#050505]' : 'bg-gray-50'
      }`}
    >
      {/* Parallax Background */}
      <motion.div 
        style={{ y, opacity: backgroundOpacity }}
        className="absolute inset-0 z-0"
      >
        <div className="relative w-full h-full">
          <ImageWithFallback
            src="https://images.unsplash.com/photo-1711733654648-8bef15e30e94?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxjY3R2JTIwY2FtZXJhJTIwbmlnaHR8ZW58MXx8fHwxNzY4NDA0MzczfDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
            alt="CCTV Camera Night Vision"
            className="w-full h-full object-cover scale-110"
          />
          
          {/* Gradient overlays */}
          <div className={`absolute inset-0 ${
            theme === 'dark' 
              ? 'bg-gradient-to-b from-[#050505]/93 via-[#050505]/88 to-[#050505]/93' 
              : 'bg-gradient-to-b from-gray-50/93 via-gray-50/88 to-gray-50/93'
          }`} />
          
          {/* Brand color accent overlay */}
          <div className="absolute inset-0 bg-gradient-to-bl from-[#06b6d4]/12 via-transparent to-[#0ea5e9]/12" />
        </div>
      </motion.div>

      <div className="max-w-7xl mx-auto relative z-10">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className={`text-4xl sm:text-5xl mb-6 ${
            theme === 'dark' ? 'text-white' : 'text-gray-900'
          }`}>
            Built for Every Industry
          </h2>
          <p className={`text-xl max-w-3xl mx-auto ${
            theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
          }`}>
            VIGIL adapts to your unique security and operational needs
          </p>
        </motion.div>

        {/* Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {useCases.map((useCase, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className={`group p-8 rounded-2xl transition-all hover:scale-105 ${
                theme === 'dark'
                  ? 'bg-[#0a0a0a] border border-[#06b6d4]/20 hover:border-[#06b6d4]/50 hover:shadow-xl hover:shadow-[#06b6d4]/20'
                  : 'bg-white border border-gray-200 hover:border-[#06b6d4]/50 hover:shadow-xl hover:shadow-[#06b6d4]/10'
              }`}
            >
              {/* Icon */}
              <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-[#06b6d4] to-[#0ea5e9] flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <useCase.icon className="w-7 h-7 text-white" />
              </div>

              {/* Title */}
              <h3 className={`text-2xl mb-4 ${
                theme === 'dark' ? 'text-white' : 'text-gray-900'
              }`}>
                {useCase.title}
              </h3>

              {/* Description */}
              <p className={`text-base mb-6 ${
                theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
              }`}>
                {useCase.description}
              </p>

              {/* Benefits */}
              <ul className="space-y-2">
                {useCase.benefits.map((benefit, idx) => (
                  <li key={idx} className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9]" />
                    <span className={`text-sm ${
                      theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
                    }`}>
                      {benefit}
                    </span>
                  </li>
                ))}
              </ul>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
