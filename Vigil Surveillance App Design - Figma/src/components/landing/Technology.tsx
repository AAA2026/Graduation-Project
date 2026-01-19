import { motion, useScroll, useTransform } from 'motion/react';
import { Brain, Zap, Lock, Cloud } from 'lucide-react';
import { ImageWithFallback } from './figma/ImageWithFallback';

interface TechnologyProps {
  theme: 'light' | 'dark';
}

export function Technology({ theme }: TechnologyProps) {
  // Parallax scrolling effect for background
  const { scrollY } = useScroll();
  const y = useTransform(scrollY, [0, 5000], [0, -300]);
  const backgroundOpacity = useTransform(scrollY, [2800, 3500, 5000], [0, 0.6, 0]);

  const features = [
    {
      icon: Brain,
      title: 'Advanced AI Models',
      description: 'State-of-the-art deep learning technology for accurate, real-time detection across all three core features.',
      stat: '95%+ Accuracy'
    },
    {
      icon: Zap,
      title: 'Lightning Fast',
      description: 'Process video feeds in real-time with sub-second response times. Get instant alerts when it matters most.',
      stat: '<1s Response'
    },
    {
      icon: Lock,
      title: 'Enterprise Security',
      description: 'Bank-level encryption and compliance with global data protection standards. Your data stays secure.',
      stat: 'SOC 2 Certified'
    },
    {
      icon: Cloud,
      title: 'Seamless Integration',
      description: 'Works with your existing camera infrastructure. Cloud or on-premise deployment options available.',
      stat: '48hr Setup'
    }
  ];

  return (
    <section 
      id="technology"
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
            src="https://images.unsplash.com/photo-1739054730201-4b6463484e3c?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxkYXRhJTIwY2VudGVyJTIwYmx1ZXxlbnwxfHx8fDE3Njg0MDQzNzR8MA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
            alt="Data Center Infrastructure"
            className="w-full h-full object-cover scale-110"
          />
          
          {/* Gradient overlays */}
          <div className={`absolute inset-0 ${
            theme === 'dark' 
              ? 'bg-gradient-to-b from-[#050505]/95 via-[#050505]/90 to-[#050505]/95' 
              : 'bg-gradient-to-b from-gray-50/95 via-gray-50/90 to-gray-50/95'
          }`} />
          
          {/* Brand color accent overlay */}
          <div className="absolute inset-0 bg-gradient-to-tl from-[#06b6d4]/15 via-transparent to-[#0ea5e9]/15" />
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
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#06b6d4]/10 border border-[#06b6d4]/30 mb-6">
            <span className="text-sm bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9] bg-clip-text text-transparent font-medium">
              Technology That Delivers
            </span>
          </div>
          <h2 className={`text-4xl sm:text-5xl mb-6 ${
            theme === 'dark' ? 'text-white' : 'text-gray-900'
          }`}>
            Powerful. Reliable. Easy to Deploy.
          </h2>
          <p className={`text-xl max-w-3xl mx-auto ${
            theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
          }`}>
            Enterprise-grade AI surveillance that works with your existing infrastructure
          </p>
        </motion.div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className={`p-8 rounded-2xl transition-all group ${
                theme === 'dark'
                  ? 'bg-[#0a0a0a] border border-[#06b6d4]/20 hover:border-[#06b6d4]/50 hover:shadow-xl hover:shadow-[#06b6d4]/20'
                  : 'bg-white border border-gray-200 hover:border-[#06b6d4]/50 hover:shadow-xl hover:shadow-[#06b6d4]/10'
              }`}
            >
              {/* Icon */}
              <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-[#06b6d4] to-[#0ea5e9] flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <feature.icon className="w-7 h-7 text-white" />
              </div>

              {/* Title */}
              <h3 className={`text-xl mb-3 ${
                theme === 'dark' ? 'text-white' : 'text-gray-900'
              }`}>
                {feature.title}
              </h3>

              {/* Description */}
              <p className={`text-sm mb-4 leading-relaxed ${
                theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
              }`}>
                {feature.description}
              </p>

              {/* Stat */}
              <div className={`inline-flex px-3 py-1 rounded-full ${
                theme === 'dark'
                  ? 'bg-[#06b6d4]/10 border border-[#06b6d4]/30'
                  : 'bg-[#06b6d4]/10 border border-[#06b6d4]/30'
              }`}>
                <span className="text-sm bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9] bg-clip-text text-transparent font-medium">
                  {feature.stat}
                </span>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Integration Info */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className={`mt-16 p-8 rounded-2xl text-center ${
            theme === 'dark'
              ? 'bg-gradient-to-br from-[#06b6d4]/10 to-[#0ea5e9]/5 border border-[#06b6d4]/30'
              : 'bg-gradient-to-br from-[#06b6d4]/10 to-[#0ea5e9]/5 border border-[#06b6d4]/30'
          }`}
        >
          <h3 className={`text-2xl mb-4 ${
            theme === 'dark' ? 'text-white' : 'text-gray-900'
          }`}>
            Compatible with Your Existing Cameras
          </h3>
          <p className={`text-lg max-w-2xl mx-auto ${
            theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
          }`}>
            VIGIL integrates seamlessly with RTSP, HTTP, and major camera brands including Hikvision, Dahua, Axis, and more
          </p>
        </motion.div>
      </div>
    </section>
  );
}
