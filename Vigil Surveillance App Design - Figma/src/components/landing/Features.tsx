import { motion, useScroll, useTransform } from 'motion/react';
import { Car, Shield, Users } from 'lucide-react';
import { ImageWithFallback } from './figma/ImageWithFallback';

interface FeaturesProps {
  theme: 'light' | 'dark';
}

export function Features({ theme }: FeaturesProps) {
  // Parallax scrolling effect for background
  const { scrollY } = useScroll();
  const y = useTransform(scrollY, [0, 3000], [0, -300]);
  const backgroundOpacity = useTransform(scrollY, [800, 1500, 3500], [0, 0.6, 0]);

  const features = [
    {
      icon: Car,
      title: 'Accident Detection',
      tagline: 'Faster Emergency Response',
      description: 'Instantly detect vehicle collisions and traffic incidents. Reduce emergency response times by up to 40% with automated alerts to your security team.',
      benefits: [
        'Minimize accident impact with rapid detection',
        'Lower insurance costs through quick response',
        'Improve road safety and traffic flow',
        'Automated emergency service alerts'
      ],
      image: 'https://images.unsplash.com/photo-1673187139211-1e7ec3dd60ec?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxjYXIlMjBhY2NpZGVudCUyMG1pbm9yJTIwY29sbGlzaW9ufGVufDF8fHx8MTc2ODM5NTE3NXww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral',
      stat: '95.8%',
      statLabel: 'Accuracy Rate'
    },
    {
      icon: Shield,
      title: 'Violence Detection',
      tagline: 'Protect Your People',
      description: 'Real-time monitoring for aggressive behavior and violent incidents. Create safer environments for employees, customers, and the public.',
      benefits: [
        'Prevent incidents before they escalate',
        'Protect staff and customer safety',
        'Reduce liability and legal risks',
        'Instant security team notifications'
      ],
      image: 'https://images.unsplash.com/photo-1591073214788-90028ecf6798?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxwZW9wbGUlMjBmaWdodGluZyUyMGNvbmZyb250YXRpb24lMjBzZWN1cml0eXxlbnwxfHx8fDE3NjgzOTU0MDd8MA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral',
      stat: '96.2%',
      statLabel: 'Accuracy Rate'
    },
    {
      icon: Users,
      title: 'People Counting',
      tagline: 'Optimize Your Operations',
      description: 'Track foot traffic and occupancy levels in real-time. Make data-driven decisions to optimize staffing, marketing, and customer experience.',
      benefits: [
        'Optimize staff scheduling and reduce costs',
        'Monitor capacity limits and compliance',
        'Understand customer flow patterns',
        'Measure marketing campaign effectiveness'
      ],
      image: 'https://images.unsplash.com/photo-1698487346343-073614df5ac4?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxwZW9wbGUlMjB3YWxraW5nJTIwZGV0ZWN0aW9uJTIwY291bnRpbmd8ZW58MXx8fHwxNzY4Mzk1MzQ0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral',
      stat: '97.1%',
      statLabel: 'Accuracy Rate'
    }
  ];

  return (
    <section 
      id="features"
      className={`relative py-24 px-4 sm:px-6 lg:px-8 overflow-hidden ${
        theme === 'dark' ? 'bg-[#0a0a0a]' : 'bg-white'
      }`}
    >
      {/* Parallax Background */}
      <motion.div 
        style={{ y, opacity: backgroundOpacity }}
        className="absolute inset-0 z-0"
      >
        <div className="relative w-full h-full">
          <ImageWithFallback
            src="https://images.unsplash.com/photo-1757323148943-2ae82a19ec9f?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxzdXJ2ZWlsbGFuY2UlMjBtb25pdG9ycyUyMHNjcmVlbnN8ZW58MXx8fHwxNzY4NDA0MzczfDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
            alt="Surveillance Monitors"
            className="w-full h-full object-cover scale-110"
          />
          
          {/* Gradient overlays */}
          <div className={`absolute inset-0 ${
            theme === 'dark' 
              ? 'bg-gradient-to-b from-[#0a0a0a]/90 via-[#0a0a0a]/85 to-[#0a0a0a]/90' 
              : 'bg-gradient-to-b from-white/90 via-white/85 to-white/90'
          }`} />
          
          {/* Brand color accent overlay */}
          <div className="absolute inset-0 bg-gradient-to-br from-[#06b6d4]/10 via-transparent to-[#0ea5e9]/10" />
        </div>
      </motion.div>

      <div className="max-w-7xl mx-auto relative z-10">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-20"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#06b6d4]/10 border border-[#06b6d4]/30 mb-6">
            <span className="text-sm bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9] bg-clip-text text-transparent font-medium">
              Core Capabilities
            </span>
          </div>
          <h2 className={`text-4xl sm:text-5xl lg:text-6xl mb-6 ${
            theme === 'dark' ? 'text-white' : 'text-gray-900'
          }`}>
            Three Powerful Features.<br />
            One Intelligent Platform.
          </h2>
          <p className={`text-xl max-w-3xl mx-auto ${
            theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
          }`}>
            Transform your existing camera infrastructure into an intelligent safety and analytics system
          </p>
        </motion.div>

        {/* Features */}
        <div className="space-y-32">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className={`grid lg:grid-cols-2 gap-12 items-center ${
                index % 2 === 1 ? 'lg:flex-row-reverse' : ''
              }`}
            >
              {/* Content */}
              <div className={index % 2 === 1 ? 'lg:order-2' : ''}>
                {/* Icon */}
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#06b6d4] to-[#0ea5e9] flex items-center justify-center mb-6">
                  <feature.icon className="w-8 h-8 text-white" />
                </div>

                {/* Title */}
                <h3 className={`text-3xl sm:text-4xl mb-3 ${
                  theme === 'dark' ? 'text-white' : 'text-gray-900'
                }`}>
                  {feature.title}
                </h3>
                
                {/* Tagline */}
                <p className="text-xl bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9] bg-clip-text text-transparent mb-6">
                  {feature.tagline}
                </p>

                {/* Description */}
                <p className={`text-lg mb-8 ${
                  theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
                }`}>
                  {feature.description}
                </p>

                {/* Benefits */}
                <ul className="space-y-4 mb-8">
                  {feature.benefits.map((benefit, idx) => (
                    <li key={idx} className="flex items-start gap-3">
                      <div className="w-6 h-6 rounded-full bg-[#06b6d4]/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <svg className="w-4 h-4 text-[#06b6d4]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      </div>
                      <span className={theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}>
                        {benefit}
                      </span>
                    </li>
                  ))}
                </ul>

                {/* Stat Badge */}
                <div className={`inline-flex items-baseline gap-3 px-6 py-4 rounded-xl ${
                  theme === 'dark' 
                    ? 'bg-gradient-to-br from-[#06b6d4]/10 to-[#0ea5e9]/5 border border-[#06b6d4]/30'
                    : 'bg-gradient-to-br from-[#06b6d4]/10 to-[#0ea5e9]/10 border border-[#06b6d4]/30'
                }`}>
                  <span className="text-4xl bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9] bg-clip-text text-transparent font-bold">
                    {feature.stat}
                  </span>
                  <span className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
                    {feature.statLabel}
                  </span>
                </div>
              </div>

              {/* Image */}
              <div className={index % 2 === 1 ? 'lg:order-1' : ''}>
                <div className="relative rounded-2xl overflow-hidden shadow-2xl group">
                  <ImageWithFallback
                    src={feature.image}
                    alt={feature.title}
                    className="w-full h-[400px] object-cover group-hover:scale-105 transition-transform duration-500"
                  />
                  <div className={`absolute inset-0 ${
                    theme === 'dark'
                      ? 'bg-gradient-to-t from-[#0a0a0a]/60 to-transparent'
                      : 'bg-gradient-to-t from-black/20 to-transparent'
                  }`} />
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
