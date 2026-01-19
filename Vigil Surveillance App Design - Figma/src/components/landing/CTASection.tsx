import { motion, useScroll, useTransform } from 'motion/react';
import { ArrowRight, Shield, Zap, CheckCircle } from 'lucide-react';
import { ImageWithFallback } from './figma/ImageWithFallback';

interface CTASectionProps {
  theme: 'light' | 'dark';
  onBookDemo: () => void;
}

export function CTASection({ theme, onBookDemo }: CTASectionProps) {
  // Parallax scrolling effect for background
  const { scrollY } = useScroll();
  const y = useTransform(scrollY, [0, 8000], [0, -300]);
  const backgroundOpacity = useTransform(scrollY, [5500, 6200, 8000], [0, 0.6, 0]);

  return (
    <section 
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
            src="https://images.unsplash.com/photo-1697057406467-60340e993e6e?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx1cmJhbiUyMHN1cnZlaWxsYW5jZSUyMGNpdHl8ZW58MXx8fHwxNzY4NDA0Mzc1fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
            alt="Urban Surveillance"
            className="w-full h-full object-cover scale-110"
          />
          
          {/* Gradient overlays */}
          <div className={`absolute inset-0 ${
            theme === 'dark' 
              ? 'bg-gradient-to-b from-[#0a0a0a]/92 via-[#0a0a0a]/88 to-[#0a0a0a]/92' 
              : 'bg-gradient-to-b from-white/92 via-white/88 to-white/92'
          }`} />
          
          {/* Brand color accent overlay */}
          <div className="absolute inset-0 bg-gradient-to-tr from-[#06b6d4]/12 via-transparent to-[#0ea5e9]/12" />
        </div>
      </motion.div>

      <div className="max-w-7xl mx-auto relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-[#06b6d4] to-[#0ea5e9] p-12 md:p-16"
        >
          {/* Background Pattern */}
          <div className="absolute inset-0 opacity-10">
            <div className="absolute inset-0" style={{
              backgroundImage: `radial-gradient(circle at 2px 2px, white 1px, transparent 0)`,
              backgroundSize: '40px 40px'
            }} />
          </div>

          <div className="relative z-10 grid md:grid-cols-2 gap-12 items-center">
            {/* Left Content */}
            <div>
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.2 }}
              >
                <h2 className="text-4xl sm:text-5xl text-white mb-6 font-bold">
                  Ready to Transform Your Security?
                </h2>
                <p className="text-lg text-white/90 mb-8">
                  Join leading organizations using VIGIL to protect their people and optimize operations.
                </p>

                {/* Benefits */}
                <div className="space-y-4 mb-8">
                  {[
                    'Deploy in under 48 hours',
                    'Works with existing cameras',
                    '30-day money-back guarantee',
                    'Free consultation & setup'
                  ].map((benefit, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      viewport={{ once: true }}
                      transition={{ delay: 0.3 + index * 0.1 }}
                      className="flex items-center gap-3"
                    >
                      <div className="w-6 h-6 rounded-full bg-white/20 flex items-center justify-center flex-shrink-0">
                        <CheckCircle className="w-4 h-4 text-white" />
                      </div>
                      <span className="text-white/90">{benefit}</span>
                    </motion.div>
                  ))}
                </div>

                {/* CTA Buttons */}
                <div className="flex flex-col sm:flex-row gap-4">
                  <motion.button
                    whileHover={{ scale: 1.05, y: -2 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={onBookDemo}
                    className="group flex items-center justify-center gap-2 px-8 py-4 bg-white text-[#06b6d4] rounded-lg hover:bg-white/90 transition-all shadow-lg hover:shadow-xl font-medium"
                  >
                    <span>Schedule a Demo</span>
                    <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                  </motion.button>
                  <motion.button 
                    whileHover={{ scale: 1.05, y: -2 }}
                    whileTap={{ scale: 0.95 }}
                    className="px-8 py-4 border-2 border-white text-white rounded-lg hover:bg-white/10 transition-all font-medium"
                  >
                    Contact Sales
                  </motion.button>
                </div>
              </motion.div>
            </div>

            {/* Right Stats */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.4 }}
              className="grid grid-cols-2 gap-6"
            >
              <motion.div 
                whileHover={{ scale: 1.05, y: -4 }}
                className="p-6 rounded-2xl bg-white/10 backdrop-blur-sm border border-white/20 hover:bg-white/15 transition-all cursor-default"
              >
                <motion.div
                  whileHover={{ rotate: 360 }}
                  transition={{ duration: 0.6 }}
                >
                  <Shield className="w-10 h-10 text-white mb-4" />
                </motion.div>
                <div className="text-3xl text-white mb-2 font-bold">100+</div>
                <div className="text-sm text-white/80">Businesses Protected</div>
              </motion.div>
              <motion.div 
                whileHover={{ scale: 1.05, y: -4 }}
                className="p-6 rounded-2xl bg-white/10 backdrop-blur-sm border border-white/20 hover:bg-white/15 transition-all cursor-default"
              >
                <motion.div
                  whileHover={{ rotate: 360 }}
                  transition={{ duration: 0.6 }}
                >
                  <Zap className="w-10 h-10 text-white mb-4" />
                </motion.div>
                <div className="text-3xl text-white mb-2 font-bold">99.9%</div>
                <div className="text-sm text-white/80">System Uptime</div>
              </motion.div>
              <motion.div 
                whileHover={{ scale: 1.05, y: -4 }}
                className="p-6 rounded-2xl bg-white/10 backdrop-blur-sm border border-white/20 hover:bg-white/15 transition-all cursor-default"
              >
                <div className="text-3xl text-white mb-2 font-bold">&lt;48h</div>
                <div className="text-sm text-white/80">Deployment Time</div>
              </motion.div>
              <motion.div 
                whileHover={{ scale: 1.05, y: -4 }}
                className="p-6 rounded-2xl bg-white/10 backdrop-blur-sm border border-white/20 hover:bg-white/15 transition-all cursor-default"
              >
                <div className="text-3xl text-white mb-2 font-bold">24/7</div>
                <div className="text-sm text-white/80">Expert Support</div>
              </motion.div>
            </motion.div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
