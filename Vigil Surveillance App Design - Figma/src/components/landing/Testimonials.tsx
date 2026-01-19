import { motion, useScroll, useTransform } from 'motion/react';
import { Quote, Building2, School, ShoppingBag } from 'lucide-react';
import { ImageWithFallback } from './figma/ImageWithFallback';

interface TestimonialsProps {
  theme: 'light' | 'dark';
}

export function Testimonials({ theme }: TestimonialsProps) {
  // Parallax scrolling effect for background
  const { scrollY } = useScroll();
  const y = useTransform(scrollY, [0, 7000], [0, -300]);
  const backgroundOpacity = useTransform(scrollY, [4500, 5200, 7000], [0, 0.6, 0]);

  const testimonials = [
    {
      icon: Building2,
      quote: "VIGIL's accident detection has reduced our emergency response times by 45%. The instant alerts have been crucial for protecting our employees and assets.",
      company: "Global Logistics Corporation",
      role: "Safety Director",
      industry: "Transportation & Logistics",
      stat: "45% faster response"
    },
    {
      icon: School,
      quote: "The violence detection system gives us peace of mind. We can monitor our entire campus in real-time and respond to incidents before they escalate.",
      company: "University Campus Security",
      role: "Director of Campus Safety",
      industry: "Education",
      stat: "96% detection rate"
    },
    {
      icon: ShoppingBag,
      quote: "People counting analytics transformed our staffing strategy. We've reduced labor costs by 30% while improving customer service during peak hours.",
      company: "National Retail Chain",
      role: "Operations Director",
      industry: "Retail",
      stat: "30% cost savings"
    }
  ];

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
            src="https://images.unsplash.com/photo-1764383381198-0fff09cb38b1?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxzZWN1cml0eSUyMGNvbnRyb2wlMjBjZW50ZXJ8ZW58MXx8fHwxNzY4NDA0MzczfDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
            alt="Security Control Center"
            className="w-full h-full object-cover scale-110"
          />
          
          {/* Gradient overlays */}
          <div className={`absolute inset-0 ${
            theme === 'dark' 
              ? 'bg-gradient-to-b from-[#0a0a0a]/94 via-[#0a0a0a]/90 to-[#0a0a0a]/94' 
              : 'bg-gradient-to-b from-white/94 via-white/90 to-white/94'
          }`} />
          
          {/* Brand color accent overlay */}
          <div className="absolute inset-0 bg-gradient-to-br from-[#06b6d4]/8 via-transparent to-[#0ea5e9]/8" />
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
          <h2 className={`text-4xl sm:text-5xl mb-4 ${
            theme === 'dark' ? 'text-white' : 'text-gray-900'
          }`}>
            Trusted by Industry Leaders
          </h2>
          <p className={`text-xl max-w-2xl mx-auto ${
            theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
          }`}>
            See how businesses are transforming safety and operations with VIGIL
          </p>
        </motion.div>

        {/* Testimonials Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {testimonials.map((testimonial, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className={`group relative p-8 rounded-2xl transition-all ${
                theme === 'dark'
                  ? 'bg-gradient-to-br from-[#06b6d4]/5 to-transparent border border-[#06b6d4]/20 hover:border-[#06b6d4]/50 hover:shadow-xl hover:shadow-[#06b6d4]/10'
                  : 'bg-gradient-to-br from-gray-50 to-white border border-gray-200 hover:border-[#06b6d4]/50 hover:shadow-xl hover:shadow-[#06b6d4]/10'
              }`}
            >
              {/* Quote Icon */}
              <div className="absolute top-8 right-8 opacity-10">
                <Quote className="w-16 h-16 text-[#06b6d4]" />
              </div>

              {/* Icon */}
              <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-[#06b6d4] to-[#0ea5e9] flex items-center justify-center mb-6">
                <testimonial.icon className="w-7 h-7 text-white" />
              </div>

              {/* Quote */}
              <p className={`mb-6 relative z-10 leading-relaxed ${
                theme === 'dark' ? 'text-gray-300' : 'text-gray-600'
              }`}>
                "{testimonial.quote}"
              </p>

              {/* Company Info */}
              <div className="mb-4">
                <div className={`mb-1 font-medium ${
                  theme === 'dark' ? 'text-white' : 'text-gray-900'
                }`}>
                  {testimonial.company}
                </div>
                <div className={`text-sm ${
                  theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
                }`}>
                  {testimonial.role} • {testimonial.industry}
                </div>
              </div>

              {/* Stat Badge */}
              <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full ${
                theme === 'dark'
                  ? 'bg-[#06b6d4]/10 border border-[#06b6d4]/30'
                  : 'bg-[#06b6d4]/10 border border-[#06b6d4]/30'
              }`}>
                <span className="text-sm bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9] bg-clip-text text-transparent font-medium">
                  {testimonial.stat}
                </span>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
