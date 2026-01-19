import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { X, User, Mail, Phone, Building, Loader2 } from 'lucide-react';

interface DemoModalProps {
  isOpen: boolean;
  onClose: () => void;
  theme: 'light' | 'dark';
}

export function DemoModal({ isOpen, onClose, theme }: DemoModalProps) {
  const [formData, setFormData] = useState({
    fullName: '',
    email: '',
    phone: '',
    organization: '',
    role: '',
    cameras: '',
    message: ''
  });
  const [charCount, setCharCount] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitMessage, setSubmitMessage] = useState('');
  const [messageType, setMessageType] = useState<'success' | 'error'>('success');

  useEffect(() => {
    setCharCount(formData.message.length);
  }, [formData.message]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setSubmitMessage('');

    try {
      const response = await fetch('/api/demo-bookings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
      });

      if (response.ok) {
        setMessageType('success');
        setSubmitMessage('✓ Demo request submitted successfully! We\'ll contact you soon.');
        setFormData({
          fullName: '',
          email: '',
          phone: '',
          organization: '',
          role: '',
          cameras: '',
          message: ''
        });
        setTimeout(() => {
          onClose();
          setSubmitMessage('');
        }, 2000);
      } else {
        setMessageType('error');
        setSubmitMessage('✗ Submission failed. Please try again.');
      }
    } catch (error) {
      setMessageType('error');
      setSubmitMessage('✗ Network error. Please check your connection.');
    } finally {
      setIsSubmitting(false);
    }
  };

  // Calculate form progress
  const filledFields = Object.values(formData).filter(val => val.trim() !== '').length;
  const totalFields = Object.keys(formData).length;
  const progress = (filledFields / totalFields) * 100;

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="absolute inset-0 bg-black/80 backdrop-blur-sm"
          />

          {/* Modal Content */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            className={`relative w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-2xl ${theme === 'dark'
              ? 'bg-[#0a0a0a] border-[#06b6d4]/30 shadow-[#06b6d4]/20'
              : 'bg-white border-gray-200 shadow-xl'
              } border shadow-2xl`}
          >
            {/* Close Button */}
            <button
              onClick={onClose}
              className={`absolute top-4 right-4 p-2 rounded-lg transition-colors z-10 ${theme === 'dark'
                ? 'hover:bg-[#06b6d4]/10 text-[#cbd5e1]'
                : 'hover:bg-gray-100 text-gray-500'
                }`}
            >
              <X className="w-6 h-6" />
            </button>

            {/* Header */}
            <div className={`p-8 border-b ${theme === 'dark' ? 'border-[#06b6d4]/20' : 'border-gray-100'
              }`}>
              <h2 className={`text-3xl mb-2 font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-900'
                }`}>Request a Demo</h2>
              <p className={theme === 'dark' ? 'text-[#cbd5e1]' : 'text-gray-600'}>
                Experience VIGIL's AI-powered surveillance capabilities
              </p>
            </div>

            {/* Progress Bar */}
            <div className={`h-1 ${theme === 'dark' ? 'bg-[#06b6d4]/20' : 'bg-gray-100'}`}>
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                className="h-full bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9]"
              />
            </div>

            {/* Form */}
            <form onSubmit={handleSubmit} className="p-8 space-y-6">
              {/* Full Name */}
              <div className="relative">
                <input
                  type="text"
                  name="fullName"
                  value={formData.fullName}
                  onChange={handleChange}
                  required
                  placeholder=" "
                  className={`peer w-full px-12 py-4 border rounded-lg focus:outline-none placeholder-transparent transition-all ${theme === 'dark'
                    ? 'bg-[#06b6d4]/5 border-[#06b6d4]/20 text-white focus:border-[#06b6d4]'
                    : 'bg-white border-gray-200 text-gray-900 focus:border-[#06b6d4] focus:ring-1 focus:ring-[#06b6d4]/20'
                    }`}
                />
                <label className={`absolute left-12 top-4 transition-all peer-placeholder-shown:top-4 peer-placeholder-shown:text-base peer-focus:top-1 peer-focus:text-xs peer-focus:text-[#06b6d4] peer-[:not(:placeholder-shown)]:top-1 peer-[:not(:placeholder-shown)]:text-xs ${theme === 'dark' ? 'text-[#94a3b8]' : 'text-gray-500'
                  }`}>
                  Full Name *
                </label>
                <User className={`absolute left-4 top-4 w-5 h-5 ${theme === 'dark' ? 'text-[#06b6d4]' : 'text-gray-400 peer-focus:text-[#06b6d4]'
                  }`} />
              </div>

              {/* Email */}
              <div className="relative">
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  required
                  placeholder=" "
                  className={`peer w-full px-12 py-4 border rounded-lg focus:outline-none placeholder-transparent transition-all ${theme === 'dark'
                    ? 'bg-[#06b6d4]/5 border-[#06b6d4]/20 text-white focus:border-[#06b6d4]'
                    : 'bg-white border-gray-200 text-gray-900 focus:border-[#06b6d4] focus:ring-1 focus:ring-[#06b6d4]/20'
                    }`}
                />
                <label className={`absolute left-12 top-4 transition-all peer-placeholder-shown:top-4 peer-placeholder-shown:text-base peer-focus:top-1 peer-focus:text-xs peer-focus:text-[#06b6d4] peer-[:not(:placeholder-shown)]:top-1 peer-[:not(:placeholder-shown)]:text-xs ${theme === 'dark' ? 'text-[#94a3b8]' : 'text-gray-500'
                  }`}>
                  Email Address *
                </label>
                <Mail className={`absolute left-4 top-4 w-5 h-5 ${theme === 'dark' ? 'text-[#06b6d4]' : 'text-gray-400 peer-focus:text-[#06b6d4]'
                  }`} />
              </div>

              {/* Phone */}
              <div className="relative">
                <input
                  type="tel"
                  name="phone"
                  value={formData.phone}
                  onChange={handleChange}
                  required
                  placeholder=" "
                  className={`peer w-full px-12 py-4 border rounded-lg focus:outline-none placeholder-transparent transition-all ${theme === 'dark'
                    ? 'bg-[#06b6d4]/5 border-[#06b6d4]/20 text-white focus:border-[#06b6d4]'
                    : 'bg-white border-gray-200 text-gray-900 focus:border-[#06b6d4] focus:ring-1 focus:ring-[#06b6d4]/20'
                    }`}
                />
                <label className={`absolute left-12 top-4 transition-all peer-placeholder-shown:top-4 peer-placeholder-shown:text-base peer-focus:top-1 peer-focus:text-xs peer-focus:text-[#06b6d4] peer-[:not(:placeholder-shown)]:top-1 peer-[:not(:placeholder-shown)]:text-xs ${theme === 'dark' ? 'text-[#94a3b8]' : 'text-gray-500'
                  }`}>
                  Phone Number *
                </label>
                <Phone className={`absolute left-4 top-4 w-5 h-5 ${theme === 'dark' ? 'text-[#06b6d4]' : 'text-gray-400 peer-focus:text-[#06b6d4]'
                  }`} />
              </div>

              {/* Organization */}
              <div className="relative">
                <input
                  type="text"
                  name="organization"
                  value={formData.organization}
                  onChange={handleChange}
                  required
                  placeholder=" "
                  className={`peer w-full px-12 py-4 border rounded-lg focus:outline-none placeholder-transparent transition-all ${theme === 'dark'
                    ? 'bg-[#06b6d4]/5 border-[#06b6d4]/20 text-white focus:border-[#06b6d4]'
                    : 'bg-white border-gray-200 text-gray-900 focus:border-[#06b6d4] focus:ring-1 focus:ring-[#06b6d4]/20'
                    }`}
                />
                <label className={`absolute left-12 top-4 transition-all peer-placeholder-shown:top-4 peer-placeholder-shown:text-base peer-focus:top-1 peer-focus:text-xs peer-focus:text-[#06b6d4] peer-[:not(:placeholder-shown)]:top-1 peer-[:not(:placeholder-shown)]:text-xs ${theme === 'dark' ? 'text-[#94a3b8]' : 'text-gray-500'
                  }`}>
                  Organization *
                </label>
                <Building className={`absolute left-4 top-4 w-5 h-5 ${theme === 'dark' ? 'text-[#06b6d4]' : 'text-gray-400 peer-focus:text-[#06b6d4]'
                  }`} />
              </div>

              {/* Role */}
              <div>
                <select
                  name="role"
                  value={formData.role}
                  onChange={handleChange}
                  className={`w-full px-4 py-4 border rounded-lg focus:outline-none ${theme === 'dark'
                    ? 'bg-[#06b6d4]/5 border-[#06b6d4]/20 text-white focus:border-[#06b6d4] [&>option]:bg-[#0a0a0a] [&>option]:text-white'
                    : 'bg-white border-gray-200 text-gray-900 focus:border-[#06b6d4] focus:ring-1 focus:ring-[#06b6d4]/20'
                    }`}
                >
                  <option value="">Select your role</option>
                  <option value="security">🛡️ Security Director</option>
                  <option value="it">💻 IT Manager</option>
                  <option value="operations">⚙️ Operations Manager</option>
                  <option value="executive">👔 Executive</option>
                  <option value="other">🔧 Other</option>
                </select>
              </div>

              {/* Cameras */}
              <div>
                <select
                  name="cameras"
                  value={formData.cameras}
                  onChange={handleChange}
                  className={`w-full px-4 py-4 border rounded-lg focus:outline-none ${theme === 'dark'
                    ? 'bg-[#06b6d4]/5 border-[#06b6d4]/20 text-white focus:border-[#06b6d4] [&>option]:bg-[#0a0a0a] [&>option]:text-white'
                    : 'bg-white border-gray-200 text-gray-900 focus:border-[#06b6d4] focus:ring-1 focus:ring-[#06b6d4]/20'
                    }`}
                >
                  <option value="">Select camera range</option>
                  <option value="1-10">📹 1-10 cameras</option>
                  <option value="11-50">📹 11-50 cameras</option>
                  <option value="51-200">📹 51-200 cameras</option>
                  <option value="200+">📹 200+ cameras</option>
                </select>
              </div>

              {/* Message */}
              <div className="relative">
                <textarea
                  name="message"
                  value={formData.message}
                  onChange={handleChange}
                  maxLength={500}
                  rows={4}
                  placeholder="Tell us about your security needs"
                  className={`w-full px-4 py-4 border rounded-lg focus:outline-none resize-none ${theme === 'dark'
                    ? 'bg-[#06b6d4]/5 border-[#06b6d4]/20 text-white placeholder-gray-500 focus:border-[#06b6d4]'
                    : 'bg-white border-gray-200 text-gray-900 placeholder-gray-400 focus:border-[#06b6d4] focus:ring-1 focus:ring-[#06b6d4]/20'
                    }`}
                />
                <div className={`text-xs text-right mt-1 ${theme === 'dark' ? 'text-[#94a3b8]' : 'text-gray-500'
                  }`}>
                  {charCount} / 500
                </div>
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isSubmitting}
                className="w-full py-4 bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9] rounded-lg text-white hover:shadow-lg hover:shadow-[#06b6d4]/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-medium"
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Submitting...</span>
                  </>
                ) : (
                  <span>🚀 Request Demo Access</span>
                )}
              </button>

              {/* Submit Message */}
              {submitMessage && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`p-4 rounded-lg text-center border ${messageType === 'success'
                    ? 'bg-green-500/10 text-green-600 border-green-500/20'
                    : 'bg-red-500/10 text-red-600 border-red-500/20'
                    }`}
                >
                  {submitMessage}
                </motion.div>
              )}
            </form>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
