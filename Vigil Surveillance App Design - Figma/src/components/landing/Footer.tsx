import { motion } from 'motion/react';

interface FooterProps {
  theme: 'light' | 'dark';
  onBookDemo: () => void;
  onLogin: () => void;
}

export function Footer({ theme, onBookDemo, onLogin }: FooterProps) {
  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <footer className={`relative py-16 px-4 sm:px-6 lg:px-8 border-t ${
      theme === 'dark' ? 'bg-[#0a0a0a] border-[#06b6d4]/20' : 'bg-white border-[#06b6d4]/30'
    }`}>
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12 mb-12">
          {/* Brand */}
          <div className="lg:col-span-1">
            <div className="text-2xl mb-4 bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9] bg-clip-text text-transparent font-bold">
              VIGIL
            </div>
            <p className={`mb-6 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
              AI-powered surveillance for safer, smarter operations.
            </p>
            <p className={`text-sm ${theme === 'dark' ? 'text-gray-500' : 'text-gray-500'}`}>
              Enterprise-grade AI surveillance<br />
              Trusted by businesses worldwide<br />
              © 2025 VIGIL. All rights reserved.
            </p>
          </div>

          {/* Product Links */}
          <div>
            <h4 className={`font-semibold mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Product</h4>
            <ul className="space-y-2">
              <li>
                <button 
                  onClick={() => scrollToSection('features')}
                  className={`transition-colors ${
                    theme === 'dark' 
                      ? 'text-[#cbd5e1] hover:text-[#06b6d4]' 
                      : 'text-gray-600 hover:text-[#06b6d4]'
                  }`}
                >
                  Features
                </button>
              </li>
              <li>
                <button 
                  onClick={() => scrollToSection('use-cases')}
                  className={`transition-colors ${
                    theme === 'dark' 
                      ? 'text-[#cbd5e1] hover:text-[#06b6d4]' 
                      : 'text-gray-600 hover:text-[#06b6d4]'
                  }`}
                >
                  Use Cases
                </button>
              </li>
              <li>
                <button 
                  onClick={() => scrollToSection('technology')}
                  className={`transition-colors ${
                    theme === 'dark' 
                      ? 'text-[#cbd5e1] hover:text-[#06b6d4]' 
                      : 'text-gray-600 hover:text-[#06b6d4]'
                  }`}
                >
                  Technology
                </button>
              </li>
              <li>
                <button 
                  onClick={() => scrollToSection('team')}
                  className={`transition-colors ${
                    theme === 'dark' 
                      ? 'text-[#cbd5e1] hover:text-[#06b6d4]' 
                      : 'text-gray-600 hover:text-[#06b6d4]'
                  }`}
                >
                  Team
                </button>
              </li>
            </ul>
          </div>

          {/* Documentation Links */}
          <div>
            <h4 className={`font-semibold mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Documentation</h4>
            <ul className="space-y-2">
              <li><a href="#" className={`transition-colors ${
                theme === 'dark' 
                  ? 'text-[#cbd5e1] hover:text-[#06b6d4]' 
                  : 'text-gray-600 hover:text-[#06b6d4]'
              }`}>Technical Specs</a></li>
              <li><a href="#" className={`transition-colors ${
                theme === 'dark' 
                  ? 'text-[#cbd5e1] hover:text-[#06b6d4]' 
                  : 'text-gray-600 hover:text-[#06b6d4]'
              }`}>API Reference</a></li>
              <li><a href="#" className={`transition-colors ${
                theme === 'dark' 
                  ? 'text-[#cbd5e1] hover:text-[#06b6d4]' 
                  : 'text-gray-600 hover:text-[#06b6d4]'
              }`}>Dataset Info</a></li>
              <li><a href="#" className={`transition-colors ${
                theme === 'dark' 
                  ? 'text-[#cbd5e1] hover:text-[#06b6d4]' 
                  : 'text-gray-600 hover:text-[#06b6d4]'
              }`}>Model Architecture</a></li>
            </ul>
          </div>

          {/* Project Links */}
          <div>
            <h4 className={`font-semibold mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Project</h4>
            <ul className="space-y-2">
              <li><a href="#" className={`transition-colors ${
                theme === 'dark' 
                  ? 'text-[#cbd5e1] hover:text-[#06b6d4]' 
                  : 'text-gray-600 hover:text-[#06b6d4]'
              }`}>About</a></li>
              <li>
                <button 
                  onClick={onBookDemo}
                  className={`transition-colors ${
                    theme === 'dark' 
                      ? 'text-[#cbd5e1] hover:text-[#06b6d4]' 
                      : 'text-gray-600 hover:text-[#06b6d4]'
                  }`}
                >
                  Request Demo
                </button>
              </li>
              <li>
                <button 
                  onClick={onLogin}
                  className={`transition-colors ${
                    theme === 'dark' 
                      ? 'text-[#cbd5e1] hover:text-[#06b6d4]' 
                      : 'text-gray-600 hover:text-[#06b6d4]'
                  }`}
                >
                  System Login
                </button>
              </li>
              <li><a href="#" className={`transition-colors ${
                theme === 'dark' 
                  ? 'text-[#cbd5e1] hover:text-[#06b6d4]' 
                  : 'text-gray-600 hover:text-[#06b6d4]'
              }`}>Contact</a></li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="pt-8 border-t border-[#06b6d4]/20 flex flex-col sm:flex-row justify-between items-center gap-4">
          <p className={`text-sm ${theme === 'dark' ? 'text-[#94a3b8]' : 'text-gray-500'}`}>
            © 2025 VIGIL Systems. All rights reserved.
          </p>
          <div className={`flex items-center gap-4 text-sm ${theme === 'dark' ? 'text-[#94a3b8]' : 'text-gray-500'}`}>
            <a href="#" className="hover:text-[#06b6d4] transition-colors">Privacy Policy</a>
            <span>•</span>
            <a href="#" className="hover:text-[#06b6d4] transition-colors">Terms of Service</a>
            <span>•</span>
            <a href="#" className="hover:text-[#06b6d4] transition-colors">Security</a>
          </div>
        </div>
      </div>
    </footer>
  );
}
