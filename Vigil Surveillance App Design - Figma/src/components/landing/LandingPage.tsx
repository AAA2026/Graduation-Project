import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Navigation } from './Navigation';
import { Hero } from './Hero';
import { StatsShowcase } from './StatsShowcase';
import { Features } from './Features';
import { UseCases } from './UseCases';
import { Technology } from './Technology';
import { Team } from './Team';
import { SecurityCompliance } from './SecurityCompliance';
import { Testimonials } from './Testimonials';
import { CTASection } from './CTASection';
import { Footer } from './Footer';
import { DemoModal } from './DemoModal';
import { ParticleBackground } from './ParticleBackground';

export function LandingPage() {
    const [theme, setTheme] = useState<'light' | 'dark'>('dark');
    const [isDemoModalOpen, setIsDemoModalOpen] = useState(false);
    const navigate = useNavigate();

    useEffect(() => {
        document.documentElement.classList.remove('light', 'dark');
        document.documentElement.classList.add(theme);
    }, [theme]);

    const toggleTheme = () => {
        setTheme(prevTheme => prevTheme === 'dark' ? 'light' : 'dark');
    };

    const openDemoModal = () => {
        setIsDemoModalOpen(true);
    };

    const closeDemoModal = () => {
        setIsDemoModalOpen(false);
    };

    const handleLogin = () => {
        navigate('/app');
    };

    return (
        <div className={`min-h-screen transition-colors duration-300 ${theme === 'dark'
                ? 'bg-[#050505] text-[#f8fafc]'
                : 'bg-[#f8fafc] text-[#0f172a]'
            }`}>
            <ParticleBackground theme={theme} />
            <Navigation
                theme={theme}
                toggleTheme={toggleTheme}
                onLogin={handleLogin}
            />
            <Hero theme={theme} onBookDemo={openDemoModal} />
            <StatsShowcase theme={theme} />
            <UseCases theme={theme} />
            <Features theme={theme} />
            <Technology theme={theme} />
            <Team theme={theme} />
            <SecurityCompliance theme={theme} />
            <Testimonials theme={theme} />
            <CTASection theme={theme} onBookDemo={openDemoModal} />
            <Footer theme={theme} onBookDemo={openDemoModal} onLogin={handleLogin} />
            <DemoModal isOpen={isDemoModalOpen} onClose={closeDemoModal} theme={theme} />
        </div>
    );
}
