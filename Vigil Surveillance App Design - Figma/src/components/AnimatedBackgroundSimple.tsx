
import React, { useRef, useEffect } from "react";
import "../styles/animated-background.css";

interface AnimatedBackgroundSimpleProps {
  mode: "dark" | "light";
  children?: React.ReactNode;
}

// Flying particle config
const PARTICLE_COUNT = 32;
const COLORS = ["#06b6d4", "#6366f1", "#60a5fa", "#fff"];
const GLYPHS = [
  // SVG paths for cyber glyphs (hex, lock, eye, shield, etc.)
  (color: string) => (<svg width="38" height="38" viewBox="0 0 38 38"><polygon points="19,3 35,11.5 35,26.5 19,35 3,26.5 3,11.5" fill="none" stroke={color} strokeWidth="2.5" opacity="0.8" /></svg>),
  (color: string) => (<svg width="38" height="38" viewBox="0 0 38 38"><circle cx="19" cy="19" r="13" fill="none" stroke={color} strokeWidth="2.5" opacity="0.8" /><circle cx="19" cy="19" r="5" fill={color} opacity="0.5" /></svg>),
  (color: string) => (<svg width="38" height="38" viewBox="0 0 38 38"><rect x="8" y="14" width="22" height="12" rx="6" fill="none" stroke={color} strokeWidth="2.5" opacity="0.8" /><rect x="16" y="18" width="6" height="8" rx="2" fill={color} opacity="0.5" /></svg>),
  (color: string) => (<svg width="38" height="38" viewBox="0 0 38 38"><path d="M19 6 L32 14 L32 24 L19 32 L6 24 L6 14 Z" fill="none" stroke={color} strokeWidth="2.5" opacity="0.8" /><circle cx="19" cy="19" r="4" fill={color} opacity="0.5" /></svg>),
  (color: string) => (<svg width="38" height="38" viewBox="0 0 38 38"><ellipse cx="19" cy="19" rx="14" ry="8" fill="none" stroke={color} strokeWidth="2.5" opacity="0.8" /><ellipse cx="19" cy="19" rx="5" ry="3" fill={color} opacity="0.5" /></svg>),
];
const GLYPH_COUNT = 18;

function randomBetween(a: number, b: number) {
  return a + Math.random() * (b - a);
}

const AnimatedBackgroundSimple: React.FC<AnimatedBackgroundSimpleProps> = ({ mode, children }) => {
  const [glyphs, setGlyphs] = React.useState<any[]>([]);
  const animFrame = useRef<number | null>(null);

  // Initialize glyphs
  useEffect(() => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    setGlyphs(Array.from({ length: GLYPH_COUNT }, () => {
      const color = COLORS[Math.floor(Math.random() * COLORS.length)];
      const glyph = Math.floor(Math.random() * GLYPHS.length);
      return {
        x: randomBetween(0, w),
        y: randomBetween(0, h),
        size: randomBetween(32, 54),
        color,
        glyph,
        opacity: randomBetween(0.18, 0.38),
        speed: randomBetween(0.12, 0.32),
        angle: randomBetween(0, 2 * Math.PI),
        rotate: randomBetween(0, 360),
        rotateSpeed: randomBetween(-0.12, 0.12),
      };
    }));
    // eslint-disable-next-line
  }, []);

  // Animate glyphs
  useEffect(() => {
    function animate() {
      setGlyphs(prev => prev.map(g => {
        let x = g.x + Math.cos(g.angle) * g.speed;
        let y = g.y + Math.sin(g.angle) * g.speed;
        let rotate = g.rotate + g.rotateSpeed;
        const w = window.innerWidth;
        const h = window.innerHeight;
        // Wrap
        if (x < -60) x = w + 30;
        if (x > w + 60) x = -30;
        if (y < -60) y = h + 30;
        if (y > h + 60) y = -30;
        return { ...g, x, y, rotate };
      }));
      animFrame.current = requestAnimationFrame(animate);
    }
    animFrame.current = requestAnimationFrame(animate);
    return () => {
      if (animFrame.current !== null) {
        cancelAnimationFrame(animFrame.current);
      }
    };
  }, []);

  return (
    <div className={`animated-bg flying-bg ${mode}`} style={{ position: "absolute", left: 0, top: 0, width: "100vw", height: "100vh", overflow: "hidden" }}>
      {/* Floating cyber glyphs */}
      {glyphs.map((g, i) => (
        <div
          key={i}
          style={{
            position: "absolute",
            left: g.x,
            top: g.y,
            width: g.size,
            height: g.size,
            opacity: g.opacity,
            filter: `drop-shadow(0 0 12px ${g.color}99)`,
            transform: `rotate(${g.rotate}deg)`,
            transition: "filter 0.3s"
          }}
        >
          {GLYPHS[g.glyph](g.color)}
        </div>
      ))}
      {children}
    </div>
  );
};

export { AnimatedBackgroundSimple };
