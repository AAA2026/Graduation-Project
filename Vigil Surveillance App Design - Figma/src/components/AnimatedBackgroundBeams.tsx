import React, { useRef, useEffect } from "react";

interface AnimatedBackgroundBeamsProps {
  mode: "dark" | "light";
  children?: React.ReactNode;
}

// Responsive node count, optimized for performance (reduced counts to lower O(N^2) loop cost)
const getNodeCount = (mode: "dark" | "light") => {
  const w = window.innerWidth;
  if (mode === "light") {
    if (w > 1800) return 40;
    if (w > 1200) return 30;
    return 20;
  }
  if (w > 1800) return 35;
  if (w > 1200) return 28;
  return 22;
};
const BEAM_DISTANCE = 110;
const COLORS_DARK = ["#38bdf8", "#0ea5e9", "#818cf8"];
const COLORS_LIGHT = ["#1e40af", "#2563eb", "#0ea5e9", "#1e3a8a"];
const AnimatedBackgroundBeams: React.FC<AnimatedBackgroundBeamsProps> = ({ mode, children }) => {
  const [nodes, setNodes] = React.useState<any[]>([]);
  const [twinkle, setTwinkle] = React.useState<number[]>([]);
  const animFrame = useRef<number | null>(null);

  // Responsive node count
  useEffect(() => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const count = getNodeCount(mode);
    setNodes(Array.from({ length: count }, (_, idx) => {
      const color = (mode === "dark" ? COLORS_DARK : COLORS_LIGHT)[Math.floor(Math.random() * (mode === "dark" ? COLORS_DARK.length : COLORS_LIGHT.length))];
      const depth = 0.7 + 0.6 * (idx / count);
      const baseR = mode === "light" ? randomBetween(2.7, 5.2) : randomBetween(2.2, 5);
      const baseDx = mode === "light" ? randomBetween(-1.5, 1.5) : randomBetween(-0.8, 0.8);
      const baseDy = mode === "light" ? randomBetween(-1.5, 1.5) : randomBetween(-0.8, 0.8);
      const baseOpacity = mode === "light" ? randomBetween(0.72, 1.0) : randomBetween(0.38, 0.65);
      return {
        x: randomBetween(-0.07 * w, 1.07 * w),
        y: randomBetween(-0.07 * h, 1.07 * h),
        r: baseR * (1.1 - 0.3 * (idx / count)),
        dx: baseDx * (0.7 + 0.5 * (idx / count)),
        dy: baseDy * (0.7 + 0.5 * (idx / count)),
        color,
        opacity: baseOpacity * (0.8 + 0.2 * Math.random()),
        depth,
      };
    }));
    // eslint-disable-next-line
  }, [mode]);

  // Animate nodes
  useEffect(() => {
    function animate() {
      setNodes(prev => prev.map((n, idx) => {
        let x = n.x + n.dx * n.depth;
        let y = n.y + n.dy * n.depth;
        const w = window.innerWidth;
        const h = window.innerHeight;
        const minX = -0.07 * w, maxX = 1.07 * w;
        const minY = -0.07 * h, maxY = 1.07 * h;
        if (x < minX) x = maxX;
        if (x > maxX) x = minX;
        if (y < minY) y = maxY;
        if (y > maxY) y = minY;
        return { ...n, x, y };
      }));
      setTwinkle(() => {
        const count = Math.floor(Math.random() * 3) + 2;
        const arr = [];
        for (let i = 0; i < count; ++i) arr.push(Math.floor(Math.random() * nodes.length));
        return arr;
      });
      animFrame.current = requestAnimationFrame(animate);
    }
    animFrame.current = requestAnimationFrame(animate);
    return () => {
      if (animFrame.current !== null) {
        cancelAnimationFrame(animFrame.current);
      }
    };
  }, [nodes.length]);

  // Utility
  function randomBetween(a: number, b: number) {
    return a + Math.random() * (b - a);
  }

  // Background gradient
  const bgGradient = mode === "dark"
    ? "radial-gradient(ellipse at 60% 40%, #0a192f 60%, #07101a 100%)"
    : "radial-gradient(ellipse at 60% 40%, #eaf3fb 60%, #cfe0f7 100%)";

  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 0, overflow: "hidden", background: bgGradient, transition: 'background 0.5s' }}>
      {/* Soft animated gradient overlay for light mode */}

      {/* Animated overlays and SVGs for both modes */}
      <>
        {mode === "dark" ? (
          <svg width="100vw" height="100vh" style={{ position: "absolute", inset: 0, width: "100vw", height: "100vh", pointerEvents: "none" }}>
            <defs>
              <linearGradient id="beam-gradient" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.85" />
                <stop offset="100%" stopColor="#818cf8" stopOpacity="0.15" />
              </linearGradient>
              <radialGradient id="vignette" cx="50%" cy="50%" r="80%">
                <stop offset="80%" stopColor="white" stopOpacity="0" />
                <stop offset="100%" stopColor="#07101a" stopOpacity="0.38" />
              </radialGradient>
            </defs>
            <rect x="0" y="0" width="100vw" height="100vh" fill="url(#vignette)" style={{ pointerEvents: 'none' }} />
            {/* Beams */}
            {nodes.flatMap((n1, i) =>
              nodes.map((n2, j) => {
                if (i >= j) return null;
                const dx = n1.x - n2.x;
                const dy = n1.y - n2.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                const beamChance = 0.33;
                const beamDist = BEAM_DISTANCE;
                if (dist < beamDist && Math.random() > beamChance) {
                  const alpha = 0.19 + 0.32 * (1 - dist / beamDist);
                  const beamColor = `url(#beam-gradient)`;
                  return (
                    <line
                      key={"beam-" + i + "-" + j}
                      x1={n1.x}
                      y1={n1.y}
                      x2={n2.x}
                      y2={n2.y}
                      stroke={beamColor}
                      strokeWidth={1.1}
                      opacity={alpha}
                    />
                  );
                }
                return null;
              })
            )}
            {/* Nodes */}
            {nodes.map((n, i) => {
              const twinkleOpacity = twinkle.includes(i) ? Math.max(0.2, n.opacity * 0.45 + 0.55 * Math.random()) : n.opacity;
              const t = Date.now() / 1200 + i;
              const highlightAngle = Math.sin(t) * Math.PI;
              const highlightX = n.x + Math.cos(highlightAngle) * n.r * 0.35;
              const highlightY = n.y + Math.sin(highlightAngle) * n.r * 0.35;
              return (
                <g key={i}>
                  <circle
                    cx={n.x}
                    cy={n.y}
                    r={n.r}
                    fill={n.color}
                    opacity={twinkleOpacity}
                  />
                  <ellipse
                    cx={highlightX}
                    cy={highlightY}
                    rx={n.r * 0.18}
                    ry={n.r * 0.32}
                    fill="rgba(255,255,255,0.32)"
                    opacity={0.9}
                  />
                </g>
              );
            })}
          </svg>
        ) : (
          <>
            {/* Soft animated gradient overlay for light mode */}
            <div
              style={{
                position: 'absolute',
                inset: 0,
                zIndex: 0,
                pointerEvents: 'none',
                background: 'linear-gradient(120deg, #e0e7ff 0%, #f0fdfa 50%, #fef9c3 100%)',
                opacity: 0.55,
                animation: 'gradientMove 4s ease-in-out infinite',
                backgroundSize: '200% 200%',
                filter: 'blur(16px)'
              }}
            />
            <svg width="100vw" height="100vh" style={{ position: "absolute", inset: 0, width: "100vw", height: "100vh", pointerEvents: "none" }}>
              <defs>
                <radialGradient id="flare" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#fff" stopOpacity="0.95" />
                  <stop offset="60%" stopColor="#fff" stopOpacity="0.18" />
                  <stop offset="100%" stopColor="#fff" stopOpacity="0" />
                </radialGradient>
                <radialGradient id="vignetteLight" cx="50%" cy="50%" r="80%">
                  <stop offset="80%" stopColor="white" stopOpacity="0" />
                  <stop offset="100%" stopColor="#b6cbe0" stopOpacity="0.08" />
                </radialGradient>
              </defs>
              <rect x="0" y="0" width="100vw" height="100vh" fill="url(#vignetteLight)" style={{ pointerEvents: 'none' }} />
              {/* Particles */}
              {nodes.map((n, i) => {
                const sparkle = twinkle.includes(i);
                const sparkleOpacity = sparkle ? 0.7 + 0.3 * Math.abs(Math.sin(Date.now() / 400 + i)) : n.opacity;
                const showFlare = i % 18 === (Math.floor(Date.now() / 1800) % 18);
                return (
                  <g key={i}>
                    <circle
                      cx={n.x}
                      cy={n.y}
                      r={n.r}
                      fill={n.color}
                      opacity={sparkleOpacity}
                    />
                    {showFlare && (
                      <circle
                        cx={n.x}
                        cy={n.y}
                        r={n.r * 3.2}
                        fill="url(#flare)"
                        opacity={0.38}
                      />
                    )}
                  </g>
                );
              })}
            </svg>
          </>
        )}
        {/* Animated overlay: subtle moving noise/light (CSS only, not SVG) */}
        <div style={{
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
          zIndex: 0,
          background: 'repeating-linear-gradient(120deg, rgba(255,255,255,0.03) 0 2px, transparent 2px 20px)',
          mixBlendMode: 'soft-light',
          animation: 'moveOverlay 3s linear infinite',
          opacity: 0.7
        }} />
      </>
      <style>{`
        @keyframes moveOverlay {
          0% { background-position: 0 0; }
          100% { background-position: 120px 80px; }
        }
        @keyframes gradientMove {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
      `}</style>
      <div style={{ position: "relative", zIndex: 1 }}>{children}</div>
    </div>
  );
};

export { AnimatedBackgroundBeams };