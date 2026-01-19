import React, { useRef, useEffect } from "react";

interface AnimatedBackgroundGridProps {
  mode: "dark" | "light";
  children?: React.ReactNode;
}

// Modern cyber grid: animated glowing lines and subtle moving nodes
const GRID_COLOR_DARK = "#0ea5e9";
const GRID_COLOR_LIGHT = "#38bdf8";
const NODE_COLOR_DARK = "#38bdf8";
const NODE_COLOR_LIGHT = "#0ea5e9";
const NODE_COLOR_LIGHT_ALT = "#0284c7"; // Brighter blue for visibility
const GRID_SIZE = 64;
const NODE_COUNT = 18;

function randomBetween(a: number, b: number) {
  return a + Math.random() * (b - a);
}

const AnimatedBackgroundGrid: React.FC<AnimatedBackgroundGridProps> = ({ mode, children }) => {
  const [nodes, setNodes] = React.useState<any[]>([]);
  const animFrame = useRef<number | null>(null);

  useEffect(() => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    setNodes(Array.from({ length: NODE_COUNT }, () => {
      const isLight = mode === "light";
      return {
        x: randomBetween(0, w),
        y: randomBetween(0, h),
        r: randomBetween(4, isLight ? 10 : 7),
        dx: randomBetween(isLight ? -0.28 : -0.16, isLight ? 0.28 : 0.16),
        dy: randomBetween(isLight ? -0.28 : -0.16, isLight ? 0.28 : 0.16),
        opacity: randomBetween(isLight ? 0.32 : 0.18, isLight ? 0.55 : 0.32),
      };
    }));
    // eslint-disable-next-line
  }, [mode]);

  useEffect(() => {
    function animate() {
      setNodes(prev => prev.map(n => {
        let x = n.x + n.dx;
        let y = n.y + n.dy;
        const w = window.innerWidth;
        const h = window.innerHeight;
        // Wrap
        if (x < 0) x = w;
        if (x > w) x = 0;
        if (y < 0) y = h;
        if (y > h) y = 0;
        return { ...n, x, y };
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

  const gridColor = mode === "dark" ? GRID_COLOR_DARK : GRID_COLOR_LIGHT;
  const nodeColor = mode === "dark" ? NODE_COLOR_DARK : NODE_COLOR_LIGHT_ALT;
  const bg = mode === "dark" ? "#0a192f" : "#e0f2fe";

  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 0, overflow: "hidden", background: bg, transition: 'background 0.5s' }}>
      <svg width="100vw" height="100vh" style={{ position: "absolute", inset: 0, width: "100vw", height: "100vh", pointerEvents: "none" }}>
        {/* Animated nodes only, no grid lines */}
        {nodes.map((n, i) => (
          <circle
            key={i}
            cx={n.x}
            cy={n.y}
            r={n.r}
            fill={nodeColor}
            opacity={n.opacity}
            style={{ filter: "blur(2px) drop-shadow(0 0 8px "+nodeColor+")" }}
          />
        ))}
      </svg>
      <div style={{ position: "relative", zIndex: 1 }}>{children}</div>
    </div>
  );
};

export { AnimatedBackgroundGrid };