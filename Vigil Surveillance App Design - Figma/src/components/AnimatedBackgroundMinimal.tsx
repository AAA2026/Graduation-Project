import React, { useRef, useEffect } from "react";

interface AnimatedBackgroundMinimalProps {
  mode: "dark" | "light";
  children?: React.ReactNode;
}

// Minimal, modern animated background: floating blurred circles
const CIRCLE_COUNT = 12;
const COLORS_LIGHT = ["#bae6fd", "#a7f3d0", "#fcd34d", "#fca5a5"];
const COLORS_DARK = ["#0ea5e9", "#06b6d4", "#818cf8", "#f472b6"];

function randomBetween(a: number, b: number) {
  return a + Math.random() * (b - a);
}

const AnimatedBackgroundMinimal: React.FC<AnimatedBackgroundMinimalProps> = ({ mode, children }) => {
  const [circles, setCircles] = React.useState<any[]>([]);
  const animFrame = useRef<number | null>(null);

  useEffect(() => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    setCircles(Array.from({ length: CIRCLE_COUNT }, () => {
      const color = (mode === "dark" ? COLORS_DARK : COLORS_LIGHT)[Math.floor(Math.random() * 4)];
      return {
        x: randomBetween(0, w),
        y: randomBetween(0, h),
        r: randomBetween(60, 120),
        color,
        opacity: randomBetween(0.10, 0.22),
        dx: randomBetween(-0.18, 0.18),
        dy: randomBetween(-0.18, 0.18),
      };
    }));
    // eslint-disable-next-line
  }, [mode]);

  useEffect(() => {
    function animate() {
      setCircles(prev => prev.map(c => {
        let x = c.x + c.dx;
        let y = c.y + c.dy;
        const w = window.innerWidth;
        const h = window.innerHeight;
        // Wrap
        if (x < -c.r) x = w + c.r;
        if (x > w + c.r) x = -c.r;
        if (y < -c.r) y = h + c.r;
        if (y > h + c.r) y = -c.r;
        return { ...c, x, y };
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
    <div style={{ position: "fixed", inset: 0, zIndex: 0, overflow: "hidden" }}>
      <svg width="100vw" height="100vh" style={{ position: "absolute", inset: 0, width: "100vw", height: "100vh", pointerEvents: "none" }}>
        {circles.map((c, i) => (
          <circle
            key={i}
            cx={c.x}
            cy={c.y}
            r={c.r}
            fill={c.color}
            opacity={c.opacity}
            style={{ filter: "blur(32px)" }}
          />
        ))}
      </svg>
      <div style={{ position: "relative", zIndex: 1 }}>{children}</div>
    </div>
  );
};

export { AnimatedBackgroundMinimal };