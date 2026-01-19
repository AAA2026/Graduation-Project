import React, { useRef, useEffect } from "react";

interface AnimatedBackgroundShapesProps {
  mode: "dark" | "light";
  children?: React.ReactNode;
}

const SHAPE_COUNT = 18;
const COLORS_LIGHT = ["#bae6fd", "#a7f3d0", "#fcd34d", "#fca5a5", "#f3e8ff"];
const COLORS_DARK = ["#0ea5e9", "#06b6d4", "#818cf8", "#f472b6", "#facc15"];
const SHAPES = [
  // SVG path generators for different shapes
  (color: string, size: number) => <rect width={size} height={size} rx={size * 0.3} fill={color} />,
  (color: string, size: number) => <circle r={size / 2} cx={size / 2} cy={size / 2} fill={color} />,
  (color: string, size: number) => <polygon points={`${size/2},0 ${size},${size} 0,${size}`} fill={color} />, // triangle
  (color: string, size: number) => <ellipse cx={size/2} cy={size/2} rx={size/2} ry={size/3} fill={color} />,
  (color: string, size: number) => <polygon points={`${size/2},0 ${size},${size/3} ${size*0.8},${size} ${size*0.2},${size} 0,${size/3}`} fill={color} />, // pentagon
];

function randomBetween(a: number, b: number) {
  return a + Math.random() * (b - a);
}

const AnimatedBackgroundShapes: React.FC<AnimatedBackgroundShapesProps> = ({ mode, children }) => {
  const [shapes, setShapes] = React.useState<any[]>([]);
  const animFrame = useRef<number | null>(null);

  useEffect(() => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    setShapes(Array.from({ length: SHAPE_COUNT }, () => {
      const color = (mode === "dark" ? COLORS_DARK : COLORS_LIGHT)[Math.floor(Math.random() * 5)];
      const shape = Math.floor(Math.random() * SHAPES.length);
      const size = randomBetween(28, 54);
      return {
        x: randomBetween(0, w),
        y: randomBetween(0, h),
        size,
        color,
        shape,
        opacity: randomBetween(0.18, 0.32),
        speed: randomBetween(0.18, 0.38),
        angle: randomBetween(0, 2 * Math.PI),
        rotate: randomBetween(0, 360),
        rotateSpeed: randomBetween(-0.18, 0.18),
      };
    }));
    // eslint-disable-next-line
  }, [mode]);

  useEffect(() => {
    function animate() {
      setShapes(prev => prev.map(s => {
        let x = s.x + Math.cos(s.angle) * s.speed;
        let y = s.y + Math.sin(s.angle) * s.speed;
        let rotate = s.rotate + s.rotateSpeed;
        const w = window.innerWidth;
        const h = window.innerHeight;
        // Wrap
        if (x < -60) x = w + 30;
        if (x > w + 60) x = -30;
        if (y < -60) y = h + 30;
        if (y > h + 60) y = -30;
        return { ...s, x, y, rotate };
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
        {shapes.map((s, i) => (
          <g
            key={i}
            transform={`translate(${s.x},${s.y}) rotate(${s.rotate} ${s.size/2} ${s.size/2})`}
            opacity={s.opacity}
          >
            {SHAPES[s.shape](s.color, s.size)}
          </g>
        ))}
      </svg>
      <div style={{ position: "relative", zIndex: 1 }}>{children}</div>
    </div>
  );
};

export { AnimatedBackgroundShapes };