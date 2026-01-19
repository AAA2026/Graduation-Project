import { useEffect, useRef } from 'react';
import { motion } from 'motion/react';

interface MockCameraFeedProps {
  cameraId: string;
  className?: string;
}

export function MockCameraFeed({ cameraId, className = '' }: MockCameraFeedProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let time = 0;

    const drawFrame = () => {
      time += 0.01;

      // Create gradient background
      const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
      gradient.addColorStop(0, '#0f172a');
      gradient.addColorStop(0.5, '#1e293b');
      gradient.addColorStop(1, '#0f172a');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Add grid pattern
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.1)';
      ctx.lineWidth = 1;
      const gridSize = 30;
      for (let x = 0; x < canvas.width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
      }
      for (let y = 0; y < canvas.height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
      }

      // Add moving scanline
      const scanlineY = (Math.sin(time) * 0.5 + 0.5) * canvas.height;
      const scanlineGradient = ctx.createLinearGradient(0, scanlineY - 50, 0, scanlineY + 50);
      scanlineGradient.addColorStop(0, 'rgba(6, 182, 212, 0)');
      scanlineGradient.addColorStop(0.5, 'rgba(6, 182, 212, 0.3)');
      scanlineGradient.addColorStop(1, 'rgba(6, 182, 212, 0)');
      ctx.fillStyle = scanlineGradient;
      ctx.fillRect(0, scanlineY - 50, canvas.width, 100);

      // Add animated circles (simulating motion detection zones)
      const numCircles = 3;
      for (let i = 0; i < numCircles; i++) {
        const angle = time + (i * Math.PI * 2) / numCircles;
        const x = canvas.width / 2 + Math.cos(angle) * 100;
        const y = canvas.height / 2 + Math.sin(angle) * 60;
        const radius = 20 + Math.sin(time * 2 + i) * 5;

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(6, 182, 212, ${0.3 + Math.sin(time * 3 + i) * 0.2})`;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Inner dot
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(6, 182, 212, 0.8)';
        ctx.fill();
      }

      // Add camera info text
      ctx.font = 'bold 14px monospace';
      ctx.fillStyle = 'rgba(6, 182, 212, 0.8)';
      ctx.fillText('LIVE FEED', 20, 30);
      
      ctx.font = '12px monospace';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
      ctx.fillText(cameraId, 20, 50);

      // Add timestamp
      const now = new Date();
      const timeStr = now.toLocaleTimeString('en-US', { 
        hour12: false, 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit' 
      });
      ctx.fillText(timeStr, canvas.width - 100, 30);

      // Add noise/static effect
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        if (Math.random() > 0.97) {
          const noise = Math.random() * 50;
          data[i] += noise;
          data[i + 1] += noise;
          data[i + 2] += noise;
        }
      }
      ctx.putImageData(imageData, 0, 0);

      animationFrameId = requestAnimationFrame(drawFrame);
    };

    drawFrame();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [cameraId]);

  return (
    <canvas
      ref={canvasRef}
      width={1280}
      height={720}
      className={`w-full h-full object-cover ${className}`}
    />
  );
}
