/**
 * VIGIL LOGO - Standalone Component (Theme-Aware)
 *
 * A self-contained logo component with built-in light/dark theme support.
 * No external dependencies - perfect for easy integration into any React project.
 *
 * PROFESSIONAL BRANDING:
 * The electric cyan eye icon (#06b6d4) remains consistent across both themes
 * for strong brand recognition. Only the text color adapts to the theme.
 *
 * Usage Examples:
 * ---------------
 * import { VigilLogo } from './components/VigilLogo';
 *
 * // Dark mode (default)
 * <VigilLogo />
 *
 * // Light mode
 * <VigilLogo theme="light" />
 *
 * // Custom size
 * <VigilLogo width={200} height={60} />
 *
 * // With custom className
 * <VigilLogo className="my-logo" theme="dark" />
 *
 * Brand Colors:
 * -------------
 * Eye Icon (Consistent):
 *   - Electric Cyan: #06b6d4
 *   - Background: #0a0a0a (dark charcoal)
 *   - Pupil: #000000 (black)
 *   - Highlight: #f8fafc (white)
 *
 * Text (Theme-Adaptive):
 *   - Dark Mode: #f8fafc (off-white)
 *   - Light Mode: #000000 (black)
 */

interface VigilLogoProps {
  width?: number;
  height?: number;
  className?: string;
  theme?: 'light' | 'dark';
}

export function VigilLogo({
  width = 240,
  height = 80,
  className = "",
  theme = 'dark'
}: VigilLogoProps) {
  const eyeColors = {
    primary: '#06b6d4',
    background: '#0a0a0a',
    pupil: '#000000',
    highlight: '#f8fafc',
  };

  const textColor = theme === 'dark' ? '#f8fafc' : '#000000';

  return (
    <svg
      width={width}
      height={height}
      viewBox="0 0 240 80"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="Vigil Logo"
    >
      {/* Eye Icon - CONSISTENT for both themes */}
      <g transform="translate(15, 20)">
        {/* Outer eye shape - almond/lens shape */}
        <path
          d="M 5 20 Q 5 8, 25 5 Q 45 8, 45 20 Q 45 32, 25 35 Q 5 32, 5 20 Z"
          fill={eyeColors.background}
          stroke={eyeColors.primary}
          strokeWidth="2"
        />
        
        {/* Upper eyelid accent */}
        <path
          d="M 8 20 Q 8 11, 25 8 Q 42 11, 42 20"
          stroke={eyeColors.primary}
          strokeWidth="1.5"
          fill="none"
          opacity="0.5"
        />
        
        {/* Iris outer ring */}
        <circle
          cx="25"
          cy="20"
          r="9"
          fill="url(#irisGradient)"
          stroke={eyeColors.primary}
          strokeWidth="1.5"
        />
        
        {/* Iris detail lines - radial pattern */}
        <g opacity="0.4">
          <line x1="25" y1="20" x2="25" y2="12" stroke={eyeColors.primary} strokeWidth="0.8"/>
          <line x1="25" y1="20" x2="31" y2="14" stroke={eyeColors.primary} strokeWidth="0.8"/>
          <line x1="25" y1="20" x2="33" y2="20" stroke={eyeColors.primary} strokeWidth="0.8"/>
          <line x1="25" y1="20" x2="31" y2="26" stroke={eyeColors.primary} strokeWidth="0.8"/>
          <line x1="25" y1="20" x2="25" y2="28" stroke={eyeColors.primary} strokeWidth="0.8"/>
          <line x1="25" y1="20" x2="19" y2="26" stroke={eyeColors.primary} strokeWidth="0.8"/>
          <line x1="25" y1="20" x2="17" y2="20" stroke={eyeColors.primary} strokeWidth="0.8"/>
          <line x1="25" y1="20" x2="19" y2="14" stroke={eyeColors.primary} strokeWidth="0.8"/>
        </g>
        
        {/* Pupil - darker center */}
        <circle
          cx="25"
          cy="20"
          r="5"
          fill={eyeColors.pupil}
        />
        
        {/* Pupil highlight - makes it look alive */}
        <circle
          cx="27"
          cy="18"
          r="2"
          fill={eyeColors.primary}
          opacity="0.9"
        />
        <circle
          cx="28"
          cy="17"
          r="1"
          fill={eyeColors.highlight}
        />
        
        {/* Scanning reticle overlay */}
        <g opacity="0.7">
          {/* Corner brackets */}
          <path d="M 10 12 L 10 10 L 12 10" stroke={eyeColors.highlight} strokeWidth="1.5" strokeLinecap="round"/>
          <path d="M 40 10 L 38 10 L 38 12" stroke={eyeColors.highlight} strokeWidth="1.5" strokeLinecap="round"/>
          <path d="M 10 28 L 10 30 L 12 30" stroke={eyeColors.highlight} strokeWidth="1.5" strokeLinecap="round"/>
          <path d="M 40 30 L 38 30 L 38 28" stroke={eyeColors.highlight} strokeWidth="1.5" strokeLinecap="round"/>
        </g>
        
        {/* Scan line effect */}
        <line
          x1="10"
          y1="20"
          x2="40"
          y2="20"
          stroke={eyeColors.primary}
          strokeWidth="0.5"
          opacity="0.3"
          strokeDasharray="2,2"
        />
      </g>
      
      {/* Gradients - consistent for both themes */}
      <defs>
        <radialGradient id="irisGradient" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor={eyeColors.primary} stopOpacity="0.3"/>
          <stop offset="50%" stopColor={eyeColors.primary} stopOpacity="0.5"/>
          <stop offset="100%" stopColor={eyeColors.primary} stopOpacity="0.7"/>
        </radialGradient>
      </defs>
      
      {/* Text: VIGIL - Adapts to theme */}
      <g transform="translate(75, 18)">
        {/* V - stylized with sharp angles */}
        <path
          d="M 0 0 L 9 40 L 16 40 L 25 0 L 18 0 L 12.5 30 L 7 0 Z"
          fill={textColor}
        />
        
        {/* I */}
        <rect x="35" y="0" width="7" height="40" fill={textColor}/>
        
        {/* G - proper G shape with open top-right */}
        {/* Top horizontal bar (shorter - leaves opening on right) */}
        <rect x="52" y="0" width="14" height="7" fill={textColor}/>
        {/* Left vertical bar (full height) */}
        <rect x="52" y="0" width="7" height="40" fill={textColor}/>
        {/* Bottom horizontal bar (full width) */}
        <rect x="52" y="33" width="20" height="7" fill={textColor}/>
        {/* Middle horizontal bar (G's inward bar) */}
        <rect x="62" y="18" width="10" height="7" fill={textColor}/>
        {/* Right vertical bar (bottom half only) */}
        <rect x="65" y="18" width="7" height="22" fill={textColor}/>
        
        {/* I */}
        <rect x="82" y="0" width="7" height="40" fill={textColor}/>
        
        {/* L */}
        <rect x="99" y="0" width="7" height="40" fill={textColor}/>
        <rect x="99" y="33" width="20" height="7" fill={textColor}/>
      </g>
      
      {/* Status indicator dot - animated pulse with electric cyan */}
      <circle cx="228" cy="48" r="3" fill={eyeColors.primary}>
        <animate
          attributeName="opacity"
          values="1;0.3;1"
          dur="2s"
          repeatCount="indefinite"
        />
      </circle>
    </svg>
  );
}

// Additional utility: Export as pure SVG string for non-React usage
export function getVigilLogoSVG(
  theme: 'light' | 'dark' = 'dark',
  width: number = 240,
  height: number = 80
): string {
  const colors = {
    dark: {
      primary: '#06b6d4',
      text: '#f8fafc',
      background: '#0a0a0a',
      highlight: '#f8fafc',
      scanLine: '#06b6d4'
    },
    light: {
      primary: '#3b82f6',
      text: '#000000',
      background: '#f8fafc',
      highlight: '#2563eb',
      scanLine: '#3b82f6'
    }
  };

  const c = colors[theme];

  return `<svg width="${width}" height="${height}" viewBox="0 0 240 80" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="Vigil Logo">
  <g transform="translate(15, 20)">
    <path d="M 5 20 Q 5 8, 25 5 Q 45 8, 45 20 Q 45 32, 25 35 Q 5 32, 5 20 Z" fill="${c.background}" stroke="${c.primary}" stroke-width="2"/>
    <path d="M 8 20 Q 8 11, 25 8 Q 42 11, 42 20" stroke="${c.primary}" stroke-width="1.5" fill="none" opacity="0.5"/>
    <circle cx="25" cy="20" r="9" fill="url(#irisGradient-${theme})" stroke="${c.primary}" stroke-width="1.5"/>
    <g opacity="0.4">
      <line x1="25" y1="20" x2="25" y2="12" stroke="${c.primary}" stroke-width="0.8"/>
      <line x1="25" y1="20" x2="31" y2="14" stroke="${c.primary}" stroke-width="0.8"/>
      <line x1="25" y1="20" x2="33" y2="20" stroke="${c.primary}" stroke-width="0.8"/>
      <line x1="25" y1="20" x2="31" y2="26" stroke="${c.primary}" stroke-width="0.8"/>
      <line x1="25" y1="20" x2="25" y2="28" stroke="${c.primary}" stroke-width="0.8"/>
      <line x1="25" y1="20" x2="19" y2="26" stroke="${c.primary}" stroke-width="0.8"/>
      <line x1="25" y1="20" x2="17" y2="20" stroke="${c.primary}" stroke-width="0.8"/>
      <line x1="25" y1="20" x2="19" y2="14" stroke="${c.primary}" stroke-width="0.8"/>
    </g>
    <circle cx="25" cy="20" r="5" fill="${theme === 'dark' ? '#000000' : '#0a0a0a'}"/>
    <circle cx="27" cy="18" r="2" fill="${c.primary}" opacity="0.9"/>
    <circle cx="28" cy="17" r="1" fill="${c.highlight}"/>
    <g opacity="0.7">
      <path d="M 10 12 L 10 10 L 12 10" stroke="${c.highlight}" stroke-width="1.5" stroke-linecap="round"/>
      <path d="M 40 10 L 38 10 L 38 12" stroke="${c.highlight}" stroke-width="1.5" stroke-linecap="round"/>
      <path d="M 10 28 L 10 30 L 12 30" stroke="${c.highlight}" stroke-width="1.5" stroke-linecap="round"/>
      <path d="M 40 30 L 38 30 L 38 28" stroke="${c.highlight}" stroke-width="1.5" stroke-linecap="round"/>
    </g>
    <line x1="10" y1="20" x2="40" y2="20" stroke="${c.scanLine}" stroke-width="0.5" opacity="0.3" stroke-dasharray="2,2"/>
  </g>
  <defs>
    <radialGradient id="irisGradient-${theme}" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="${c.primary}" stop-opacity="0.3"/>
      <stop offset="50%" stop-color="${c.primary}" stop-opacity="0.5"/>
      <stop offset="100%" stop-color="${c.primary}" stop-opacity="0.7"/>
    </radialGradient>
  </defs>
  <g transform="translate(75, 18)">
    <path d="M 0 0 L 9 40 L 16 40 L 25 0 L 18 0 L 12.5 30 L 7 0 Z" fill="${c.text}"/>
    <rect x="35" y="0" width="7" height="40" fill="${c.text}"/>
    <rect x="52" y="0" width="14" height="7" fill="${c.text}"/>
    <rect x="52" y="0" width="7" height="40" fill="${c.text}"/>
    <rect x="52" y="33" width="20" height="7" fill="${c.text}"/>
    <rect x="62" y="18" width="10" height="7" fill="${c.text}"/>
    <rect x="65" y="18" width="7" height="22" fill="${c.text}"/>
    <rect x="82" y="0" width="7" height="40" fill="${c.text}"/>
    <rect x="99" y="0" width="7" height="40" fill="${c.text}"/>
    <rect x="99" y="33" width="20" height="7" fill="${c.text}"/>
  </g>
  <circle cx="228" cy="48" r="3" fill="${c.primary}">
    <animate attributeName="opacity" values="1;0.3;1" dur="2s" repeatCount="indefinite"/>
  </circle>
</svg>`;
}
