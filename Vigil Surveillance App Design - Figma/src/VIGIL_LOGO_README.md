# Vigil Logo - Complete Guide

## 📦 Available Logo Components

### 1. **VigilLogo** (`/components/VigilLogo.tsx`)
The main logo component used throughout the Vigil application.

```tsx
import { VigilLogo } from './components/VigilLogo';

// Dark mode (default)
<VigilLogo />

// Light mode
<VigilLogo theme="light" />

// Custom size
<VigilLogo width={200} height={67} />
```

### 2. **VigilLogoStandalone** (`/components/VigilLogoStandalone.tsx`)
A standalone version with additional utilities for exporting pure SVG strings.

```tsx
import { VigilLogo, getVigilLogoSVG } from './components/VigilLogoStandalone';

// Use as React component
<VigilLogo theme="dark" width={240} height={80} />

// Get pure SVG string for non-React usage
const svgString = getVigilLogoSVG('dark', 240, 80);
```

---

## 🎨 Theme Variations

### Dark Mode (Default)
**Best for:** Dark backgrounds, colorful gradients, default application theme

**Colors:**
- Primary Cyan: `#06b6d4` (Electric cyan - bright and vibrant)
- Text: `#f8fafc` (Off-white)
- Background: `#0a0a0a` (Dark charcoal)
- Highlights: `#f8fafc` (White accents)

**Use cases:**
- Main application header
- Dark themed websites
- Over gradient backgrounds
- Night mode interfaces

### Light Mode
**Best for:** Light backgrounds, print materials, professional documents

**Colors:**
- Primary Blue: `#3b82f6` (Light blue for excellent visibility)
- Text: `#0f172a` (Dark slate - highly readable)
- Background: `#f8fafc` (Off-white)
- Highlights: `#2563eb` (Darker blue accents)

**Use cases:**
- Light themed websites
- Print materials (business cards, letterhead)
- Documentation
- Day mode interfaces

---

## 📐 Size Reference

| Size Preset | Width × Height | Use Case |
|-------------|----------------|----------|
| Small | 180 × 60 | Mobile header, compact spaces |
| Default | 240 × 80 | Standard desktop header |
| Large | 300 × 100 | Hero sections, splash screens |

**Aspect Ratio:** 3:1 (always maintain this ratio)

**Custom sizes:**
```tsx
// Calculate height: width ÷ 3
<VigilLogo width={360} height={120} />
```

---

## 📥 Export Options

### React Component
```tsx
import { VigilLogo } from './components/VigilLogoStandalone';

function Header() {
  return <VigilLogo theme="dark" width={240} height={80} />;
}
```

### Pure SVG String
```tsx
import { getVigilLogoSVG } from './components/VigilLogoStandalone';

// Get SVG as string
const darkLogo = getVigilLogoSVG('dark', 240, 80);
const lightLogo = getVigilLogoSVG('light', 240, 80);

// Use in HTML
document.getElementById('logo').innerHTML = darkLogo;

// Use as data URI
const dataUri = `data:image/svg+xml,${encodeURIComponent(darkLogo)}`;
```

### Download as File
Use the demo app (if available) or copy the SVG code directly from the component files.

---

## 🎯 Design Specifications

### Typography
- **Style:** Custom geometric sans-serif
- **Weight:** Bold
- **Height:** 40px (matches icon height for perfect alignment)
- **Kerning:** Optimized spacing between letters

### Icon Details
- **Eye dimensions:** 50 × 40px
- **Pupil:** Centered with highlight for 3D effect
- **Scanning reticle:** Corner brackets with animated scan line
- **Status indicator:** Animated pulsing dot (right side)

### Animation
- **Status dot:** 2s pulse cycle (opacity: 1 → 0.3 → 1)
- **Frame rate:** Smooth CSS animation
- **Loop:** Infinite

---

## 💡 Usage Best Practices

### ✅ DO:
- Use dark theme on dark or colorful backgrounds
- Use light theme on light/white backgrounds
- Maintain the 3:1 aspect ratio
- Allow the status indicator to animate
- Provide adequate padding around the logo

### ❌ DON'T:
- Stretch or distort the logo
- Change the color scheme manually
- Remove the eye icon or text separately
- Use dark theme on light backgrounds (visibility issues)
- Crop the status indicator

---

## 🔧 Integration Examples

### Next.js
```tsx
import { VigilLogo } from '@/components/VigilLogoStandalone';

export default function Header() {
  return (
    <header className="bg-gray-900 p-4">
      <VigilLogo theme="dark" width={200} height={67} />
    </header>
  );
}
```

### React with Theme Context
```tsx
import { VigilLogo } from './components/VigilLogoStandalone';
import { useTheme } from './ThemeProvider';

export function Header() {
  const { theme } = useTheme();
  
  return (
    <header>
      <VigilLogo theme={theme} width={180} height={60} />
    </header>
  );
}
```

### Vanilla HTML
```html
<!DOCTYPE html>
<html>
<head>
  <title>Vigil</title>
</head>
<body style="background: #0a0a0a; padding: 20px;">
  <!-- Paste the SVG code from VigilLogoStandalone.tsx -->
  <svg width="240" height="80" viewBox="0 0 240 80" fill="none">
    <!-- SVG paths here -->
  </svg>
</body>
</html>
```

---

## 📄 File Structure

```
components/
├── VigilLogo.tsx              # Main logo component (used in app)
└── VigilLogoStandalone.tsx    # Standalone with export utilities
```

---

## 🎨 Color Palette Reference

### Dark Mode Palette
```css
--vigil-cyan-dark: #06b6d4;
--vigil-text-dark: #f8fafc;
--vigil-bg-dark: #0a0a0a;
```

### Light Mode Palette
```css
--vigil-blue-light: #3b82f6;
--vigil-text-light: #0f172a;
--vigil-bg-light: #f8fafc;
```

---

## 📝 License & Usage

This logo is part of the Vigil AI-Powered Smart Surveillance System.

**Components:**
- Custom eye icon with scanning reticle
- Custom geometric "VIGIL" wordmark
- Animated status indicator

**Created:** 2025  
**Design System:** Vigil Brand Identity

---

## 🔗 Quick Links

- [Main Logo Component](/components/VigilLogo.tsx)
- [Standalone Component](/components/VigilLogoStandalone.tsx)
- [Theme Provider](/components/ThemeProvider.tsx)
- [App Integration](/App.tsx)

---

## ✨ Features

- ✅ Full React/TypeScript support
- ✅ Light & dark theme variants
- ✅ Responsive sizing
- ✅ Animated status indicator
- ✅ Zero external dependencies (standalone version)
- ✅ Accessible (aria-label included)
- ✅ SVG export utility function
- ✅ Professional design system

---

**Need help?** Check the component comments or refer to this guide.
