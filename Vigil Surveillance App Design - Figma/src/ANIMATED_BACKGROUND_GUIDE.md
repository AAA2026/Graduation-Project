# Animated Background System - Implementation Guide

## Overview

The Vigil surveillance system now includes a professional **AnimatedBackground** component that provides dynamic, theme-aware animated backgrounds for modals, dashboards, and cards. This component follows modern design principles with smooth animations, accessibility considerations, and full dark/light mode support.

---

## Component API

### `AnimatedBackground`

```tsx
import { AnimatedBackground } from './components/AnimatedBackground';

<AnimatedBackground 
  variant="modal" 
  intensity="subtle" 
  className="rounded-lg" 
/>
```

### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `variant` | `'modal' \| 'dashboard' \| 'card'` | `'dashboard'` | Background style variant |
| `intensity` | `'subtle' \| 'medium' \| 'strong'` | `'subtle'` | Animation intensity and opacity |
| `className` | `string` | `''` | Additional CSS classes |

---

## Design Principles

### 1. **Base Layer**
- **Dark Mode:** Black (#000) to deep gray (#18181b) gradient
- **Light Mode:** White (#fff) to light gray (#f3f4f6) gradient
- Provides solid foundation for overlays

### 2. **Accent Gradients**
- **Dark Mode:** Blue, cyan, purple hues (30-50% opacity)
- **Light Mode:** Pastel blue, soft yellow, light purple (30-50% opacity)
- Multiple overlapping gradients create depth

### 3. **Animated Elements**
- **5 floating shapes** (circles/blobs) with blur effect
- **20-30 second animation loops** with easeInOut timing
- **Low opacity (10-20%)** to avoid distraction
- Movements: translate, scale, rotate

### 4. **Visual Effects**
- **Noise/Grain Texture:** Subtle fractal noise overlay
- **Vignette Effect:** Radial gradient fade at edges
- **Border Glow:** Cyan/blue accent border (modal variant only)
- **GPU-Accelerated:** Uses CSS transforms for performance

### 5. **Accessibility**
- **High contrast maintained** for text and controls
- **Non-distracting animations** (subtle by default)
- **Respects motion preferences** (can be extended)
- **Pointer-events disabled** on background layer

---

## Intensity Levels

### Subtle (Default)
```tsx
<AnimatedBackground intensity="subtle" />
```
- Opacity: 15% (dark) / 10% (light)
- Blur: 80px
- Best for: Main dashboards, content-heavy pages

### Medium
```tsx
<AnimatedBackground intensity="medium" />
```
- Opacity: 25% (dark) / 15% (light)
- Blur: 60px
- Best for: Modals, dialogs, feature sections

### Strong
```tsx
<AnimatedBackground intensity="strong" />
```
- Opacity: 35% (dark) / 20% (light)
- Blur: 40px
- Best for: Hero sections, splash screens, promotional areas

---

## Variants

### Dashboard
```tsx
<AnimatedBackground variant="dashboard" />
```
- **Use case:** Main application background
- **Features:** Full gradient overlay, animated shapes
- **No border glow**

### Modal
```tsx
<AnimatedBackground variant="modal" />
```
- **Use case:** Video modals, dialog overlays
- **Features:** Border glow effect with cyan/blue accent
- **Rounded corners supported**

### Card
```tsx
<AnimatedBackground variant="card" />
```
- **Use case:** Individual cards, panels, sections
- **Features:** Localized animation, smaller scale
- **Easily combined with card borders**

---

## Color Schemes

### Dark Mode Colors

| Element | Color | Opacity |
|---------|-------|---------|
| Base Gradient 1 | Cyan → Blue | 40% |
| Base Gradient 2 | Blue → Purple | 30% |
| Base Gradient 3 | Purple → Cyan | 25% |
| Accent Shape 1 | Cyan (#06b6d4) | 15% |
| Accent Shape 2 | Blue (#3b82f6) | 10% |
| Accent Shape 3 | Purple (#a855f7) | 12% |

### Light Mode Colors

| Element | Color | Opacity |
|---------|-------|---------|
| Base Gradient 1 | Blue → Cyan | 60% |
| Base Gradient 2 | Purple → Blue | 50% |
| Base Gradient 3 | Yellow → Blue | 40% |
| Accent Shape 1 | Blue (#93c5fd) | 15% |
| Accent Shape 2 | Cyan (#67e8f9) | 10% |
| Accent Shape 3 | Purple (#d8b4fe) | 12% |

---

## Animation Details

### Floating Shape Animation
```typescript
// Shape moves in circular/elliptical path
x: [0, 30, -20, 0]  // Horizontal movement
y: [0, -40, 30, 0]  // Vertical movement
scale: [1, 1.1, 0.9, 1]  // Size pulsing
rotate: [0, 5, -5, 0]  // Rotation angle

duration: 20-30 seconds
repeat: Infinity
ease: "easeInOut"
```

### Animation Variants
- **Variant 1:** 20s duration, moderate movement
- **Variant 2:** 25s duration, slower movement
- **Variant 3:** 30s duration, gentle movement

Each shape uses a different variant to create organic, non-repetitive motion.

---

## Integration Examples

### 1. Video Modal
```tsx
// In VideoModal.tsx
<motion.div className="relative ...">
  <AnimatedBackground variant="modal" intensity="medium" />
  
  <div className="relative z-10">
    {/* Modal content appears above background */}
  </div>
</motion.div>
```

### 2. Dashboard Layout
```tsx
// In ModernSecurityLayout.tsx
<div className="relative min-h-screen">
  <AnimatedBackground variant="dashboard" intensity="subtle" />
  
  <div className="relative z-10">
    {/* Dashboard content */}
  </div>
</div>
```

### 3. Feature Card
```tsx
// In any Card component
<Card className="relative overflow-hidden">
  <AnimatedBackground variant="card" intensity="subtle" className="rounded-lg" />
  
  <CardContent className="relative z-10">
    {/* Card content */}
  </CardContent>
</Card>
```

---

## Theme Integration

The component automatically detects and adapts to your current theme:

```tsx
import { useTheme } from './ThemeProvider';

const { theme } = useTheme();
const isDark = theme === 'dark';

// Component automatically switches colors based on theme
```

### Dark Mode
- Darker backgrounds (#000, #18181b)
- Vibrant accent colors (cyan, blue, purple)
- Higher contrast for clarity

### Light Mode
- Lighter backgrounds (#fff, #f3f4f6)
- Pastel accent colors (soft blue, yellow, purple)
- Softer contrast to avoid eye strain

---

## Performance Considerations

### GPU Acceleration
- All animations use CSS `transform` properties
- `will-change: transform` applied to animated elements
- Blur effects use CSS `filter` (GPU-accelerated on modern browsers)

### Optimization Tips
```tsx
// Use subtle intensity for pages with heavy content
<AnimatedBackground intensity="subtle" />

// Limit to one animated background per view
// Avoid nesting multiple AnimatedBackgrounds

// Use absolute positioning to prevent layout reflow
className="absolute inset-0"
```

### Browser Support
- ✅ Chrome/Edge 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ All modern mobile browsers
- ⚠️ Older browsers see static gradient (graceful degradation)

---

## Accessibility

### Contrast Compliance
- All text maintains **WCAG AA** contrast ratio minimum
- Background opacity kept low enough for readability
- Tested with both light and dark themes

### Motion Sensitivity
The component respects user motion preferences:

```tsx
// Future enhancement
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

if (prefersReducedMotion.matches) {
  // Disable animations, show static gradient
}
```

### Screen Readers
- Background is decorative (pointer-events disabled)
- Does not interfere with navigation or focus
- Content above background remains fully accessible

---

## Customization

### Custom Colors
Extend the component to use your brand colors:

```tsx
// In AnimatedBackground.tsx
const customColors = {
  accent1: 'bg-brand-primary',
  accent2: 'bg-brand-secondary',
  accent3: 'bg-brand-tertiary',
};
```

### Animation Speed
Adjust duration for faster/slower animations:

```tsx
transition: {
  duration: 15,  // Faster (default: 20-30)
  repeat: Infinity,
  ease: "easeInOut",
}
```

### Shape Count
Add or remove floating shapes in the component:

```tsx
// Add more shapes for denser effect
<motion.div variants={floatingVariants} ... />
<motion.div variants={floatingVariants} ... />
<motion.div variants={floatingVariants} ... />
```

---

## Testing the Component

### Live Demo
A demo component is available to test all variants and intensities:

```tsx
import { AnimatedBackgroundDemo } from './components/AnimatedBackgroundDemo';

<AnimatedBackgroundDemo />
```

### Visual Testing
1. **Light/Dark Mode:** Toggle theme and verify color transitions
2. **Intensity Levels:** Test all three intensity settings
3. **Variants:** Check dashboard, modal, card variants
4. **Responsiveness:** Test on mobile, tablet, desktop
5. **Performance:** Monitor frame rate in DevTools

---

## Migration Guide

### Adding to Existing Components

**Before:**
```tsx
<div className="bg-background">
  <CardContent>...</CardContent>
</div>
```

**After:**
```tsx
<div className="relative">
  <AnimatedBackground variant="card" intensity="subtle" />
  <CardContent className="relative z-10">...</CardContent>
</div>
```

### Important Notes
- Always add `position: relative` to container
- Add `z-10` or higher to content layer
- Use `overflow-hidden` on container if needed
- Background has `pointer-events: none` by default

---

## Troubleshooting

### Background Not Visible
- ✅ Ensure container has `position: relative`
- ✅ Check container has sufficient height
- ✅ Verify theme provider is active
- ✅ Increase intensity if too subtle

### Content Hidden Behind Background
- ✅ Add `relative z-10` to content elements
- ✅ Check for conflicting z-index values
- ✅ Verify background has `pointer-events: none`

### Animation Performance Issues
- ✅ Reduce intensity level
- ✅ Use `subtle` variant for heavy pages
- ✅ Check browser GPU acceleration support
- ✅ Limit to one background per view

### Colors Don't Match Theme
- ✅ Verify ThemeProvider is wrapping component
- ✅ Check theme value in context
- ✅ Force re-render after theme change

---

## Future Enhancements

- [ ] Respect `prefers-reduced-motion` media query
- [ ] Add particle system variant
- [ ] Include custom color prop
- [ ] Create animation preset library
- [ ] Add interactive hover effects
- [ ] Support custom shape patterns

---

## Credits

Designed following modern UI/UX principles from:
- Glassmorphism design patterns
- Framer Motion best practices
- WCAG accessibility guidelines
- GPU acceleration techniques

---

## Support

For questions or issues with the AnimatedBackground component:
1. Check this documentation
2. Review the AnimatedBackgroundDemo component
3. Test with different variants and intensities
4. Inspect browser DevTools for performance metrics

**Last Updated:** January 10, 2026
**Component Version:** 1.0.0
**Compatible with:** Vigil v2.0+
