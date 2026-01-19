# Vigil Theme System Guide

## Overview

The Vigil surveillance system uses a comprehensive dual-theme system with carefully chosen colors for optimal visibility and professional appearance in both light and dark modes.

---

## Color Philosophy

### Light Mode
- **Primary Accent:** Light Blue (#3b82f6) - Vibrant and visible on light backgrounds
- **Secondary Accent:** Cyan (#06b6d4) - For highlights and special elements
- **Text:** Dark slate colors for high contrast and readability
- **Backgrounds:** Off-white and pure white for cards

### Dark Mode
- **Primary Accent:** Electric Cyan (#06b6d4) - The signature Vigil color
- **Secondary Accent:** Amber (#f59e0b) - For warnings and highlights
- **Text:** Off-white and light grays for comfortable reading
- **Backgrounds:** Deep charcoal for reduced eye strain

---

## Complete Color Palette

### Light Mode Colors

| Usage | Color | Hex Code | Notes |
|-------|-------|----------|-------|
| Primary | Light Blue | `#3b82f6` | Main accent, buttons, active states |
| Primary Hover | Darker Blue | `#2563eb` | Hover states for primary elements |
| Secondary | Cyan | `#06b6d4` | Highlights, special accents |
| Background | Off-White | `#f8fafc` | Main page background |
| Card Background | White | `#ffffff` | Card and panel backgrounds |
| Card Hover | Light Gray | `#f1f5f9` | Card hover states |
| Text Primary | Dark Slate | `#0f172a` | Headings, primary text |
| Text Secondary | Gray | `#475569` | Secondary text, labels |
| Text Muted | Muted Gray | `#64748b` | Hints, disabled text |
| Border | Light Gray | `#e2e8f0` | Standard borders |
| Border Hover | Medium Gray | `#cbd5e1` | Hover state borders |
| Success | Green | `#10b981` | Success messages, positive states |
| Warning | Amber | `#f59e0b` | Warnings, caution |
| Danger | Red | `#ef4444` | Errors, critical alerts |
| Info | Blue | `#3b82f6` | Information, tips |

### Dark Mode Colors

| Usage | Color | Hex Code | Notes |
|-------|-------|----------|-------|
| Primary | Electric Cyan | `#06b6d4` | Main accent, signature color |
| Primary Hover | Darker Cyan | `#0891b2` | Hover states |
| Secondary | Amber | `#f59e0b` | Warnings, highlights |
| Background | Dark Charcoal | `#0a0e1a` | Main page background |
| Card Background | Lighter Charcoal | `#13182b` | Card and panel backgrounds |
| Card Hover | Slate | `#1e293b` | Card hover states |
| Text Primary | Off-White | `#f8fafc` | Headings, primary text |
| Text Secondary | Light Gray | `#cbd5e1` | Secondary text, labels |
| Text Muted | Muted Gray | `#94a3b8` | Hints, disabled text |
| Border | Dark Slate | `#1e293b` | Standard borders |
| Border Hover | Lighter Slate | `#334155` | Hover state borders |
| Success | Green | `#10b981` | Success messages |
| Warning | Amber | `#f59e0b` | Warnings |
| Danger | Red | `#ef4444` | Errors, critical alerts |
| Info | Cyan | `#06b6d4` | Information |

---

## Tailwind Class Utilities

The `vigil-theme.ts` file provides pre-built Tailwind class combinations:

```tsx
import { vigilClasses } from './components/vigil-theme';

// Example usage
<div className={vigilClasses.bg}>
  <div className={vigilClasses.card}>
    <h2 className={vigilClasses.textPrimary}>Title</h2>
    <p className={vigilClasses.textSecondary}>Description</p>
  </div>
</div>
```

### Available Class Utilities

```typescript
// Backgrounds
vigilClasses.bg               // Page background
vigilClasses.bgCard           // Card background
vigilClasses.bgCardHover      // Card hover state

// Text
vigilClasses.textPrimary      // Primary text (headings)
vigilClasses.textSecondary    // Secondary text (body)
vigilClasses.textMuted        // Muted text (hints)

// Borders
vigilClasses.border           // Standard border
vigilClasses.borderHover      // Hover state border

// Buttons
vigilClasses.btnPrimary       // Primary button
vigilClasses.btnPrimaryOutline // Outlined primary button
vigilClasses.btnSecondary     // Secondary button
vigilClasses.btnGhost         // Ghost/transparent button

// Status Badges
vigilClasses.badgeSuccess     // Success badge
vigilClasses.badgeWarning     // Warning badge
vigilClasses.badgeDanger      // Danger badge
vigilClasses.badgeInfo        // Info badge

// Cards
vigilClasses.card             // Standard card
vigilClasses.cardHover        // Card with hover effect

// Inputs
vigilClasses.input            // Form input styling
```

---

## Component Examples

### Card Component
```tsx
<Card className="bg-white dark:bg-[#13182b] border-gray-200 dark:border-gray-800">
  <CardHeader>
    <CardTitle className="text-gray-900 dark:text-white">
      Title
    </CardTitle>
    <CardDescription className="text-gray-600 dark:text-gray-400">
      Description
    </CardDescription>
  </CardHeader>
  <CardContent>
    {/* Content */}
  </CardContent>
</Card>
```

### Button Component
```tsx
// Primary button
<Button className="bg-blue-500 hover:bg-blue-600 dark:bg-cyan-500 dark:hover:bg-cyan-600 text-white">
  Primary Action
</Button>

// Ghost button
<Button 
  variant="ghost" 
  className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
>
  Secondary Action
</Button>
```

### Badge Component
```tsx
// Success badge
<Badge className="bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400">
  Active
</Badge>

// Warning badge
<Badge className="bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400">
  Pending
</Badge>
```

### Input Component
```tsx
<Input
  className="bg-gray-50 dark:bg-[#0a0e1a] border-gray-300 dark:border-gray-700 text-gray-900 dark:text-white placeholder:text-gray-400 dark:placeholder:text-gray-500"
  placeholder="Enter text..."
/>
```

---

## Charts & Data Visualization

### Chart Colors

Light Mode:
- Primary: `#3b82f6` (Blue)
- Secondary: `#06b6d4` (Cyan)
- Tertiary: `#8b5cf6` (Purple)
- Quaternary: `#10b981` (Green)
- Quinary: `#f59e0b` (Amber)

Dark Mode:
- Primary: `#06b6d4` (Cyan)
- Secondary: `#8b5cf6` (Purple)
- Tertiary: `#10b981` (Green)
- Quaternary: `#f59e0b` (Amber)
- Quinary: `#ef4444` (Red)

### Using with Recharts
```tsx
import { getChartColors } from './components/vigil-theme';
import { useTheme } from './components/ThemeProvider';

function MyChart() {
  const { theme } = useTheme();
  const colors = getChartColors(theme);
  
  return (
    <ResponsiveContainer>
      <BarChart data={data}>
        <Bar dataKey="value" fill={colors[0]} />
        <Bar dataKey="value2" fill={colors[1]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
```

---

## Logo Integration

The VigilLogo component automatically adapts to the current theme:

```tsx
import { VigilLogo } from './components/VigilLogo';
import { useTheme } from './components/ThemeProvider';

function Header() {
  const { theme } = useTheme();
  
  return (
    <header>
      <VigilLogo width={200} height={67} theme={theme} />
    </header>
  );
}
```

Logo colors change automatically:
- **Light Mode:** Blue (#3b82f6) with dark slate text (#0f172a)
- **Dark Mode:** Cyan (#06b6d4) with off-white text (#f8fafc)

---

## CSS Variables

The following CSS variables are defined in `globals.css`:

### Light Mode (`:root`)
```css
--vigil-primary: #3b82f6;
--vigil-primary-dark: #2563eb;
--vigil-secondary: #06b6d4;
--vigil-bg: #f8fafc;
--vigil-card-bg: #ffffff;
--vigil-text-primary: #0f172a;
--vigil-text-secondary: #475569;
--vigil-text-muted: #64748b;
--vigil-border: #e2e8f0;
--vigil-border-hover: #cbd5e1;
--vigil-success: #10b981;
--vigil-warning: #f59e0b;
--vigil-danger: #ef4444;
```

### Dark Mode (`.dark`)
```css
--vigil-primary: #06b6d4;
--vigil-primary-dark: #0891b2;
--vigil-secondary: #f59e0b;
--vigil-bg: #0a0e1a;
--vigil-card-bg: #13182b;
--vigil-text-primary: #f8fafc;
--vigil-text-secondary: #cbd5e1;
--vigil-text-muted: #94a3b8;
--vigil-border: #1e293b;
--vigil-border-hover: #334155;
--vigil-success: #10b981;
--vigil-warning: #f59e0b;
--vigil-danger: #ef4444;
```

---

## Best Practices

### ✅ DO:
- Use `vigilClasses` utilities for consistency
- Always provide both light and dark variants
- Test color contrast for accessibility (WCAG AA minimum)
- Use semantic color names (success, warning, danger)
- Maintain the blue/cyan distinction between themes

### ❌ DON'T:
- Hardcode colors without theme variants
- Use pure white (#ffffff) text on dark backgrounds (use #f8fafc)
- Use pure black (#000000) text on light backgrounds (use #0f172a)
- Mix light and dark mode colors
- Override theme colors without good reason

---

## Accessibility

All color combinations meet WCAG 2.1 Level AA standards:
- Text contrast ratio: minimum 4.5:1
- Large text contrast ratio: minimum 3:1
- UI component contrast: minimum 3:1

---

## Migration Guide

If updating existing components:

1. Replace hardcoded cyan colors in light mode with blue:
   ```tsx
   // Old
   className="bg-cyan-500"
   
   // New
   className="bg-blue-500 dark:bg-cyan-500"
   ```

2. Update text colors to use proper contrast:
   ```tsx
   // Old
   className="text-white"
   
   // New
   className="text-gray-900 dark:text-white"
   ```

3. Use the theme utilities:
   ```tsx
   import { vigilClasses } from './components/vigil-theme';
   
   className={vigilClasses.textPrimary}
   ```

---

## Resources

- [Tailwind CSS Documentation](https://tailwindcss.com)
- [WCAG Contrast Checker](https://webaim.org/resources/contrastchecker/)
- `vigil-theme.ts` - Color utilities
- `ThemeProvider.tsx` - Theme context
- `ThemeToggle.tsx` - Theme switcher component

---

**Last Updated:** November 6, 2025  
**Version:** 2.0 (Light mode redesign)
