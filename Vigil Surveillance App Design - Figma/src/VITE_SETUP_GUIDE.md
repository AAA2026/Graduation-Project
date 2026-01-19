# Vigil - Vite Setup Guide

This guide will help you set up the Vigil surveillance system in your local Vite environment.

---

## 📋 Prerequisites

- **Node.js**: v18.0.0 or higher
- **npm**: v9.0.0 or higher (or yarn/pnpm)
- **Git**: For cloning (optional)

---

## 🚀 Quick Start

### Step 1: Create Vite Project Structure

```bash
# Create a new directory
mkdir vigil-surveillance
cd vigil-surveillance

# Initialize npm (or use existing package.json from this project)
npm init -y
```

### Step 2: Copy Project Files

Copy all files from this Figma Make project to your local directory with the following structure:

```
vigil-surveillance/
├── src/
│   ├── components/
│   │   ├── ui/              # All UI components
│   │   ├── dashboard/       # Dashboard components (if any)
│   │   ├── figma/           # Figma-specific components
│   │   ├── AIFeedbackSection.tsx
│   │   ├── AIModelManagement.tsx
│   │   ├── AnalyticsView.tsx
│   │   ├── Breadcrumb.tsx               # ✨ NEW - Navigation breadcrumbs
│   │   ├── CameraManagement.tsx
│   │   ├── DVRCameraGrid.tsx
│   │   ├── IncidentDetailModal.tsx
│   │   ├── IncidentFeed.tsx
│   │   ├── IncidentsView.tsx            # ✨ UPDATED - Export buttons added
│   │   ├── LiveActivityFeed.tsx
│   │   ├── LoginScreen.tsx
│   │   ├── MapView.tsx
│   │   ├── ModernSecurityLayout.tsx     # ✨ UPDATED - StatusWidget added
│   │   ├── NotificationCenter.tsx
│   │   ├── OfficerDashboard.tsx
│   │   ├── OfficerReports.tsx
│   │   ├── SecurityAuthorityApp.tsx
│   │   ├── StatusWidget.tsx             # ✨ NEW - System health monitor
│   │   ├── SystemHealthView.tsx
│   │   ├── ThemeProvider.tsx
│   │   ├── ThemeToggle.tsx
│   │   ├── UserManagement.tsx
│   │   ├── VigilLogo.tsx
│   │   ├── VigilLogoStandalone.tsx
│   │   └── vigil-theme.ts
│   ├── hooks/
│   │   └── useRealtimeIncidents.ts
│   ├── utils/
│   │   └── exportUtils.ts               # ✨ NEW - PDF/CSV export utilities
│   ├── styles/
│   │   └── globals.css
│   ├── App.tsx
│   └── main.tsx
├── public/                  # Static assets (if any)
├── index.html
├── package.json
├── tsconfig.json
├── tsconfig.node.json
├── vite.config.ts
└── postcss.config.js
```

### Step 3: Install Dependencies

```bash
npm install
```

This will install all dependencies listed in `package.json`:

**Core Dependencies:**
- `react` & `react-dom` - React framework
- `lucide-react` - Icon library
- `recharts` - Chart library for analytics
- `motion` (Framer Motion v11) - Animation library
- `sonner` - Toast notifications
- `class-variance-authority` - CSS variant utilities
- `clsx` - Classname utility
- `tailwind-merge` - Tailwind class merger

**Dev Dependencies:**
- `@vitejs/plugin-react` - Vite React plugin
- `tailwindcss@4.0.0` - Tailwind CSS v4
- `typescript` - TypeScript support
- `autoprefixer` & `postcss` - CSS processing

### Step 4: Start Development Server

```bash
npm run dev
```

The app should now be running at `http://localhost:3000`

---

## 📁 File Structure Explained

### `/src/App.tsx`
- Main application component
- Handles authentication state
- Routes between Login and Main Dashboard

### `/src/main.tsx`
- Entry point for the React app
- Renders the App component into the DOM

### `/src/components/`

#### Core Layout Components:
- **ModernSecurityLayout.tsx** - Main layout with horizontal navigation
- **LoginScreen.tsx** - Authentication screen
- **ThemeProvider.tsx** - Dark/Light mode context

#### Dashboard Views:
- **DVRCameraGrid.tsx** - Live camera monitor (2×2, 2×3, 3×3, 4×4 grids)
- **IncidentsView.tsx** - Active incidents with filters
- **AnalyticsView.tsx** - Performance metrics and charts
- **CameraManagement.tsx** - Camera CRUD operations
- **UserManagement.tsx** - User CRUD operations
- **AIModelManagement.tsx** - AI model versioning
- **SystemHealthView.tsx** - System monitoring
- **AIFeedbackSection.tsx** - AI feedback review
- **OfficerReports.tsx** - Report generation
- **MapView.tsx** - Geographic camera/incident view
- **NotificationCenter.tsx** - Notification management

#### Supporting Components:
- **IncidentDetailModal.tsx** - Detailed incident modal (98vh × 98vw)
- **IncidentFeed.tsx** - Live incident feed
- **LiveActivityFeed.tsx** - Activity timeline
- **VigilLogo.tsx** - Custom Vigil logo component

### `/src/components/ui/`
All shadcn/ui-style components:
- `button.tsx`, `card.tsx`, `badge.tsx`, `input.tsx`, etc.
- Pre-styled, reusable UI primitives

### `/src/hooks/`
- **useRealtimeIncidents.ts** - Real-time incident simulation with toast notifications

### `/src/utils/`
- **exportUtils.ts** - PDF/CSV export utilities

### `/src/styles/globals.css`
- Tailwind CSS v4 imports
- CSS custom properties (variables)
- Theme definitions (light/dark)
- Vigil brand colors
- Custom animations
- Logo optimizations

---

## 🎨 Tailwind CSS v4 Configuration

This project uses **Tailwind CSS v4** which has a different setup than v3:

### Key Differences:

1. **No `tailwind.config.js` needed** - Configuration is done in CSS using `@theme`
2. **Import in CSS**: Use `@import "tailwindcss"` instead of base/components/utilities
3. **PostCSS Plugin**: Use `@tailwindcss/postcss` instead of `tailwindcss`

### CSS Variables (in globals.css):

```css
:root {
  --background: oklch(0.985 0.01 95);
  --foreground: oklch(0.20 0.015 250);
  --vigil-primary: #3b82f6;
  --vigil-secondary: #06b6d4;
  /* ... more variables */
}

.dark {
  --background: oklch(0.145 0 0);
  --vigil-primary: #06b6d4;
  /* ... dark mode overrides */
}
```

---

## 🔧 Configuration Files

### `vite.config.ts`
```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@/components': path.resolve(__dirname, './src/components'),
      '@/hooks': path.resolve(__dirname, './src/hooks'),
      '@/styles': path.resolve(__dirname, './src/styles'),
    },
  },
  server: {
    port: 3000,
    open: true,
    host: true,
  },
})
```

### `tsconfig.json`
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "jsx": "react-jsx",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  }
}
```

### `postcss.config.js`
```javascript
export default {
  plugins: {
    '@tailwindcss/postcss': {},
  },
}
```

---

## 🎯 Default Login Credentials

The app has **three roles** you can test:

1. **Admin** - Full access to all features
2. **Officer** - Limited to operational views
3. **Security Authority** - Mobile-optimized view

**Note**: Currently using mock authentication. Click any role to login.

---

## 🚨 Real-time Features

### Incident Simulation
The app includes a real-time incident simulation system:

- **Auto-generates incidents** every 10-15 seconds
- **Toast notifications** with sound alerts
- **Live activity feed** updates
- **Badge counters** on navigation tabs

Toggle sound on/off using the volume button in the header.

---

## 📱 Responsive Breakpoints

The app is fully responsive:

- **Mobile**: `< 640px` - 1-2 column layouts
- **Tablet**: `640px - 1024px` - 2-3 column layouts
- **Desktop**: `> 1024px` - 3-4 column layouts
- **Large Desktop**: `> 1536px` - 4+ column layouts

---

## 🎨 Theme System

### Toggle Theme
Click the Sun/Moon icon in the header to switch between light and dark modes.

### Theme is persisted in localStorage
The `ThemeProvider` component saves your preference.

### Vigil Brand Colors

**Light Mode:**
- Primary: `#3b82f6` (Blue)
- Secondary: `#06b6d4` (Cyan)
- Background: Soft warm white

**Dark Mode:**
- Primary: `#06b6d4` (Electric Cyan)
- Secondary: `#f59e0b` (Amber)
- Background: `#0a0e1a` (Deep charcoal)

---

## 🔍 Camera Grid Layouts

The DVR Camera Grid supports 4 layout modes:

1. **2×2** - 4 cameras
2. **2×3** - 6 cameras (default)
3. **3×3** - 9 cameras
4. **4×4** - 16 cameras

All maintain 16:9 aspect ratio for professional DVR look.

---

## 📊 Analytics & Charts

The Analytics tab uses **Recharts** library:

- **Bar Charts** - Incidents per day
- **Area Charts** - Response time trends
- **Pie Charts** - Incidents by zone
- **Line Charts** - AI accuracy trends

All charts are responsive and theme-aware.

---

## 🛠️ Development Commands

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Run TypeScript compiler
npm run lint
```

---

## 📦 Build for Production

```bash
npm run build
```

This creates an optimized production build in the `/dist` folder.

**Build optimizations included:**
- Code splitting (React, Charts, Motion vendors)
- CSS minification
- Tree shaking
- Asset optimization

Deploy the `/dist` folder to any static hosting service:
- Vercel
- Netlify
- AWS S3 + CloudFront
- GitHub Pages

---

## 🐛 Troubleshooting

### Issue: `motion` import errors
**Solution**: Make sure you import from `motion/react`:
```typescript
import { motion } from 'motion/react'
```

### Issue: Tailwind classes not working
**Solution**: 
1. Check that `globals.css` has `@import "tailwindcss"`
2. Verify `postcss.config.js` uses `@tailwindcss/postcss`
3. Restart dev server

### Issue: Module not found errors
**Solution**: 
1. Ensure all files are in `/src/` directory
2. Check import paths use relative `./` or alias `@/`
3. Run `npm install` again

### Issue: Types errors
**Solution**:
1. Ensure `@types/react` and `@types/react-dom` are installed
2. Check `tsconfig.json` includes `"jsx": "react-jsx"`

---

## 🚀 Performance Tips

1. **Lazy load heavy components**:
```typescript
const AnalyticsView = lazy(() => import('./components/AnalyticsView'))
```

2. **Memoize expensive computations**:
```typescript
const filteredIncidents = useMemo(() => 
  incidents.filter(i => i.status === 'active'), 
  [incidents]
)
```

3. **Virtual scrolling for long lists**:
Consider using `react-virtual` for tables with 100+ rows

4. **Optimize images**:
Use WebP format and appropriate sizes

---

## 📚 Additional Resources

- **Vite Documentation**: https://vitejs.dev
- **React Documentation**: https://react.dev
- **Tailwind CSS v4**: https://tailwindcss.com/docs/v4-beta
- **Recharts**: https://recharts.org
- **Motion (Framer Motion)**: https://motion.dev
- **Lucide Icons**: https://lucide.dev

---

## 🎉 You're All Set!

Your Vigil surveillance system should now be running locally. The app will look **exactly the same** as the Figma Make version with all features working identically.

### Next Steps:
1. Customize mock data in each component
2. Connect to real backend APIs
3. Add WebSocket for true real-time updates
4. Implement actual authentication
5. Connect to real camera feeds

**Happy coding!** 🚀