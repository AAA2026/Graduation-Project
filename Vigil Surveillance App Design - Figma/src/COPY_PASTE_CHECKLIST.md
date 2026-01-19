# Vigil - Copy-Paste Checklist for Vite Setup

Use this checklist to ensure you've copied all necessary files from Figma Make to your Vite project.

---

## ✅ Configuration Files (Root Directory)

Copy these files to your **project root**:

- [ ] `package.json` - Dependencies and scripts
- [ ] `vite.config.ts` - Vite configuration
- [ ] `tsconfig.json` - TypeScript configuration
- [ ] `tsconfig.node.json` - TypeScript node configuration
- [ ] `postcss.config.js` - PostCSS configuration for Tailwind v4
- [ ] `index.html` - HTML entry point
- [ ] `.gitignore` - Git ignore rules (create if needed)
- [ ] `README.md` - Project documentation (optional)

---

## ✅ Source Files (`/src` directory)

### Main Entry Points
- [ ] `src/main.tsx` - React app entry point
- [ ] `src/App.tsx` - Main app component

### Styles
- [ ] `src/styles/globals.css` - Global styles + Tailwind v4

### Hooks
- [ ] `src/hooks/useRealtimeIncidents.ts` - Real-time incident hook

---

## ✅ Components (`/src/components`)

### Core Layout
- [ ] `ModernSecurityLayout.tsx` - Main layout with navigation
- [ ] `LoginScreen.tsx` - Login screen
- [ ] `ThemeProvider.tsx` - Theme context provider
- [ ] `ThemeToggle.tsx` - Theme toggle button
- [ ] `VigilLogo.tsx` - Vigil logo component
- [ ] `VigilLogoStandalone.tsx` - Standalone logo
- [ ] `vigil-theme.ts` - Theme utilities

### Dashboard Views
- [ ] `DVRCameraGrid.tsx` - Live camera monitor
- [ ] `IncidentsView.tsx` - Incidents list view
- [ ] `AnalyticsView.tsx` - Analytics dashboard
- [ ] `CameraManagement.tsx` - Camera CRUD
- [ ] `UserManagement.tsx` - User CRUD
- [ ] `AIModelManagement.tsx` - AI model management
- [ ] `SystemHealthView.tsx` - System health monitoring
- [ ] `AIFeedbackSection.tsx` - AI feedback review
- [ ] `OfficerReports.tsx` - Reports section
- [ ] `MapView.tsx` - Map with cameras/incidents
- [ ] `NotificationCenter.tsx` - Notifications

### Supporting Components
- [ ] `IncidentDetailModal.tsx` - Incident detail modal
- [ ] `IncidentFeed.tsx` - Live incident feed
- [ ] `LiveActivityFeed.tsx` - Activity timeline
- [ ] `AdminDashboard.tsx` - Admin dashboard (if used)
- [ ] `OfficerDashboard.tsx` - Officer dashboard (if used)
- [ ] `SecurityAuthorityApp.tsx` - Security authority view
- [ ] `SecurityAuthorityDashboard.tsx` - Security authority dashboard
- [ ] `DashboardLayout.tsx` - Legacy layout (if used)
- [ ] `DashboardHome.tsx` - Dashboard home (if used)
- [ ] `CompactStatsBar.tsx` - Stats bar (if used)
- [ ] `LiveCameraGrid.tsx` - Alternative camera grid (if used)
- [ ] `CameraWall.tsx` - Camera wall (if used)
- [ ] `FloatingIncidentPanel.tsx` - Floating panel (if used)
- [ ] `EventConfirmation.tsx` - Event confirmation (if used)

---

## ✅ UI Components (`/src/components/ui`)

Copy all UI primitive components:

- [ ] `accordion.tsx`
- [ ] `alert-dialog.tsx`
- [ ] `alert.tsx`
- [ ] `aspect-ratio.tsx`
- [ ] `avatar.tsx`
- [ ] `badge.tsx`
- [ ] `breadcrumb.tsx`
- [ ] `button.tsx`
- [ ] `calendar.tsx`
- [ ] `card.tsx`
- [ ] `carousel.tsx`
- [ ] `chart.tsx`
- [ ] `checkbox.tsx`
- [ ] `collapsible.tsx`
- [ ] `command.tsx`
- [ ] `context-menu.tsx`
- [ ] `dialog.tsx`
- [ ] `drawer.tsx`
- [ ] `dropdown-menu.tsx`
- [ ] `form.tsx`
- [ ] `hover-card.tsx`
- [ ] `input-otp.tsx`
- [ ] `input.tsx`
- [ ] `label.tsx`
- [ ] `menubar.tsx`
- [ ] `navigation-menu.tsx`
- [ ] `pagination.tsx`
- [ ] `popover.tsx`
- [ ] `progress.tsx`
- [ ] `radio-group.tsx`
- [ ] `resizable.tsx`
- [ ] `scroll-area.tsx`
- [ ] `select.tsx`
- [ ] `separator.tsx`
- [ ] `sheet.tsx`
- [ ] `sidebar.tsx`
- [ ] `skeleton.tsx`
- [ ] `slider.tsx`
- [ ] `sonner.tsx` - Toast notifications
- [ ] `switch.tsx`
- [ ] `table.tsx`
- [ ] `tabs.tsx`
- [ ] `textarea.tsx`
- [ ] `toggle-group.tsx`
- [ ] `toggle.tsx`
- [ ] `tooltip.tsx`
- [ ] `use-mobile.ts` - Mobile hook
- [ ] `utils.ts` - Utility functions

---

## ✅ Figma Components (`/src/components/figma`)

- [ ] `ImageWithFallback.tsx` - Image with fallback component

---

## ✅ Dashboard Components (`/src/components/dashboard`)

- [ ] `SecurityMonitoring.tsx` - Security monitoring component

---

## ✅ After Copying Files

### 1. Install Dependencies
```bash
cd your-vite-project
npm install
```

### 2. Verify File Structure
```bash
# Check if all files are in place
ls -R src/
```

### 3. Check Imports
Ensure all imports use correct paths:
- Relative imports: `./` or `../`
- Alias imports: `@/` (if configured)

### 4. Start Dev Server
```bash
npm run dev
```

### 5. Test All Views
- [ ] Login screen loads
- [ ] Admin dashboard accessible
- [ ] Officer dashboard accessible
- [ ] Security authority dashboard accessible
- [ ] All tabs render without errors
- [ ] Theme toggle works
- [ ] Real-time incidents appear
- [ ] Modal opens correctly
- [ ] Charts render
- [ ] Forms work

---

## ✅ Common Import Fixes

If you see import errors, update these:

### ❌ Wrong (Figma Make style):
```typescript
import { Button } from './components/ui/button'
import { useRealtimeIncidents } from '../hooks/useRealtimeIncidents'
```

### ✅ Correct (Vite style):
```typescript
import { Button } from './components/ui/button'  // If same directory
import { Button } from '../ui/button'            // If in components/
import { Button } from '@/components/ui/button'  // With alias
import { useRealtimeIncidents } from '@/hooks/useRealtimeIncidents'
```

### Motion Import:
```typescript
// Always use this:
import { motion, AnimatePresence } from 'motion/react'
```

### Sonner Import:
```typescript
// Use specific version:
import { toast } from 'sonner@2.0.3'
// Or generic:
import { toast } from 'sonner'
```

---

## ✅ File Size Check

Expected file counts:
- **UI Components**: ~50 files in `/src/components/ui/`
- **Main Components**: ~25 files in `/src/components/`
- **Hooks**: 1 file in `/src/hooks/`
- **Styles**: 1 file in `/src/styles/`

Total TypeScript files: **~75-80 `.tsx` and `.ts` files**

---

## ✅ Critical Files Double-Check

These files MUST be present for the app to work:

### Top Priority:
1. **package.json** - Without this, npm install won't work
2. **vite.config.ts** - Vite won't start without it
3. **tsconfig.json** - TypeScript won't compile
4. **postcss.config.js** - Tailwind v4 won't work
5. **index.html** - Entry point for Vite
6. **src/main.tsx** - React won't render
7. **src/App.tsx** - No app component
8. **src/styles/globals.css** - No styles

### Secondary Priority:
9. **ModernSecurityLayout.tsx** - Main layout
10. **LoginScreen.tsx** - Can't login without it
11. **ThemeProvider.tsx** - Theme won't work
12. **All UI components** - UI will break

---

## 📝 Final Verification Script

Run this in your project root to verify structure:

```bash
#!/bin/bash

echo "Checking Vigil project structure..."

# Check config files
files=(
  "package.json"
  "vite.config.ts"
  "tsconfig.json"
  "postcss.config.js"
  "index.html"
  "src/main.tsx"
  "src/App.tsx"
  "src/styles/globals.css"
)

for file in "${files[@]}"; do
  if [ -f "$file" ]; then
    echo "✅ $file"
  else
    echo "❌ MISSING: $file"
  fi
done

# Count component files
echo ""
echo "Component counts:"
echo "Main components: $(ls -1 src/components/*.tsx 2>/dev/null | wc -l)"
echo "UI components: $(ls -1 src/components/ui/*.tsx 2>/dev/null | wc -l)"
echo "Hooks: $(ls -1 src/hooks/*.ts 2>/dev/null | wc -l)"

echo ""
echo "Run 'npm install' to install dependencies"
echo "Then run 'npm run dev' to start the development server"
```

Save as `check-structure.sh`, make executable, and run:
```bash
chmod +x check-structure.sh
./check-structure.sh
```

---

## 🎉 You're Done!

If all checkboxes are checked and `npm run dev` works, your Vigil app is ready!

The app should look **exactly identical** to the Figma Make version with:
- Same layout and navigation
- Same colors and theme
- Same animations and transitions
- Same functionality
- Same responsive behavior

**Enjoy your production-ready surveillance system!** 🚀
