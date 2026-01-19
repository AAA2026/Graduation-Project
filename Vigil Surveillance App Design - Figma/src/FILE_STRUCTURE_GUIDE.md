# File Structure & Editing Guide

## Quick Reference: What to Edit & Where

This guide tells you **exactly which files to edit** for common customizations.

---

## 📁 Project Structure Overview

```
vigil-surveillance-system/
│
├── 🟢 /src/                          ← EDIT THESE FILES (Active JavaScript)
│   ├── /components/                  ← Your React components
│   ├── /hooks/                       ← Custom React hooks (if added)
│   ├── /utils/                       ← Utility functions (if added)
│   ├── /styles/                      ← CSS files
│   ├── App.jsx                       ← Main app component
│   └── index.js                      ← Entry point
│
├── 🔴 /components/                   ← DON'T EDIT (Legacy TypeScript)
│   └── (Various .tsx files)          ← For reference only
│
├── 🟡 /public/                       ← Static assets
│   ├── index.html                    ← HTML template
│   └── favicon.ico                   ← App icon
│
├── 🔴 /node_modules/                 ← DON'T EDIT (Auto-generated)
├── 🔴 /dist/                         ← DON'T EDIT (Build output)
│
├── package.json                      ← Dependencies & scripts
├── webpack.config.js                 ← Build configuration
├── postcss.config.js                 ← CSS processing
├── .gitignore                        ← Git ignore rules
│
└── 📚 Documentation Files
    ├── PROJECT_README.md
    ├── INSTALLATION_GUIDE.md
    ├── BACKEND_INTEGRATION_GUIDE.md
    └── (this file)
```

### Color Code:
- 🟢 **Green** = Edit freely (active code)
- 🟡 **Yellow** = Edit carefully (config/assets)
- 🔴 **Red** = Don't edit (auto-generated or legacy)

---

## Common Customizations

### 1. Change App Name/Branding

#### Update App Title
**File:** `/public/index.html` (line 8)

```html
<title>Your App Name Here</title>
```

#### Update Logo Component
**File:** `/components/VigilLogo.tsx` (lines 30-35)

```tsx
<span className="...">
  Your Company Name  {/* Change "Vigil" to your name */}
</span>
```

#### Update Logo in Header
**File:** `/components/ModernSecurityLayout.tsx` (search for "VigilLogo")

---

### 2. Change Theme Colors

#### Primary Colors
**File:** `/src/styles/globals.css` (lines 72-88)

```css
:root {
  --vigil-primary: #3b82f6;        /* Change this */
  --vigil-secondary: #06b6d4;      /* And this */
  --vigil-success: #10b981;
  --vigil-warning: #f59e0b;
  --vigil-danger: #ef4444;
}
```

#### Dark Mode Colors
**File:** `/src/styles/globals.css` (lines 91-140)

```css
.dark {
  --background: oklch(0.145 0 0);  /* Dark background */
  --vigil-dark-bg: #0a0f1e;        /* Main dark color */
  /* ... etc */
}
```

#### Accent Colors (Cyan/Amber)
**File:** `/components/vigil-theme.ts` (lines 15-30)

```typescript
export const vigilClasses = {
  accentCyan: 'text-cyan-400',     /* Change cyan shades */
  accentAmber: 'text-amber-500',   /* Change amber shades */
  // ...
};
```

---

### 3. Modify Camera Grid

#### Change Number of Cameras
**File:** `/components/DVRCameraGrid.tsx` (line 15)

```javascript
const totalCameras = 9;  // Change to 4, 16, 25, etc.
```

#### Change Grid Layout
**File:** `/components/DVRCameraGrid.tsx` (search for "grid-cols-3")

```javascript
// For 9 cameras (3x3)
className="grid grid-cols-3 grid-rows-3 gap-1"

// For 4 cameras (2x2)
className="grid grid-cols-2 grid-rows-2 gap-1"

// For 16 cameras (4x4)
className="grid grid-cols-4 grid-rows-4 gap-1"
```

#### Camera Placeholders
**File:** `/components/DVRCameraGrid.tsx` (lines 30-60)

```javascript
const mockCameras = [
  {
    id: 1,
    name: 'Your Camera Name',      // Edit names
    location: 'Your Location',      // Edit locations
    streamUrl: '...',               // Will be from backend later
    aiModel: {
      type: 'violence',             // Edit AI model type
      // ...
    }
  },
  // ... add more cameras
];
```

---

### 4. Customize AI Models

#### Add/Edit AI Model Types
**File:** `/components/AIModelManagement.tsx` (lines 8-50)

```javascript
const initialModels = [
  {
    id: 'violence-detection',
    name: 'Violence Detection',
    type: 'violence',              // Model type identifier
    status: 'active',
    accuracy: 0.95,
    // ...
  },
  // Add your custom model:
  {
    id: 'custom-model',
    name: 'Your Custom Model',
    type: 'custom',
    status: 'active',
    accuracy: 0.92,
    description: 'Your model description',
  }
];
```

#### Model Type Badges/Colors
**File:** `/components/DVRCameraGrid.tsx` (search for "modelTypeColors")

```javascript
const modelTypeColors = {
  violence: 'bg-red-500/90',       // Violence badge color
  car_crash: 'bg-orange-500/90',   // Car crash badge color
  people_counter: 'bg-blue-500/90', // People counter badge color
  custom: 'bg-purple-500/90',      // Add your custom colors
};
```

---

### 5. Modify Login Screen

#### Remove/Add Login Roles
**File:** `/components/LoginScreen.tsx` (lines 70-140)

```tsx
const roles = [
  {
    id: 'admin',
    title: 'Admin Dashboard',
    description: 'Full system control',
    icon: Shield,
    gradient: 'from-cyan-500 to-blue-600',
  },
  // Add your custom role:
  {
    id: 'custom-role',
    title: 'Custom Role',
    description: 'Custom permissions',
    icon: YourIcon,  // Import from lucide-react
    gradient: 'from-purple-500 to-pink-600',
  }
];
```

#### Change Demo Credentials
**File:** `/components/LoginScreen.tsx` (search for "handleLoginAttempt")

```javascript
const validCredentials = {
  admin: { password: 'admin123', name: 'Admin User' },
  officer: { password: 'officer123', name: 'Officer Smith' },
  // Add custom:
  custom: { password: 'custom123', name: 'Custom User' }
};
```

---

### 6. Customize Navigation Tabs

#### Add/Remove Dashboard Tabs
**File:** `/components/ModernSecurityLayout.tsx` (lines 200-280)

```javascript
const adminTabs = [
  { id: 'live', label: 'Live Feed', icon: Camera },
  { id: 'incidents', label: 'Incidents', icon: AlertCircle },
  // Add custom tab:
  { id: 'custom', label: 'Custom View', icon: YourIcon },
];
```

#### Create Custom View Component
**Create new file:** `/src/components/CustomView.jsx`

```javascript
export function CustomView() {
  return (
    <div className="...">
      <h2>Your Custom View</h2>
      {/* Your custom content */}
    </div>
  );
}
```

#### Wire Up Custom View
**File:** `/components/ModernSecurityLayout.tsx` (in renderContent function)

```javascript
import { CustomView } from '../src/components/CustomView';

const renderContent = () => {
  switch (currentView) {
    case 'live': return <DVRCameraGrid />;
    case 'incidents': return <IncidentsView />;
    case 'custom': return <CustomView />;  // Add this
    // ...
  }
};
```

---

### 7. Modify Notifications

#### Change Notification Position
**File:** `/components/ModernSecurityLayout.tsx` (search for "Toaster")

```tsx
<Toaster 
  position="top-right"  // Options: top-left, top-center, top-right,
                        //          bottom-left, bottom-center, bottom-right
  theme={theme}
/>
```

#### Customize Notification Sound
**File:** `/components/ModernSecurityLayout.tsx` (search for "alertSound")

```javascript
const alertSound = new Audio('/your-custom-alert.mp3');
alertSound.volume = 0.5;  // Adjust volume
```

#### Change Notification Duration
**File:** `/components/ModernSecurityLayout.tsx` (search for "toast.error")

```javascript
toast.error('Message', {
  duration: 5000,  // milliseconds (default: 4000)
});
```

---

### 8. Customize Analytics Charts

#### Edit Chart Colors
**File:** `/components/AnalyticsView.tsx` (lines 50-100)

```javascript
<Line 
  stroke="#06b6d4"     // Change line color
  strokeWidth={2}       // Change line thickness
  dot={{ fill: '#06b6d4', r: 4 }}  // Change dot color/size
/>
```

#### Add Custom Chart
**File:** `/components/AnalyticsView.tsx`

```javascript
import { PieChart, Pie } from 'recharts';

// Add in component:
<PieChart width={300} height={300}>
  <Pie data={yourData} dataKey="value" fill="#8884d8" />
</PieChart>
```

---

### 9. Modify Map View

#### Change Default Map Center
**File:** `/components/MapView.tsx` (lines 30-40)

```javascript
const [center, setCenter] = useState({
  lat: 40.7128,   // Your latitude
  lng: -74.0060   // Your longitude
});
```

#### Add Custom Markers
**File:** `/components/MapView.tsx` (search for "markers")

```javascript
const customMarkers = [
  {
    id: 1,
    position: { lat: 40.7128, lng: -74.0060 },
    type: 'incident',
    label: 'Custom Marker',
  }
];
```

---

### 10. User Management

#### Edit User Table Columns
**File:** `/components/UserManagement.tsx` (lines 80-120)

```tsx
<TableHeader>
  <TableRow>
    <TableHead>Name</TableHead>
    <TableHead>Email</TableHead>
    <TableHead>Your Custom Column</TableHead>  {/* Add custom */}
  </TableRow>
</TableHeader>
```

---

## Configuration Files

### package.json
**When to Edit:**
- Adding new npm packages
- Changing scripts
- Updating project metadata

**Common Edits:**

```json
{
  "name": "your-app-name",
  "version": "1.0.0",
  "scripts": {
    "start": "webpack serve --mode development --port 3001",  // Change port
    "custom-script": "your-custom-command"  // Add custom scripts
  },
  "dependencies": {
    "your-package": "^1.0.0"  // Add new dependencies
  }
}
```

---

### webpack.config.js
**When to Edit:**
- Changing dev server settings
- Adding webpack plugins
- Modifying build output

**Common Edits:**

```javascript
devServer: {
  port: 3001,           // Change port
  open: false,          // Disable auto-open
  hot: true,            // Hot reload
},
output: {
  filename: 'app.[contenthash].js',  // Change output filename
  publicPath: '/your-path/',          // For subdirectory hosting
}
```

---

### postcss.config.js
**When to Edit:**
- Adding PostCSS plugins
- Changing CSS processing

**Usually Don't Need to Edit**

---

## Adding New Components

### 1. Create Component File

**Create:** `/src/components/YourComponent.jsx`

```javascript
export function YourComponent() {
  return (
    <div className="...">
      <h2>Your Component</h2>
    </div>
  );
}
```

### 2. Import in Parent

**Edit:** `/src/App.jsx` or `/components/ModernSecurityLayout.tsx`

```javascript
import { YourComponent } from './components/YourComponent';

// Use it:
<YourComponent />
```

---

## Adding New Pages/Views

### 1. Create View Component

**Create:** `/src/components/YourView.jsx`

```javascript
export function YourView({ role }) {
  return (
    <div className="p-6">
      <h1>Your Custom View</h1>
      {/* Your content */}
    </div>
  );
}
```

### 2. Add to Navigation

**Edit:** `/components/ModernSecurityLayout.tsx`

```javascript
import { YourView } from '../src/components/YourView';

// Add tab
const tabs = [
  { id: 'your-view', label: 'Your View', icon: YourIcon },
  // ...
];

// Add to switch
case 'your-view':
  return <YourView role={role} />;
```

---

## Styling Guide

### Using Tailwind Classes

**Edit any component file:**

```javascript
<div className="
  bg-slate-900        // Background color
  text-white          // Text color
  p-4                 // Padding
  rounded-lg          // Border radius
  shadow-xl           // Box shadow
  hover:bg-slate-800  // Hover state
  dark:bg-slate-950   // Dark mode
">
  Content
</div>
```

### Custom CSS

**Edit:** `/src/styles/globals.css`

```css
/* Add at bottom */
.your-custom-class {
  background: linear-gradient(to right, #06b6d4, #3b82f6);
  padding: 1rem;
}
```

**Use in component:**

```javascript
<div className="your-custom-class">
  Content
</div>
```

---

## Icons

### Using Lucide Icons

**Import in component:**

```javascript
import { 
  Camera,      // Camera icon
  AlertCircle, // Alert icon
  Settings,    // Settings icon
  // ... see https://lucide.dev
} from 'lucide-react';

// Use:
<Camera className="w-6 h-6 text-cyan-400" />
```

### Icon Sizes

```javascript
<Camera className="w-4 h-4" />  // Small
<Camera className="w-6 h-6" />  // Medium
<Camera className="w-8 h-8" />  // Large
<Camera className="w-12 h-12" /> // XL
```

---

## Responsive Design

### Tailwind Responsive Classes

```javascript
<div className="
  grid 
  grid-cols-1      // 1 column on mobile
  md:grid-cols-2   // 2 columns on tablet
  lg:grid-cols-3   // 3 columns on desktop
  xl:grid-cols-4   // 4 columns on large screens
">
```

### Breakpoints

- `sm:` - 640px and up
- `md:` - 768px and up
- `lg:` - 1024px and up
- `xl:` - 1280px and up
- `2xl:` - 1536px and up

---

## File Naming Conventions

### Components
```
✅ YourComponent.jsx       (PascalCase)
❌ your-component.jsx
❌ yourComponent.jsx
```

### Utilities/Hooks
```
✅ useYourHook.jsx         (camelCase with 'use' prefix)
✅ yourUtil.js
```

### Styles
```
✅ globals.css             (kebab-case)
✅ your-styles.css
```

---

## Common Mistakes to Avoid

### ❌ Don't Edit
- Files in `/components` root (legacy TypeScript)
- Files in `/node_modules`
- Files in `/dist`
- `package-lock.json` manually

### ❌ Don't Delete
- `/public/index.html`
- `/src/index.js`
- `/src/App.jsx`
- `package.json`
- `webpack.config.js`

### ✅ Do Edit
- Files in `/src/components`
- `/src/styles/globals.css`
- `/public/index.html` (carefully)
- Component files (.jsx)

---

## Quick Reference: File Locations

| What You Want | File to Edit | Line Numbers |
|---------------|--------------|--------------|
| App name | `/public/index.html` | 8 |
| Logo text | `/components/VigilLogo.tsx` | 30-35 |
| Theme colors | `/src/styles/globals.css` | 72-88 |
| Camera count | `/components/DVRCameraGrid.tsx` | 15 |
| Login roles | `/components/LoginScreen.tsx` | 70-140 |
| Navigation tabs | `/components/ModernSecurityLayout.tsx` | 200-280 |
| Notification position | `/components/ModernSecurityLayout.tsx` | Search "Toaster" |
| Chart colors | `/components/AnalyticsView.tsx` | 50-100 |

---

## Next Steps

1. ✅ Identify what you want to customize
2. ✅ Find the file in this guide
3. ✅ Open file in VS Code
4. ✅ Make your changes
5. ✅ Save and check browser (auto-refresh)
6. ✅ Test thoroughly

---

*File Structure Guide - Last Updated: January 2026*
