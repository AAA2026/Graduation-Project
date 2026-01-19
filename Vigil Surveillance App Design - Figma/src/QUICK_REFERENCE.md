# Vigil - Quick Reference Guide

Quick commands and tips for daily development.

## 🚀 Common Commands

```bash
# Start development server
npm start

# Build for production  
npm run build

# Preview production build
npm run serve

# Install dependencies (first time)
npm install

# Clean everything and start fresh
npm run clean && npm install
```

## 📂 Key Files to Edit

### Styling & Branding
- `/src/styles/globals.css` - Theme colors, fonts
- `/src/components/VigilLogo.jsx` - Logo component
- `/src/components/ThemeProvider.jsx` - Theme system

### Main Components
- `/src/App.jsx` - Main app & authentication
- `/src/components/ModernSecurityLayout.jsx` - Dashboard layout
- `/src/components/LoginScreen.jsx` - Login page

### Role-Specific Dashboards
- `/src/components/AdminDashboard.jsx` - Admin interface
- `/src/components/OfficerDashboard.jsx` - Officer interface  
- `/src/components/SecurityAuthorityApp.jsx` - Security authority interface

### Camera System
- `/src/components/DVRCameraGrid.jsx` - Camera grid layout
- `/src/components/LiveCameraGrid.jsx` - Live camera feeds
- `/src/components/MockCameraFeed.jsx` - Simulated camera feed

### Incidents
- `/src/hooks/useRealtimeIncidents.js` - Incident simulation/data
- `/src/components/IncidentDetailModal.jsx` - Incident popup
- `/src/components/VideoModal.jsx` - Video playback

## 🎨 Quick Customizations

### Change Theme Colors

Edit `/src/styles/globals.css`:

```css
:root {
  --background: 222 47% 11%;    /* Main background color */
  --foreground: 0 0% 95%;       /* Main text color */
  --primary: 186 100% 50%;      /* Primary accent (cyan) */
  --accent: 38 100% 50%;        /* Secondary accent (amber) */
}
```

### Change Number of Cameras

Edit `/src/components/DVRCameraGrid.jsx`:

```javascript
const mockCameras = [
  { id: 1, name: 'Entrance', location: 'Main Gate', status: 'online' },
  { id: 2, name: 'Parking', location: 'Lot A', status: 'online' },
  // Add or remove cameras here
];
```

Change grid layout class:
```javascript
// For 4 cameras (2x2):
<div className="grid grid-cols-2 gap-4">

// For 6 cameras (3x2):
<div className="grid grid-cols-3 gap-4">

// For 9 cameras (3x3):
<div className="grid grid-cols-3 gap-4">
```

### Change Incident Frequency

Edit `/src/hooks/useRealtimeIncidents.js`:

```javascript
// Change these values (in milliseconds):
const minDelay = 5000;  // Minimum 5 seconds
const maxDelay = 15000; // Maximum 15 seconds

// Or disable auto-generation:
// Comment out the useEffect that calls simulateIncident()
```

### Add New Navigation Tab

Edit `/src/components/ModernSecurityLayout.jsx`:

```javascript
// 1. Add tab to tabs array:
const tabs = [
  // ... existing tabs
  { id: 'mynewpage', label: 'My Page', icon: Star },
];

// 2. Add content in the switch statement:
{currentView === 'mynewpage' && <MyNewPageComponent />}
```

## 🔌 API Integration Points

Replace mock data with real API calls in these files:

### Authentication
- `/src/components/LoginScreen.jsx` - `handleLogin` function

### Live Status
- `/src/hooks/useLiveStatus.js` - Polls `/api/live-status`

### Incidents
- `/src/hooks/useRealtimeIncidents.js` - Replace simulation with WebSocket

### Cameras
- `/src/components/DVRCameraGrid.jsx` - Fetch camera list
- `/src/components/MockCameraFeed.jsx` - Replace with real stream

### Users (Admin)
- `/src/components/UserManagement.jsx` - User CRUD operations

### Reports
- `/src/components/AdminReports.jsx` - Fetch reports data
- `/src/components/OfficerReports.jsx` - Officer reports

## 🌐 Environment Variables

Create `.env` file:

```env
# API Configuration
REACT_APP_API_URL=http://localhost:5000/api
REACT_APP_WS_URL=ws://localhost:5000

# Polling intervals (milliseconds)
REACT_APP_STATUS_POLL_INTERVAL=3000
REACT_APP_INCIDENT_POLL_INTERVAL=5000

# Feature flags
REACT_APP_ENABLE_SOUND_ALERTS=true
REACT_APP_ENABLE_NOTIFICATIONS=true
```

Use in code:
```javascript
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';
```

## 🎯 Component Patterns

### Creating a New Component

```javascript
// /src/components/MyComponent.jsx
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';

export function MyComponent({ title, data }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        {/* Your content */}
      </CardContent>
    </Card>
  );
}
```

### Using State

```javascript
import { useState } from 'react';

function MyComponent() {
  const [count, setCount] = useState(0);
  
  return (
    <button onClick={() => setCount(count + 1)}>
      Count: {count}
    </button>
  );
}
```

### Fetching Data

```javascript
import { useEffect, useState } from 'react';

function MyComponent() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    fetch(`${API_URL}/data`)
      .then(res => res.json())
      .then(data => {
        setData(data);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, []);
  
  if (loading) return <div>Loading...</div>;
  
  return <div>{/* Render data */}</div>;
}
```

## 🎨 Tailwind CSS Classes

### Common Patterns

```javascript
// Card with dark mode support
<div className="bg-white dark:bg-gray-900 rounded-lg shadow-lg p-6">

// Button styles
<button className="bg-cyan-500 hover:bg-cyan-600 text-white px-4 py-2 rounded">

// Grid layouts
<div className="grid grid-cols-3 gap-4">

// Flex layouts
<div className="flex items-center justify-between">

// Responsive design
<div className="hidden md:block"> {/* Show on desktop only */}
<div className="md:hidden"> {/* Show on mobile only */}

// Colors with dark mode
<p className="text-gray-900 dark:text-white">
```

## 🔍 Debugging

### Browser Console

```javascript
// Add console logs
console.log('Debug:', { state, props, data });

// Check component renders
useEffect(() => {
  console.log('Component mounted');
}, []);
```

### React DevTools

1. Install React DevTools browser extension
2. Open DevTools (F12)
3. Go to "Components" tab
4. Inspect component state and props

### Network Requests

1. Open DevTools (F12)
2. Go to "Network" tab
3. Filter by "Fetch/XHR"
4. See all API calls

## 📦 Adding New Dependencies

```bash
# Install a package
npm install package-name

# Install specific version
npm install package-name@version

# Remove a package
npm uninstall package-name
```

### Popular Packages

```bash
# Date handling
npm install date-fns

# Form validation
npm install react-hook-form@7.55.0

# State management (if needed)
npm install zustand

# HTTP client
npm install axios
```

## 🚨 Common Issues & Fixes

### Port Already in Use
```bash
# Change port in webpack.config.js:
devServer: { port: 3001 }
```

### Module Not Found
```bash
npm install
# or
rm -rf node_modules package-lock.json && npm install
```

### Styles Not Updating
```bash
# Hard refresh browser
Ctrl + Shift + R (Windows/Linux)
Cmd + Shift + R (Mac)
```

### Build Fails
```bash
rm -rf dist
npm run build
```

## 📱 Testing on Mobile

1. Find your IP address:
   ```bash
   # Mac/Linux
   ifconfig | grep inet
   
   # Windows
   ipconfig
   ```

2. Update webpack.config.js:
   ```javascript
   devServer: {
     host: '0.0.0.0',
     port: 3000,
   }
   ```

3. Access from mobile: `http://YOUR_IP:3000`

## 🎯 Performance Tips

### Optimize Images
- Use WebP format when possible
- Compress images before using
- Use lazy loading for off-screen images

### Code Splitting
```javascript
// Lazy load components
import { lazy, Suspense } from 'react';

const AdminDashboard = lazy(() => import('./components/AdminDashboard'));

function App() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <AdminDashboard />
    </Suspense>
  );
}
```

### Memoization
```javascript
import { useMemo, useCallback } from 'react';

// Memoize expensive calculations
const expensiveValue = useMemo(() => {
  return computeExpensiveValue(data);
}, [data]);

// Memoize callback functions
const handleClick = useCallback(() => {
  doSomething();
}, [dependency]);
```

## 📊 Project Stats

After running conversion:
- **~85+ React components**
- **~50+ UI components** (buttons, cards, modals, etc.)
- **Custom hooks** for real-time features
- **Full theme system** (light/dark)
- **Responsive design** for all screen sizes
- **Production-ready** build configuration

## 🔗 Useful Links

- **React Docs**: https://react.dev/
- **Tailwind CSS**: https://tailwindcss.com/docs
- **Lucide Icons**: https://lucide.dev/
- **Recharts**: https://recharts.org/
- **Motion**: https://motion.dev/

## 💡 Pro Tips

1. **Use .gitignore**: Already included - keeps repo clean
2. **Comment your code**: Future you will thank you
3. **Test often**: Run `npm start` frequently while developing
4. **Commit often**: If using Git, commit after each feature
5. **Keep it simple**: Don't over-engineer initially

---

**This is your go-to reference for quick lookups. Bookmark it!** 🔖
