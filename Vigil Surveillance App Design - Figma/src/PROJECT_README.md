# Vigil - AI-Powered Smart Surveillance System

A modern, full-featured surveillance system with real-time AI-powered incident detection, designed for easy setup and backend integration.

![Vigil Surveillance System](https://images.unsplash.com/photo-1557597774-9d273605dfa9?w=1200&h=400&fit=crop)

## Features

✅ **Three Role-Based Dashboards**
- Admin Dashboard (full system control)
- Officer Dashboard (monitoring & reports)
- Security Authority Mobile App (field operations)

✅ **Real-Time AI Detection**
- Violence Detection
- Car Crash Detection
- People Counter

✅ **Advanced Camera System**
- 9-camera DVR-style grid (3x3 layout)
- Live video feeds with AI model badges
- Fullscreen playback controls
- Multi-camera simultaneous viewing

✅ **Smart Features**
- Real-time incident notifications (toast + sound alerts)
- Live analytics and charts
- Interactive map view with incident markers
- Comprehensive reporting system
- Dark/Light theme support with glassmorphism UI

---

## Quick Start (5 Minutes)

### Prerequisites
- Node.js 18+ installed ([Download here](https://nodejs.org/))
- VS Code (recommended) or any code editor

### Installation

1. **Download & Extract** this project folder

2. **Open in VS Code**
   ```bash
   cd vigil-surveillance-system
   code .
   ```

3. **Install Dependencies**
   ```bash
   npm install
   ```

4. **Start Development Server**
   ```bash
   npm start
   ```

5. **Open in Browser**
   - Automatically opens at http://localhost:3000
   - Login with any of the demo credentials below

### Demo Credentials

| Role | Username | Password |
|------|----------|----------|
| Admin | admin | admin123 |
| Officer | officer | officer123 |
| Security Authority | security | security123 |

---

## Project Structure

```
vigil-surveillance-system/
├── public/
│   ├── index.html          # HTML template
│   └── favicon.ico         # App icon
├── src/
│   ├── components/         # All React components
│   │   ├── ui/            # Reusable UI components (buttons, cards, etc.)
│   │   ├── hooks/         # Custom React hooks
│   │   ├── LoginScreen.jsx
│   │   ├── ModernSecurityLayout.jsx
│   │   ├── DVRCameraGrid.jsx
│   │   ├── IncidentsView.jsx
│   │   ├── AnalyticsView.jsx
│   │   └── ... (all other components)
│   ├── hooks/             # Shared custom hooks
│   ├── utils/             # Utility functions
│   ├── styles/
│   │   ├── globals.css    # Global styles & Tailwind
│   │   └── animated-background.css
│   ├── index.js           # App entry point
│   └── App.jsx            # Main app component
├── webpack.config.js      # Webpack configuration
├── package.json           # Dependencies & scripts
├── postcss.config.js      # PostCSS configuration
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

---

## Available Scripts

### Development
```bash
npm start                  # Start development server (localhost:3000)
```

### Production
```bash
npm run build             # Build for production (creates /dist folder)
npm run serve             # Preview production build locally
```

### Maintenance
```bash
npm run clean             # Clean build artifacts and node_modules
```

---

## Building for Production

1. **Create Production Build**
   ```bash
   npm run build
   ```

2. **Output**
   - Optimized files in `/dist` folder
   - Ready to deploy to any static hosting (Netlify, Vercel, AWS S3, etc.)

3. **Preview Locally**
   ```bash
   npm run serve
   ```

---

## Customization Guide

### 🎨 Change Theme Colors

Edit `/src/styles/globals.css`:

```css
:root {
  --color-primary-cyan: 0 227 255;      /* Main cyan accent */
  --color-primary-amber: 251 191 36;    /* Warning/highlight color */
  --color-dark-bg: 10 15 30;            /* Dark background */
}
```

### 📹 Modify Camera Grid

Edit `/src/components/DVRCameraGrid.jsx`:

```javascript
// Change number of cameras (line ~15)
const totalCameras = 9;  // Change to 4, 9, 16, etc.

// Modify grid layout (line ~400)
className="grid grid-cols-3 grid-rows-3"  // Change to grid-cols-2, etc.
```

### 🔔 Customize Notifications

Edit `/src/components/ModernSecurityLayout.jsx`:

```javascript
// Change notification position (line ~250)
<Toaster position="top-right" />

// Modify sound alerts (line ~180)
const alertSound = new Audio('/alert.mp3');
```

### 🤖 Add New AI Models

Edit `/src/components/AIModelManagement.jsx`:

```javascript
// Add to initialModels array (line ~8)
{
  id: 'new-model',
  name: 'Custom Detection',
  type: 'custom',
  status: 'active',
  accuracy: 0.92,
  // ... other properties
}
```

---

## Backend Integration

This is a **frontend-only** application. To add a backend:

### Option 1: REST API

```javascript
// Example: Fetch real incidents from your API
// Edit /src/components/hooks/useRealtimeIncidents.jsx

const fetchIncidents = async () => {
  const response = await fetch('https://your-api.com/incidents');
  const data = await response.json();
  setIncidents(data);
};
```

### Option 2: WebSocket (Real-time)

```javascript
// Example: Connect to WebSocket server
// Add to /src/components/ModernSecurityLayout.jsx

useEffect(() => {
  const ws = new WebSocket('wss://your-server.com/live');
  ws.onmessage = (event) => {
    const incident = JSON.parse(event.data);
    handleNewIncident(incident);
  };
}, []);
```

### Option 3: Firebase/Supabase

```javascript
// Install: npm install firebase
// Create /src/firebase.js and configure

import { initializeApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';

const firebaseConfig = { /* your config */ };
const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);
```

### Recommended Backend Structure

```
backend/
├── /api
│   ├── /incidents      # GET, POST incidents
│   ├── /cameras        # Manage cameras
│   ├── /analytics      # Fetch analytics data
│   └── /auth           # User authentication
├── /ai-models          # AI detection services
└── /websocket          # Real-time updates
```

---

## Key Files to Edit

### Add Real Authentication
📄 `/src/components/LoginScreen.jsx` (line 45-70)
- Replace demo login with API calls
- Add JWT token handling

### Connect Real Cameras
📄 `/src/components/DVRCameraGrid.jsx` (line 120-160)
- Replace mock camera URLs with real streams
- Integrate RTSP/HLS video sources

### Use Real Analytics
📄 `/src/components/AnalyticsView.jsx` (line 15-45)
- Fetch real data from your analytics API
- Update chart data structures

### Implement Real Notifications
📄 `/src/components/ModernSecurityLayout.jsx` (line 85-140)
- Connect to WebSocket server
- Handle real-time incident streams

---

## Common Customizations

### 1. Change App Name/Branding

Edit `/src/components/VigilLogo.jsx`:
```javascript
<span>Your Company Name</span>
```

Edit `/public/index.html`:
```html
<title>Your App Name</title>
```

### 2. Add More Dashboard Views

Create new component in `/src/components/YourView.jsx`:
```javascript
export function YourView() {
  return <div>Your custom view</div>;
}
```

Add to `/src/components/ModernSecurityLayout.jsx`:
```javascript
import { YourView } from './YourView';

// Add to navigation tabs
const tabs = [
  { id: 'your-view', label: 'Your View', icon: YourIcon },
  // ... existing tabs
];
```

### 3. Modify User Roles

Edit `/src/App.jsx` to add/remove roles:
```javascript
const handleLogin = (role) => {
  // Add custom role logic
  if (role === 'custom-role') {
    setCurrentRole('custom-role');
  }
};
```

---

## Troubleshooting

### Port 3000 Already in Use
```bash
# Use different port
PORT=3001 npm start
```

### Build Errors
```bash
# Clean and reinstall
npm run clean
npm install
npm start
```

### Styling Issues
```bash
# Rebuild Tailwind CSS
npm start
# (Webpack will recompile automatically)
```

### Module Not Found
- Check import paths are relative to /src
- Webpack aliases: `@/components` = `/src/components`

---

## Tech Stack

- **Framework:** React 18
- **Styling:** Tailwind CSS v4
- **Animations:** Motion (Framer Motion)
- **Charts:** Recharts
- **Icons:** Lucide React
- **Build Tool:** Webpack 5
- **Toast Notifications:** Sonner

---

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

---

## Performance Tips

1. **Lazy Load Components**
   ```javascript
   const HeavyComponent = lazy(() => import('./HeavyComponent'));
   ```

2. **Optimize Images**
   - Use WebP format
   - Compress before adding to /public

3. **Code Splitting**
   - Webpack automatically splits vendor code
   - Keep components modular

---

## Deployment

### Netlify (Recommended)
1. Connect your Git repository
2. Build command: `npm run build`
3. Publish directory: `dist`

### Vercel
```bash
npm install -g vercel
vercel --prod
```

### AWS S3 + CloudFront
```bash
npm run build
aws s3 sync dist/ s3://your-bucket-name
```

### Traditional Hosting
1. Run `npm run build`
2. Upload `/dist` folder contents
3. Configure server to redirect all routes to `index.html`

---

## VS Code Recommended Extensions

- ES7+ React/Redux/React-Native snippets
- Tailwind CSS IntelliSense
- Prettier - Code formatter
- ESLint

---

## Support & Documentation

- 📧 Email: support@vigil-app.com
- 📚 Full Docs: [Link to full documentation]
- 🐛 Report Issues: [GitHub Issues]

---

## License

MIT License - Feel free to use for personal or commercial projects

---

## Next Steps

After setup, you might want to:

1. ✅ Customize theme colors to match your brand
2. ✅ Add your backend API endpoints
3. ✅ Replace demo data with real data sources
4. ✅ Implement real user authentication
5. ✅ Deploy to production hosting
6. ✅ Set up CI/CD pipeline

---

**Made with ❤️ for smart surveillance**

*Last Updated: January 2026*
