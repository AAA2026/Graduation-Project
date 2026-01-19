# Vigil Setup Guide

Complete step-by-step guide to get Vigil up and running on your machine.

## 📋 Prerequisites

### Required Software
1. **Node.js** (v16 or higher)
   - Download from: https://nodejs.org/
   - Verify installation: `node --version`

2. **npm** (comes with Node.js)
   - Verify installation: `npm --version`

3. **Code Editor** (recommended)
   - VS Code: https://code.visualstudio.com/
   - WebStorm, Sublime Text, or any editor you prefer

### Optional but Recommended
- **Git** for version control
- **Browser DevTools** for debugging

## 🚀 Installation Steps

### Step 1: Download the Project

Option A: If you have the ZIP file
```bash
# Extract the ZIP file to your desired location
cd /path/to/vigil-surveillance-system
```

Option B: If using Git
```bash
git clone <repository-url>
cd vigil-surveillance-system
```

### Step 2: Install Dependencies

```bash
# Using npm (recommended)
npm install

# Or using yarn
yarn install
```

**Wait time**: 2-5 minutes depending on internet speed

Expected output:
```
added 1234 packages in 2m
```

### Step 3: Verify Installation

Check that all dependencies installed correctly:
```bash
npm list --depth=0
```

You should see packages like:
- react@18.3.1
- lucide-react
- recharts
- motion
- etc.

### Step 4: Start Development Server

```bash
npm start
```

Expected output:
```
<i> [webpack-dev-server] Project is running at:
<i> [webpack-dev-server] Loopback: http://localhost:3000/
✔ Compiled successfully in 3456ms
```

Your browser should automatically open to `http://localhost:3000`

### Step 5: Test the Application

1. **Login Screen** should appear
2. **Enter any credentials**:
   - Username: `admin` (or anything)
   - Password: `password` (or anything)
3. **Select Role**: Admin, Officer, or Security Authority
4. **Click "Sign In"**
5. **Dashboard should load** with camera feeds

## 🔧 Configuration

### Change Port Number

Edit `webpack.config.js`:
```javascript
devServer: {
  port: 3001, // Change from 3000 to your preferred port
}
```

### Configure API Endpoints

Create `.env` file in root directory:
```env
REACT_APP_API_URL=http://localhost:5000/api
REACT_APP_WS_URL=ws://localhost:5000
REACT_APP_POLLING_INTERVAL=3000
```

Use in code:
```javascript
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';
```

### Customize Theme Colors

Edit `/src/styles/globals.css`:
```css
:root {
  --background: 222 47% 11%;    /* Main background */
  --primary: 186 100% 50%;      /* Cyan accent */
  --accent: 38 100% 50%;        /* Amber accent */
}
```

### Change Camera Grid Layout

Edit `/src/components/DVRCameraGrid.jsx`:
```javascript
// Change from 6 cameras to 4 cameras
const mockCameras = [
  // Remove last 2 cameras, keep first 4
];
```

## 🏗️ Building for Production

### Step 1: Create Production Build

```bash
npm run build
```

This creates an optimized build in the `/dist` folder.

### Step 2: Test Production Build Locally

```bash
npm run serve
```

Opens `http://localhost:3000` with production build.

### Step 3: Deploy

Copy the `/dist` folder to your hosting service:

**Netlify** (drag & drop):
1. Go to https://app.netlify.com/drop
2. Drag `/dist` folder
3. Site is live!

**Vercel**:
```bash
npm install -g vercel
vercel deploy --prod
```

**Apache/Nginx**:
```bash
# Copy dist folder to web server
cp -r dist/* /var/www/html/
```

## 🎨 Customization Guide

### Adding New Pages/Views

1. Create component in `/src/components/YourPage.jsx`
2. Add to navigation in `/src/components/ModernSecurityLayout.jsx`

Example:
```javascript
// In ModernSecurityLayout.jsx
const tabs = [
  // ... existing tabs
  { id: 'reports', label: 'Reports', icon: FileText },
];

// In render section
{currentView === 'reports' && <YourPage />}
```

### Adding New UI Components

1. Check if component exists in `/src/components/ui/`
2. If not, create new component
3. Follow existing patterns

Example button:
```javascript
import { Button } from './components/ui/button';

<Button variant="default">Click Me</Button>
```

### Integrating Real Camera Feeds

Replace mock camera in `/src/components/MockCameraFeed.jsx`:
```javascript
// Instead of canvas animation
<img src={`${API_URL}/cameras/${cameraId}/stream`} alt="Camera Feed" />

// Or for HLS/RTMP streams
<video src={streamUrl} autoPlay muted />
```

### Adding Real-Time WebSocket Updates

Create WebSocket hook:
```javascript
// /src/hooks/useWebSocket.js
import { useEffect, useState } from 'react';

export function useWebSocket(url) {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    const ws = new WebSocket(url);
    
    ws.onmessage = (event) => {
      setData(JSON.parse(event.data));
    };
    
    return () => ws.close();
  }, [url]);
  
  return data;
}
```

Use in component:
```javascript
const liveData = useWebSocket('ws://localhost:5000/incidents');
```

## 🔌 Backend Integration

### API Setup

1. Create backend service (Node.js, Python, etc.)
2. Implement endpoints (see README.md)
3. Enable CORS for development

Example Express.js CORS:
```javascript
const cors = require('cors');
app.use(cors({
  origin: 'http://localhost:3000'
}));
```

### Replace Mock Data

Find and replace in components:
```javascript
// OLD (mock):
const [incidents, setIncidents] = useState(mockIncidents);

// NEW (real API):
const [incidents, setIncidents] = useState([]);

useEffect(() => {
  fetch(`${API_URL}/incidents`)
    .then(res => res.json())
    .then(data => setIncidents(data))
    .catch(err => console.error(err));
}, []);
```

### Authentication Integration

Update `/src/components/LoginScreen.jsx`:
```javascript
const handleLogin = async (username, password, role) => {
  try {
    const response = await fetch(`${API_URL}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, role })
    });
    
    if (response.ok) {
      const { token, user } = await response.json();
      localStorage.setItem('token', token);
      onLogin(user.role);
    }
  } catch (error) {
    console.error('Login failed:', error);
  }
};
```

## 🐛 Troubleshooting

### Problem: "npm install" fails

**Solution**:
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and package-lock.json
rm -rf node_modules package-lock.json

# Reinstall
npm install
```

### Problem: Port 3000 already in use

**Solution**:
```bash
# Find process using port 3000
lsof -i :3000  # Mac/Linux
netstat -ano | findstr :3000  # Windows

# Kill process or change port in webpack.config.js
```

### Problem: "Module not found" errors

**Solution**:
```bash
# Reinstall specific package
npm install <package-name>

# Or reinstall all
npm install
```

### Problem: Tailwind styles not working

**Solution**:
1. Check `/src/index.js` imports CSS files
2. Verify `/postcss.config.js` exists
3. Restart dev server

### Problem: Build fails

**Solution**:
```bash
# Clean and rebuild
npm run clean
npm install
npm run build
```

### Problem: Hot reload not working

**Solution**:
1. Check webpack-dev-server is running
2. Verify `webpack.config.js` has `hot: true`
3. Restart dev server

## 📱 Mobile Development

### Test on Mobile Device

1. Find your computer's IP address:
   ```bash
   # Mac/Linux
   ifconfig | grep inet
   
   # Windows
   ipconfig
   ```

2. Update `webpack.config.js`:
   ```javascript
   devServer: {
     host: '0.0.0.0', // Allow external access
     port: 3000,
   }
   ```

3. Access from mobile: `http://YOUR_IP:3000`

## 🔒 Security Notes

### Before Production Deployment

1. **Remove console.logs**
2. **Enable authentication**
3. **Use HTTPS**
4. **Implement rate limiting**
5. **Validate all inputs**
6. **Set secure headers**

### Environment Variables

Never commit sensitive data:
```bash
# Add to .gitignore
.env
.env.local
.env.production
```

## 📞 Getting Help

### Check Documentation
1. `/README.md` - Main documentation
2. `/ARCHITECTURE.md` - System architecture
3. `/THEME_SYSTEM.md` - Theme guide
4. Component files - Inline comments

### Debug Mode

Enable detailed logging:
```javascript
// In any component
console.log('Debug:', { state, props });
```

Use React DevTools:
- Install: https://react-devtools-chrome-extension.com/
- Inspect component state and props

## ✅ Post-Installation Checklist

- [ ] Node.js installed (v16+)
- [ ] Dependencies installed (`npm install`)
- [ ] Dev server starts (`npm start`)
- [ ] App opens in browser
- [ ] Can login and see dashboard
- [ ] Camera feeds display
- [ ] Theme toggle works
- [ ] Notifications appear
- [ ] Production build works (`npm run build`)

## 🎯 Next Steps

1. **Customize branding** (logo, colors, text)
2. **Connect to backend API**
3. **Add real camera streams**
4. **Implement authentication**
5. **Deploy to production**
6. **Add monitoring/analytics**

---

**You're all set! Start building your surveillance system.** 🚀
