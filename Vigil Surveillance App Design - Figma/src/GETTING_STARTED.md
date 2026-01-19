# Getting Started with Vigil

Welcome! This guide will help you set up and run the Vigil surveillance system on your machine in plain JavaScript format.

## ⚡ Quick Start (5 minutes)

### Option 1: Automatic Conversion (Recommended)

1. **Run the conversion script**:
   ```bash
   node convert-to-jsx.js
   ```
   
   This will automatically convert all TypeScript files to JavaScript and place them in the `/src` directory.

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the app**:
   ```bash
   npm start
   ```

4. **Open your browser**: The app will automatically open at `http://localhost:3000`

### Option 2: Manual Setup

If you prefer manual control or the script has issues:

1. Review the `/CONVERSION_GUIDE.md` for manual conversion steps
2. Convert each file as needed
3. Follow the same install and start steps above

## 📦 What's Included

Your Vigil project includes:

```
vigil-surveillance-system/
├── 📄 Configuration Files
│   ├── package.json           # Dependencies & scripts
│   ├── webpack.config.js      # Build configuration
│   ├── .babelrc              # JavaScript transpiler config
│   └── postcss.config.js     # CSS processing
│
├── 🎨 Public Assets
│   ├── public/
│   │   └── index.html        # HTML template
│   └── favicon.ico
│
├── 💻 Source Code (after conversion)
│   └── src/
│       ├── index.js          # Entry point
│       ├── App.jsx           # Main component
│       ├── components/       # All React components
│       ├── hooks/           # Custom React hooks  
│       ├── utils/           # Utility functions
│       └── styles/          # CSS files
│
├── 📚 Documentation
│   ├── README.md            # Main documentation
│   ├── SETUP_GUIDE.md       # Detailed setup
│   ├── CONVERSION_GUIDE.md  # TS to JS conversion
│   └── GETTING_STARTED.md   # This file
│
└── 🔧 Conversion Tools
    └── convert-to-jsx.js    # Auto-converter script
```

## 🎯 Understanding the App Structure

### Entry Point Flow

1. **index.js** - Renders the app into the DOM
2. **App.jsx** - Main component with authentication logic
3. **LoginScreen.jsx** - Login interface (if not authenticated)
4. **ModernSecurityLayout.jsx** - Main dashboard (after login)

### Key Components

- **ThemeProvider.jsx** - Dark/light mode system
- **DVRCameraGrid.jsx** - Camera feed display
- **IncidentDetailModal.jsx** - Incident information
- **AdminDashboard.jsx** - Admin-specific interface
- **OfficerDashboard.jsx** - Officer-specific interface
- **SecurityAuthorityApp.jsx** - Security authority interface

### Hooks

- **useLiveStatus.js** - Polls backend for system status
- **useRealtimeIncidents.js** - Simulates real-time incidents

### Utilities

- **exportUtils.js** - PDF/Excel export functionality

## 🔑 Login & Testing

### Demo Credentials

The app accepts ANY credentials for demo purposes:

1. Navigate to `http://localhost:3000`
2. Enter any email (e.g., `admin@vigil.com`)
3. Enter any password (e.g., `password123`)
4. Select a role:
   - **Admin** - Full system access
   - **Officer** - Incident monitoring and response
   - **Security Authority** - Mobile-optimized overview

### Test Each Role

**Admin Features to Test**:
- User management
- Camera configuration
- AI model settings
- Analytics dashboard
- Demo request management

**Officer Features to Test**:
- Live camera grid
- Incident alerts (auto-generates every 5-15 seconds)
- Video playback
- Quick action buttons
- Incident resolution

**Security Authority Features to Test**:
- Mobile interface
- Map view
- Incident overview
- Quick stats

## 🎨 Customization

### Change Theme Colors

Edit `/src/styles/globals.css`:

```css
:root {
  /* Change these values */
  --background: 222 47% 11%;    /* Main background */
  --primary: 186 100% 50%;      /* Cyan accent */
  --accent: 38 100% 50%;        /* Amber accent */
}
```

Restart the server to see changes.

### Modify Camera Layout

Edit `/src/components/DVRCameraGrid.jsx`:

```javascript
// Change number of cameras displayed
const mockCameras = [
  { id: 1, name: 'Entrance', location: 'Main Gate', status: 'online' },
  { id: 2, name: 'Parking', location: 'Lot A', status: 'online' },
  // Add or remove cameras here
];
```

### Update Logo

Replace the logo by editing `/src/components/VigilLogo.jsx` or create a new logo component.

## 🔌 Backend Integration

### API Endpoints Needed

The app is designed to integrate with your backend. Implement these endpoints:

1. **POST /api/login** - Authentication
2. **GET /api/live-status** - System status (polled every 3 seconds)
3. **GET /api/incidents** - List of incidents
4. **GET /api/cameras** - Camera list and streams
5. **GET /api/users** - User management
6. **GET /api/reports** - Reports data

### Example: Replace Mock Data

In `/src/hooks/useRealtimeIncidents.js`, replace simulation with real API:

```javascript
// OLD (simulation):
const simulateIncident = () => {
  // Generates fake incidents
};

// NEW (real API):
useEffect(() => {
  const ws = new WebSocket('ws://your-api.com/incidents');
  
  ws.onmessage = (event) => {
    const incident = JSON.parse(event.data);
    setIncidents(prev => [incident, ...prev]);
  };
  
  return () => ws.close();
}, []);
```

### Environment Variables

Create `.env` in project root:

```env
REACT_APP_API_URL=http://localhost:5000/api
REACT_APP_WS_URL=ws://localhost:5000
```

Use in code:

```javascript
const API_URL = process.env.REACT_APP_API_URL;
```

## 🚀 Development Workflow

### Daily Development

```bash
# Start dev server with hot reload
npm start
```

Make changes to any `.jsx` file - they'll automatically reload in the browser.

### Build for Testing

```bash
# Create production build
npm run build

# Test production build locally
npm run serve
```

### Project Organization

**Adding a new feature**:

1. Create component: `/src/components/MyFeature.jsx`
2. Import in parent component
3. Add styling with Tailwind classes
4. Test in browser

**Adding a new page**:

1. Create component: `/src/components/MyPage.jsx`
2. Add to ModernSecurityLayout navigation
3. Add route handling in switch statement

## 📱 Deployment

### Build for Production

```bash
npm run build
```

Outputs optimized files to `/dist` directory.

### Deploy Options

**Netlify** (easiest):
1. Drag `/dist` folder to https://app.netlify.com/drop
2. Done! Your site is live

**Vercel**:
```bash
npm install -g vercel
vercel deploy --prod
```

**Your Own Server**:
```bash
# Copy dist folder to your web server
scp -r dist/* user@server:/var/www/html/
```

### Environment for Production

Create `.env.production`:
```env
REACT_APP_API_URL=https://api.your-domain.com
REACT_APP_WS_URL=wss://api.your-domain.com
```

## 🐛 Troubleshooting

### "Module not found" errors

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Port 3000 is busy

Edit `webpack.config.js`:
```javascript
devServer: {
  port: 3001, // Change port
}
```

### Styles not updating

```bash
# Hard refresh browser
Ctrl+Shift+R (Windows/Linux)
Cmd+Shift+R (Mac)

# Or restart dev server
Ctrl+C
npm start
```

### Build fails

```bash
# Clean build
rm -rf dist
npm run build
```

## 💡 Tips & Tricks

1. **Use React DevTools**: Install the browser extension to inspect components
2. **Check Console**: Open browser DevTools (F12) to see errors
3. **Hot Reload**: Save any file to see instant updates
4. **Multiple Terminals**: Run `npm start` in one, keep another free for commands
5. **Git**: Use version control to track changes

## 📚 Learning Resources

- **React**: https://react.dev/
- **Tailwind CSS**: https://tailwindcss.com/docs
- **Lucide Icons**: https://lucide.dev/
- **Recharts**: https://recharts.org/

## ✅ Verification Checklist

After setup, verify everything works:

- [ ] `npm install` completed without errors
- [ ] `npm start` opens the app in browser
- [ ] Login screen appears correctly
- [ ] Can login with any credentials
- [ ] Dashboard loads with camera feeds
- [ ] Theme toggle works (light/dark mode)
- [ ] Incident notifications appear automatically
- [ ] Can click on cameras and incidents
- [ ] All three roles work (admin, officer, security)
- [ ] `npm run build` creates `/dist` folder

## 🤝 Need Help?

1. Check the error message in the console
2. Review the documentation in `/guidelines/` folder
3. Search for the error online
4. Check component comments in code

## 🎉 You're Ready!

Your Vigil surveillance system is now set up and ready for customization. Start by:

1. Exploring the codebase
2. Customizing colors and branding
3. Connecting to your backend API
4. Adding real camera streams
5. Deploying to production

**Happy coding!** 🚀

---

*For detailed technical documentation, see README.md*
*For backend integration guide, see SETUP_GUIDE.md*
*For conversion details, see CONVERSION_GUIDE.md*
