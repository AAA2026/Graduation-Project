# 📥 Vigil - Download and Setup Instructions

## Complete Guide to Download, Convert, and Run Vigil

This is your complete guide to getting the Vigil surveillance system running as a JavaScript-based React application.

---

## 🎯 Overview

Vigil is being converted from TypeScript to JavaScript to make it easier to:
- Download and use as a frontend
- Edit and configure
- Add new features
- Integrate with your backend

---

## 📋 Prerequisites

Before you begin, ensure you have:

1. **Node.js** installed (version 16 or higher)
   - Download from: https://nodejs.org/
   - Verify: Open terminal and run `node --version`

2. **npm** (comes with Node.js)
   - Verify: Run `npm --version`

3. **Text Editor** (VS Code recommended)
   - Download from: https://code.visualstudio.com/

---

## 🚀 Step-by-Step Setup

### Step 1: Download the Project

Download the complete Vigil project folder to your computer.

If you received a ZIP file:
```bash
# Extract it to your desired location
# Navigate to the extracted folder
cd vigil-surveillance-system
```

### Step 2: Convert TypeScript to JavaScript

We've included an automatic conversion script:

```bash
node convert-to-jsx.js
```

**What this does**:
- Converts all `.tsx` files to `.jsx`
- Converts all `.ts` files to `.js`
- Removes TypeScript type annotations
- Updates import paths
- Copies styles and assets
- Creates the `/src` directory with all converted files

**Expected output**:
```
🚀 Starting TypeScript to JavaScript conversion...

📁 Processing directory: ./components
✅ Converted: components/LoginScreen.tsx -> src/components/LoginScreen.jsx
✅ Converted: components/ThemeProvider.tsx -> src/components/ThemeProvider.jsx
...

📊 Conversion Summary:
✅ Files converted: 85
⏭️  Files skipped: 2
❌ Errors: 0

✨ Conversion completed successfully!
```

### Step 3: Install Dependencies

```bash
npm install
```

This will install all required packages (React, Tailwind, icons, charts, etc.)

**Wait time**: 2-5 minutes depending on your internet speed

### Step 4: Start Development Server

```bash
npm start
```

**Expected output**:
```
<i> [webpack-dev-server] Project is running at:
<i> [webpack-dev-server] Loopback: http://localhost:3000/
✔ Compiled successfully
```

Your browser should automatically open to `http://localhost:3000`

### Step 5: Test the Application

1. You should see the Vigil login screen
2. Enter any email and password (it's a demo)
3. Select a role (Admin, Officer, or Security Authority)
4. Click "Sign In"
5. You should see the dashboard with camera feeds!

---

## 📁 Project Structure (After Setup)

```
vigil-surveillance-system/
├── public/
│   └── index.html              # HTML template
│
├── src/                        # ✨ All converted JavaScript files
│   ├── index.js               # Application entry point
│   ├── App.jsx                # Main component
│   │
│   ├── components/            # React components
│   │   ├── ui/               # UI components (buttons, cards, etc.)
│   │   ├── LoginScreen.jsx
│   │   ├── ModernSecurityLayout.jsx
│   │   ├── AdminDashboard.jsx
│   │   ├── OfficerDashboard.jsx
│   │   ├── DVRCameraGrid.jsx
│   │   ├── ThemeProvider.jsx
│   │   └── ... (many more)
│   │
│   ├── hooks/                 # Custom React hooks
│   │   ├── useLiveStatus.js
│   │   └── useRealtimeIncidents.js
│   │
│   ├── utils/                 # Utility functions
│   │   └── exportUtils.js
│   │
│   └── styles/                # CSS files
│       ├── globals.css
│       └── animated-background.css
│
├── node_modules/               # Installed packages (auto-generated)
├── dist/                       # Production build (after npm run build)
│
├── webpack.config.js           # Build configuration
├── package.json               # Dependencies and scripts
├── .babelrc                   # JavaScript compiler config
├── postcss.config.js          # CSS processing config
├── .gitignore                 # Git ignore rules
│
├── convert-to-jsx.js          # Conversion script
│
└── Documentation/
    ├── README.md              # Main documentation
    ├── GETTING_STARTED.md     # Quick start guide
    ├── SETUP_GUIDE.md         # Detailed setup
    └── CONVERSION_GUIDE.md    # Conversion reference
```

---

## 🎨 Customization

### Change Colors

Edit `/src/styles/globals.css`:

```css
:root {
  --background: 222 47% 11%;     /* Dark background */
  --primary: 186 100% 50%;       /* Cyan accent */
  --accent: 38 100% 50%;         /* Amber accent */
}
```

### Change Logo

Edit `/src/components/VigilLogo.jsx` or replace with your own logo component.

### Add/Remove Cameras

Edit `/src/components/DVRCameraGrid.jsx`:

```javascript
const mockCameras = [
  { id: 1, name: 'Entrance', location: 'Main Gate' },
  { id: 2, name: 'Parking', location: 'Lot A' },
  // Add or remove cameras here
];
```

### Modify Incident Types

Edit `/src/hooks/useRealtimeIncidents.js`:

```javascript
const incidentTypes = ['Violence', 'Car Crash', 'Suspicious Activity'];
```

---

## 🔌 Backend Integration

The app currently uses mock data. To connect to your backend:

### 1. Create Environment File

Create `.env` in the project root:

```env
REACT_APP_API_URL=http://localhost:5000/api
REACT_APP_WS_URL=ws://localhost:5000
```

### 2. Replace Mock Data with API Calls

Example in any component:

```javascript
// OLD (mock):
const [incidents, setIncidents] = useState(mockIncidents);

// NEW (real API):
const [incidents, setIncidents] = useState([]);

useEffect(() => {
  fetch(`${process.env.REACT_APP_API_URL}/incidents`)
    .then(res => res.json())
    .then(data => setIncidents(data));
}, []);
```

### 3. Implement Required Endpoints

Your backend should provide:

- `POST /api/login` - User authentication
- `GET /api/live-status` - System status (polled every 3 seconds)
- `GET /api/incidents` - List incidents
- `GET /api/cameras` - Camera list
- `GET /api/users` - User management (admin only)
- `GET /api/reports` - Generate reports

---

## 🏗️ Building for Production

### Create Production Build

```bash
npm run build
```

This creates an optimized production build in the `/dist` folder.

### Test Production Build Locally

```bash
npm run serve
```

Opens the production build at `http://localhost:3000`

### Deploy to Hosting

Copy the `/dist` folder to your hosting provider:

**Option 1: Netlify (Easiest)**
1. Go to https://app.netlify.com/drop
2. Drag and drop the `/dist` folder
3. Your site is live!

**Option 2: Your Own Server**
```bash
# Copy dist folder to your web server
scp -r dist/* user@yourserver.com:/var/www/html/
```

---

## 📊 Available Scripts

```bash
# Start development server
npm start

# Build for production
npm run build

# Preview production build
npm run serve

# Clean build files and dependencies
npm run clean
```

---

## 🐛 Troubleshooting

### Conversion Script Fails

**Manually convert files** using `/CONVERSION_GUIDE.md`:
- Copy files from `/components` to `/src/components`
- Remove TypeScript syntax manually
- Update file extensions

### "Module not found" Errors

```bash
# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Port 3000 Already in Use

Edit `webpack.config.js`:
```javascript
devServer: {
  port: 3001, // Change to any available port
}
```

### Build Errors

```bash
# Clean and rebuild
rm -rf dist node_modules
npm install
npm run build
```

### Styles Not Loading

1. Make sure `/src/styles/globals.css` exists
2. Check `/src/index.js` imports it
3. Restart dev server

---

## ✅ Verification Checklist

After setup, verify:

- [ ] Conversion script ran successfully
- [ ] `/src` directory exists with all .jsx files
- [ ] `npm install` completed without errors
- [ ] `npm start` opens the app
- [ ] Login screen displays correctly
- [ ] Can login with any credentials
- [ ] Dashboard shows camera feeds
- [ ] Theme toggle works (light/dark)
- [ ] Incident notifications appear
- [ ] No errors in browser console

---

## 🎓 What You've Built

### Features Included

✅ **Three Role-Based Dashboards**
- Admin - Full system management
- Officer - Incident monitoring
- Security Authority - Mobile-optimized overview

✅ **DVR-Style Camera Grid**
- Live camera feeds
- Grid layout (4-6 cameras)
- Status indicators

✅ **Real-Time Incident System**
- Auto-generated incidents (demo)
- Toast notifications with sound
- Incident detail modals
- Video playback

✅ **Theme System**
- Dark mode (default)
- Light mode
- Smooth transitions
- Persistent preferences

✅ **Professional UI**
- Glassmorphism effects
- Animated backgrounds
- Responsive design
- Modern typography

✅ **Ready for Production**
- Optimized build process
- Environment variable support
- API integration ready
- Export functionality (PDF, Excel)

---

## 📚 Next Steps

1. **Explore the Code**
   - Browse `/src/components`
   - Read component comments
   - Understand the structure

2. **Customize Branding**
   - Update colors in globals.css
   - Replace logo
   - Modify text and labels

3. **Connect Backend**
   - Implement API endpoints
   - Replace mock data
   - Add authentication

4. **Add Features**
   - Create new components
   - Add new pages
   - Extend functionality

5. **Deploy**
   - Build for production
   - Deploy to hosting
   - Set up domain

---

## 📞 Documentation Reference

- **README.md** - Complete project documentation
- **GETTING_STARTED.md** - Quick start guide
- **SETUP_GUIDE.md** - Detailed configuration
- **CONVERSION_GUIDE.md** - TS to JS conversion details

---

## 🎉 Success!

You now have a fully functional, JavaScript-based surveillance system ready for customization and deployment!

**Key Advantages**:
- ✅ Pure JavaScript (no TypeScript)
- ✅ Easy to understand and modify
- ✅ Simple webpack build
- ✅ Ready for backend integration
- ✅ Production-ready

---

**Need help?** Check the documentation or review code comments.

**Ready to build?** Start customizing and integrating with your backend!

🚀 **Happy coding!**
