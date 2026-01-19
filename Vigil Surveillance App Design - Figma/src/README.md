# Vigil - AI-Powered Smart Surveillance System

A modern, real-time AI-powered surveillance system designed for detecting violence and car crashes with role-based interfaces for Admin, Officer, and Security Authority users.

## 🚀 Features

### Core Features
- **Real-time Incident Detection**: Live monitoring with AI-powered violence and car crash detection
- **Multi-Role Dashboard**: Separate interfaces for Admin, Officer, and Security Authority
- **DVR-Style Camera Grid**: Front-and-center camera feeds like a true surveillance control center
- **Live Status Monitoring**: Real-time backend polling for system status
- **Incident Management**: Comprehensive incident tracking, reporting, and analytics
- **Theme Support**: Full light/dark mode with glassmorphism effects

### Admin Dashboard
- User management with role-based access control
- Camera management and configuration
- AI model management and settings
- Comprehensive analytics and reports
- Demo request management
- System health monitoring

### Officer Dashboard
- Live camera feeds with DVR-style grid layout
- Real-time incident alerts with toast notifications
- Quick action buttons for incident response
- Video playback with download functionality
- Incident detail modals with full information

### Security Authority Features
- Mobile-optimized interface
- Push notification system for new incidents
- Interactive map with camera locations
- Quick status updates
- Incident overview and management

## 📦 Quick Start

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn package manager

### Installation

1. **Download and Extract** the project files to your local machine

2. **Install Dependencies**
   ```bash
   npm install
   ```
   or if you prefer yarn:
   ```bash
   yarn install
   ```

3. **Start Development Server**
   ```bash
   npm start
   ```
   This will start the app on `http://localhost:3000`

4. **Build for Production**
   ```bash
   npm run build
   ```
   Production files will be in the `dist/` directory

5. **Preview Production Build** (optional)
   ```bash
   npm run serve
   ```

## 🏗️ Project Structure

```
vigil-surveillance-system/
├── public/                     # Static files
│   ├── index.html             # HTML template
│   └── favicon.ico            # App icon
├── src/                       # Source code
│   ├── components/            # React components
│   │   ├── ui/               # UI components (buttons, cards, etc.)
│   │   ├── dashboard/        # Dashboard-specific components
│   │   ├── hooks/            # Custom hooks (moved here)
│   │   ├── LoginScreen.jsx   # Login interface
│   │   ├── ModernSecurityLayout.jsx  # Main layout
│   │   ├── AdminDashboard.jsx
│   │   ├── OfficerDashboard.jsx
│   │   ├── SecurityAuthorityApp.jsx
│   │   └── ... (other components)
│   ├── hooks/                 # Custom React hooks
│   │   ├── useLiveStatus.js  # Live status polling
│   │   └── useRealtimeIncidents.js  # Incident simulation
│   ├── utils/                 # Utility functions
│   │   └── exportUtils.js    # PDF/Excel export utilities
│   ├── styles/               # CSS files
│   │   ├── globals.css       # Global styles & Tailwind
│   │   └── animated-background.css  # Background animations
│   ├── App.jsx               # Main App component
│   └── index.js              # Entry point
├── webpack.config.js          # Webpack configuration
├── package.json              # Dependencies and scripts
├── .babelrc                  # Babel configuration
├── postcss.config.js         # PostCSS configuration
└── README.md                 # This file
```

## 🎨 Customization

### Theme Colors
Edit `/src/styles/globals.css` to customize colors:
```css
:root {
  --background: 222 47% 11%;    /* Deep charcoal */
  --primary: 186 100% 50%;      /* Cyan accent */
  --accent: 38 100% 50%;        /* Amber accent */
  /* ... other colors */
}
```

### Logo
- Logo component: `/src/components/VigilLogo.jsx`
- Standalone logo: `/src/components/VigilLogoStandalone.jsx`

### Camera Configuration
Edit mock camera data in:
- `/src/components/DVRCameraGrid.jsx`
- `/src/components/LiveCameraGrid.jsx`
- `/src/components/MockCameraFeed.jsx`

## 🔌 Backend Integration

### API Endpoints to Implement

The app is ready for backend integration. Replace mock data with real API calls:

1. **Authentication**
   - `POST /api/login` - User login
   - `POST /api/logout` - User logout
   - `GET /api/user` - Get current user

2. **Live Status**
   - `GET /api/live-status` - System status (polled every 3 seconds)

3. **Incidents**
   - `GET /api/incidents` - List incidents
   - `GET /api/incidents/:id` - Get incident details
   - `POST /api/incidents/:id/resolve` - Resolve incident
   - `GET /api/incidents/stats` - Get incident statistics

4. **Cameras**
   - `GET /api/cameras` - List cameras
   - `POST /api/cameras` - Add camera
   - `PUT /api/cameras/:id` - Update camera
   - `DELETE /api/cameras/:id` - Delete camera
   - `GET /api/cameras/:id/stream` - Camera stream URL

5. **Users**
   - `GET /api/users` - List users
   - `POST /api/users` - Create user
   - `PUT /api/users/:id` - Update user
   - `DELETE /api/users/:id` - Delete user

6. **Reports**
   - `GET /api/reports` - Get reports
   - `POST /api/reports/generate` - Generate report

7. **Demo Requests**
   - `GET /api/demo-requests` - List demo requests
   - `POST /api/demo-requests/:id/approve` - Approve request

### Where to Add API Calls

Look for comments like:
```javascript
// TODO: Replace with real API call
// const response = await fetch('/api/endpoint');
```

Key files for API integration:
- `/src/hooks/useLiveStatus.js` - Live status polling
- `/src/hooks/useRealtimeIncidents.js` - Incident simulation
- `/src/components/*Dashboard.jsx` - Dashboard components

## 🛠️ Adding New Features

### Adding a New Component

1. Create file in `/src/components/YourComponent.jsx`
2. Import and use in parent component
3. Follow existing patterns for props and state

Example:
```javascript
// /src/components/MyNewFeature.jsx
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';

export function MyNewFeature({ data }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>My Feature</CardTitle>
      </CardHeader>
      <CardContent>
        {/* Your content */}
      </CardContent>
    </Card>
  );
}
```

### Adding a New Route/View

1. Add view to the `ModernSecurityLayout.jsx` component
2. Add navigation tab
3. Add view content in the switch statement

### Adding New Icons

Use lucide-react icons:
```javascript
import { Camera, AlertTriangle, Settings } from 'lucide-react';

<Camera className="h-4 w-4" />
```

## 📱 Deployment

### Build for Production
```bash
npm run build
```

### Deploy to Static Hosting
The `dist/` folder can be deployed to:
- Netlify
- Vercel
- GitHub Pages
- AWS S3 + CloudFront
- Any static hosting service

### Environment Variables
Create `.env` file for environment-specific configuration:
```
REACT_APP_API_URL=https://api.your-domain.com
REACT_APP_WS_URL=wss://api.your-domain.com
```

Access in code:
```javascript
const API_URL = process.env.REACT_APP_API_URL;
```

## 🎯 Login Credentials (Demo)

The app accepts any username/password for demonstration:
- **Admin**: Select "Admin" role
- **Officer**: Select "Officer" role  
- **Security Authority**: Select "Security Authority" role

## 🔧 Configuration Files

- **webpack.config.js**: Build configuration
- **.babelrc**: JavaScript transpilation
- **postcss.config.js**: CSS processing (Tailwind)
- **package.json**: Dependencies and scripts

## 📚 Documentation

Additional documentation in `/guidelines/` folder:
- `ARCHITECTURE.md` - System architecture
- `THEME_SYSTEM.md` - Theme customization
- `COMPONENT_SPECIFICATIONS.md` - Component details
- `VIGIL_LOGO_README.md` - Logo usage guide

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## 📄 License

MIT License - feel free to use for personal or commercial projects

## 🆘 Troubleshooting

### Port Already in Use
Change port in `webpack.config.js`:
```javascript
devServer: {
  port: 3001, // Change this
}
```

### Build Errors
1. Delete `node_modules` and `package-lock.json`
2. Run `npm install` again
3. Clear cache: `npm cache clean --force`

### Styling Issues
1. Check Tailwind classes are correct
2. Verify PostCSS is processing correctly
3. Check browser console for CSS errors

## 📞 Support

For issues or questions:
1. Check the documentation in `/guidelines/`
2. Review code comments
3. Check console for errors

---

**Built with ❤️ using React, Tailwind CSS, and modern web technologies**
