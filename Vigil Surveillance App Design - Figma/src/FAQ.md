# ❓ Vigil - Frequently Asked Questions (FAQ)

Common questions and answers about Vigil surveillance system.

---

## 🚀 Getting Started

### Q: Where do I start?

**A**: Open [START_HERE.md](./START_HERE.md) - it guides you through everything!

Quick version:
1. Run `node convert-to-jsx.js`
2. Run `npm install`
3. Run `npm start`
4. Login with any credentials

---

### Q: What are the system requirements?

**A**: You need:
- Node.js v16 or higher
- npm (comes with Node.js)
- 2GB RAM minimum
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Text editor (VS Code recommended)

---

### Q: Do I need to know TypeScript?

**A**: No! The entire system converts to plain JavaScript. No TypeScript knowledge needed.

---

### Q: How long does setup take?

**A**: 
- Conversion: 10-30 seconds
- Installation: 2-5 minutes
- First run: 30 seconds
- **Total: ~10-15 minutes**

---

## 🔄 Conversion

### Q: What does the conversion script do?

**A**: It automatically:
- Converts all `.tsx` files to `.jsx`
- Converts all `.ts` files to `.js`
- Removes TypeScript type annotations
- Removes interfaces and type definitions
- Updates import paths
- Copies CSS files
- Creates `/src` directory with converted files

---

### Q: Do I need to run the conversion script?

**A**: Yes! It converts TypeScript to JavaScript. Without it, you're still using TypeScript.

---

### Q: What if the conversion script fails?

**A**: 
1. Check the error message
2. Ensure you're in the project root directory
3. Check Node.js is installed (`node --version`)
4. Try manual conversion using [CONVERSION_GUIDE.md](./CONVERSION_GUIDE.md)

---

### Q: Can I still use TypeScript if I want?

**A**: Yes, but you'd need to:
- Keep the original files
- Don't run the conversion script
- Use Vite instead of Webpack
- Keep the TypeScript dependencies

(Not recommended - defeats the purpose of this setup)

---

## 🛠️ Development

### Q: What port does the app run on?

**A**: Port 3000 by default.

Change it in `webpack.config.js`:
```javascript
devServer: {
  port: 3001, // Your custom port
}
```

---

### Q: How do I stop the development server?

**A**: Press `Ctrl + C` in the terminal.

---

### Q: Do I need to restart after every change?

**A**: No! Webpack has hot reload - save the file and the browser auto-refreshes.

**Exception**: Changes to `webpack.config.js` or `.env` require restart.

---

### Q: Where do I add my custom code?

**A**: In the `/src` directory:
- Components: `/src/components/`
- Hooks: `/src/hooks/`
- Utilities: `/src/utils/`
- Styles: `/src/styles/`

---

### Q: Can I delete the original TypeScript files?

**A**: Yes, after conversion is successful and tested:
- Delete `/components` (original)
- Delete `/hooks` (original)
- Delete `/utils` (original)
- Keep `/src` (converted)

---

## 🎨 Customization

### Q: How do I change the theme colors?

**A**: Edit `/src/styles/globals.css`:
```css
:root {
  --primary: 186 100% 50%;    /* Your color */
  --accent: 38 100% 50%;      /* Your color */
}
```

Restart the dev server to see changes.

---

### Q: How do I change the logo?

**A**: Edit `/src/components/VigilLogo.jsx` or create your own logo component.

---

### Q: Can I change the number of cameras displayed?

**A**: Yes! Edit `/src/components/DVRCameraGrid.jsx`:
```javascript
const mockCameras = [
  // Add or remove cameras here
];
```

For grid layout:
```javascript
// 4 cameras (2x2)
<div className="grid grid-cols-2 gap-4">

// 6 cameras (3x2)
<div className="grid grid-cols-3 gap-4">
```

---

### Q: How do I disable the incident simulation?

**A**: Edit `/src/hooks/useRealtimeIncidents.js`:
```javascript
// Comment out this useEffect:
// useEffect(() => {
//   simulateIncident();
// }, []);
```

---

## 🔌 Backend Integration

### Q: Does Vigil include a backend?

**A**: No, Vigil is a frontend application. You need to build your own backend API or integrate with existing services.

---

### Q: What backend should I use?

**A**: Any! Popular choices:
- **Node.js + Express** (JavaScript)
- **Python + Flask/Django**
- **PHP + Laravel**
- **Ruby on Rails**
- **Go + Gin**

Choose what you're comfortable with.

---

### Q: What API endpoints do I need to implement?

**A**: See [README.md](./README.md) - Backend Integration section.

Key endpoints:
- `POST /api/login` - Authentication
- `GET /api/live-status` - System status
- `GET /api/incidents` - Incidents list
- `GET /api/cameras` - Camera list

---

### Q: How do I connect to my backend?

**A**: 
1. Create `.env` file:
   ```env
   REACT_APP_API_URL=http://localhost:5000/api
   ```

2. Replace mock data with fetch calls:
   ```javascript
   const response = await fetch(`${process.env.REACT_APP_API_URL}/incidents`);
   const data = await response.json();
   ```

See [SETUP_GUIDE.md](./SETUP_GUIDE.md) for details.

---

### Q: Can I use Firebase/Supabase instead of custom backend?

**A**: Yes! You can integrate any backend service:
- Firebase
- Supabase
- AWS Amplify
- Custom API

Just replace the API calls in the frontend.

---

## 📹 Camera Streams

### Q: Does Vigil work with real cameras?

**A**: Currently uses simulated feeds. You need to integrate real camera streams.

---

### Q: What camera protocols are supported?

**A**: You can integrate:
- RTSP streams
- HLS streams
- WebRTC
- MJPEG
- IP camera snapshots

See [WHATS_NEXT.md](./WHATS_NEXT.md) - Phase 3 for integration guide.

---

### Q: How do I add real camera streams?

**A**: Replace `/src/components/MockCameraFeed.jsx` with:
```javascript
// For HLS
<video src={streamUrl} autoPlay muted />

// For MJPEG/snapshots
<img src={snapshotUrl} />
```

Requires backend support for stream URLs.

---

## 🚀 Deployment

### Q: How do I deploy to production?

**A**: 
1. Build: `npm run build`
2. Deploy `/dist` folder to:
   - Netlify (drag & drop)
   - Vercel (`vercel deploy`)
   - AWS S3
   - Your own server

See [DEPLOYMENT_CHECKLIST.md](./DEPLOYMENT_CHECKLIST.md) for complete guide.

---

### Q: What's the recommended hosting?

**A**: For frontend only:
- **Netlify** - Easiest (free tier)
- **Vercel** - Great for React (free tier)
- **AWS S3 + CloudFront** - Scalable
- **Your server** - Full control

---

### Q: Do I need HTTPS?

**A**: Yes, for production. Most hosting providers (Netlify, Vercel) provide free SSL.

For your own server, use Let's Encrypt (free).

---

### Q: How do I set up a custom domain?

**A**: 
1. Buy domain (GoDaddy, Namecheap, etc.)
2. Point DNS to your hosting
3. Configure in hosting settings

Each host has specific instructions.

---

## 🐛 Troubleshooting

### Q: "Module not found" error

**A**: 
```bash
rm -rf node_modules package-lock.json
npm install
```

---

### Q: "Port 3000 already in use"

**A**: Either:
- Kill the process using port 3000
- Change port in `webpack.config.js` to 3001 or another port

---

### Q: Styles not loading

**A**: 
1. Check `/src/index.js` imports CSS files
2. Verify PostCSS config exists
3. Restart dev server
4. Hard refresh browser (Ctrl+Shift+R)

---

### Q: Build fails

**A**:
```bash
rm -rf dist node_modules
npm install
npm run build
```

---

### Q: Theme toggle not working

**A**: 
1. Check ThemeProvider wraps the app
2. Verify localStorage is enabled
3. Check browser console for errors

---

### Q: Hot reload not working

**A**:
1. Restart dev server
2. Check webpack.config.js has `hot: true`
3. Clear browser cache
4. Try hard refresh

---

## 💡 Features

### Q: What roles are included?

**A**: Three roles:
- **Admin** - Full system management
- **Officer** - Incident monitoring and response
- **Security Authority** - Mobile-optimized overview

---

### Q: Can I add more roles?

**A**: Yes! Edit `/src/App.jsx` and create new dashboard components.

---

### Q: Does it have real-time updates?

**A**: 
- **Current**: Simulated real-time (demo purposes)
- **Production**: Requires WebSocket integration with your backend

---

### Q: Can I export reports?

**A**: Yes! Built-in PDF and Excel export using:
- jsPDF for PDF
- Excel export libraries
- html2canvas for screenshots

---

### Q: Does it work on mobile?

**A**: Yes! Fully responsive design. Security Authority dashboard is optimized for mobile.

---

### Q: Can I disable dark mode?

**A**: Yes, remove the theme toggle and set default theme:

In `/src/components/ThemeProvider.jsx`:
```javascript
const [theme, setTheme] = useState('light'); // Force light mode
```

---

## 📊 Performance

### Q: What's the bundle size?

**A**: After production build (gzipped):
- JavaScript: ~250KB
- CSS: ~15KB
- **Total: ~265KB**

Very reasonable for a full-featured app!

---

### Q: How can I optimize performance?

**A**:
1. Code splitting (lazy loading)
2. Image optimization
3. Use CDN for static assets
4. Enable caching
5. Compress responses (gzip/brotli)

---

### Q: Does it support PWA?

**A**: Not out of the box, but you can add:
1. Service workers
2. Manifest file
3. Offline support

See [WHATS_NEXT.md](./WHATS_NEXT.md) for guide.

---

## 🔐 Security

### Q: Is it secure?

**A**: Frontend is just UI. Security depends on your backend:
- Implement proper authentication
- Use HTTPS
- Validate inputs
- Enable CORS properly
- Use security headers

---

### Q: How do I implement authentication?

**A**: 
1. Backend: Implement JWT or session-based auth
2. Frontend: Store token, add to API requests
3. Protect routes based on user role

See [SETUP_GUIDE.md](./SETUP_GUIDE.md) - Backend Integration.

---

### Q: Can I use OAuth/Social Login?

**A**: Yes! Integrate any authentication provider:
- Google OAuth
- Facebook Login
- Auth0
- Okta

Update LoginScreen.jsx accordingly.

---

## 📚 Documentation

### Q: Where's all the documentation?

**A**: See [DOCUMENTATION_INDEX.md](./DOCUMENTATION_INDEX.md) for complete list.

Key docs:
- [START_HERE.md](./START_HERE.md) - Start here
- [README.md](./README.md) - Complete docs
- [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) - Quick tips

---

### Q: Is there a video tutorial?

**A**: Not currently, but documentation is comprehensive with code examples and step-by-step guides.

---

### Q: Can I contribute to documentation?

**A**: Yes! If you improve it:
1. Document your changes
2. Share with the community
3. Create pull request (if using Git)

---

## 🆘 Support

### Q: Where do I get help?

**A**:
1. Read appropriate documentation
2. Check browser console for errors
3. Search Stack Overflow
4. Check React/Webpack documentation

---

### Q: Can I hire someone to customize it?

**A**: Yes! Look for:
- React developers
- Full-stack developers
- Freelancers on Upwork, Fiverr

---

### Q: Is commercial use allowed?

**A**: Check the license. Most likely MIT (permissive), but verify in the project.

---

## 💰 Cost

### Q: Is Vigil free?

**A**: Yes! Open source and free to use.

---

### Q: What costs should I expect?

**A**:
- **Hosting**: $0-50/month (Netlify/Vercel free tier available)
- **Backend**: $0-100/month (depends on your choice)
- **Domain**: $10-20/year
- **SSL**: Free (Let's Encrypt, included with Netlify/Vercel)
- **Database**: $0-50/month (depends on service)

**Total**: Can start completely free!

---

## 🎯 Use Cases

### Q: What can I use Vigil for?

**A**:
- Surveillance systems
- Security operations centers
- Camera monitoring
- Incident management
- Security dashboards
- Learning React

---

### Q: Can I use it for non-security purposes?

**A**: Yes! The architecture works for:
- Video monitoring (any type)
- Multi-camera management
- Alert systems
- Dashboard applications
- Admin panels

---

### Q: Is it production-ready?

**A**: The frontend is production-ready. You need to:
1. Add your backend
2. Integrate real cameras
3. Implement authentication
4. Add proper security
5. Test thoroughly

---

## 🔮 Future

### Q: Will there be updates?

**A**: The codebase is solid and ready to use. Future enhancements depend on the maintainer.

---

### Q: Can I add AI features?

**A**: Yes! Integrate AI models:
- YOLO for object detection
- Custom violence detection models
- Face recognition
- License plate recognition

Requires backend with AI/ML capabilities.

---

### Q: Mobile app version?

**A**: Not included, but you can:
1. Use as PWA (Progressive Web App)
2. Convert to React Native
3. Build separate mobile app

---

## ❓ Still Have Questions?

### Quick Answers:
- **Setup**: See [DOWNLOAD_AND_SETUP.md](./DOWNLOAD_AND_SETUP.md)
- **Customization**: See [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)
- **Backend**: See [SETUP_GUIDE.md](./SETUP_GUIDE.md)
- **Deployment**: See [DEPLOYMENT_CHECKLIST.md](./DEPLOYMENT_CHECKLIST.md)

### Need More Help?
1. Read the specific documentation
2. Check code comments
3. Search for error message online
4. Review React/Webpack docs

---

<div align="center">

## 💬 Can't Find Your Answer?

Check [DOCUMENTATION_INDEX.md](./DOCUMENTATION_INDEX.md) for all guides

Or start with [START_HERE.md](./START_HERE.md)

---

**Most questions are answered in the comprehensive documentation!** 📚

</div>
