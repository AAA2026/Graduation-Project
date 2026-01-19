# 🚀 Vigil Deployment Checklist

Complete checklist for deploying Vigil to production.

---

## 📋 Pre-Deployment Checklist

### Development Complete
- [ ] All features tested and working
- [ ] Backend API connected and tested
- [ ] Authentication implemented
- [ ] Real camera streams integrated (if applicable)
- [ ] All mock data replaced with real data
- [ ] Error handling implemented
- [ ] Loading states added where needed

### Code Quality
- [ ] Console.log() statements removed
- [ ] Debug code removed
- [ ] Code commented appropriately
- [ ] No hardcoded credentials
- [ ] Environment variables configured
- [ ] Unused components removed
- [ ] Unused dependencies removed

### Security
- [ ] API keys in environment variables
- [ ] CORS configured correctly
- [ ] Authentication secured
- [ ] Input validation on all forms
- [ ] XSS protection implemented
- [ ] CSRF tokens where needed
- [ ] HTTPS enforced (for production)
- [ ] Security headers configured

### Performance
- [ ] Images optimized
- [ ] Code splitting implemented (if needed)
- [ ] Lazy loading for heavy components
- [ ] Bundle size checked and optimized
- [ ] API calls optimized
- [ ] Caching strategy implemented

### Testing
- [ ] All features tested on Chrome
- [ ] All features tested on Firefox
- [ ] All features tested on Safari
- [ ] Mobile responsiveness tested
- [ ] All three roles tested
- [ ] Error scenarios tested
- [ ] Backend integration tested
- [ ] Production build tested locally

---

## 🔧 Build Configuration

### 1. Environment Variables

Create `.env.production`:

```env
# API Configuration
REACT_APP_API_URL=https://api.yourdomain.com
REACT_APP_WS_URL=wss://api.yourdomain.com

# Feature Flags
REACT_APP_ENABLE_ANALYTICS=true
REACT_APP_ENABLE_SOUND_ALERTS=true

# Polling Intervals (milliseconds)
REACT_APP_STATUS_POLL_INTERVAL=3000
REACT_APP_INCIDENT_POLL_INTERVAL=5000

# Environment
REACT_APP_ENV=production
```

### 2. Verify webpack.config.js

Check production optimizations:

```javascript
module.exports = {
  // ...
  mode: 'production',
  optimization: {
    splitChunks: {
      chunks: 'all',
    },
  },
};
```

### 3. Build for Production

```bash
# Clean previous builds
rm -rf dist

# Build
npm run build

# Verify build size
du -sh dist/*
```

**Expected output**:
```
Creating an optimized production build...
✔ Compiled successfully.

File sizes after gzip:
  250.5 kB  dist/bundle.abc123.js
  15.2 kB   dist/styles.abc123.css
```

### 4. Test Production Build Locally

```bash
npm run serve
```

Test thoroughly:
- [ ] Login works
- [ ] All pages load
- [ ] No console errors
- [ ] All features functional
- [ ] Theme toggle works
- [ ] API calls successful

---

## 🌐 Deployment Options

### Option 1: Netlify (Recommended for Frontend)

#### Setup:
1. Create account at https://www.netlify.com/
2. Install Netlify CLI:
   ```bash
   npm install -g netlify-cli
   ```

#### Deploy:
```bash
# Login
netlify login

# Deploy
netlify deploy --prod --dir=dist
```

#### Or Use Drag & Drop:
1. Go to https://app.netlify.com/drop
2. Drag `/dist` folder
3. Site is live!

#### Configure:
- [ ] Custom domain set up
- [ ] HTTPS enabled (automatic)
- [ ] Environment variables added
- [ ] Redirects configured (_redirects file)

**Create** `/public/_redirects`:
```
/*    /index.html   200
```

---

### Option 2: Vercel

#### Setup:
```bash
npm install -g vercel
```

#### Deploy:
```bash
vercel login
vercel --prod
```

#### Configure:
- [ ] Custom domain
- [ ] Environment variables
- [ ] Auto-deployments from Git

---

### Option 3: AWS S3 + CloudFront

#### Setup:
1. Create S3 bucket
2. Enable static website hosting
3. Set bucket policy for public read
4. Create CloudFront distribution

#### Deploy:
```bash
# Install AWS CLI
aws configure

# Sync to S3
aws s3 sync dist/ s3://your-bucket-name --delete

# Invalidate CloudFront cache
aws cloudfront create-invalidation --distribution-id YOUR_ID --paths "/*"
```

#### Configure:
- [ ] S3 bucket created
- [ ] Static hosting enabled
- [ ] CloudFront distribution created
- [ ] Custom domain (Route 53)
- [ ] SSL certificate (ACM)

---

### Option 4: Your Own Server (Apache/Nginx)

#### Apache Configuration

Create `/etc/apache2/sites-available/vigil.conf`:
```apache
<VirtualHost *:80>
    ServerName yourdomain.com
    DocumentRoot /var/www/html/vigil
    
    <Directory /var/www/html/vigil>
        Options -Indexes +FollowSymLinks
        AllowOverride All
        Require all granted
        
        # SPA routing
        RewriteEngine On
        RewriteBase /
        RewriteRule ^index\.html$ - [L]
        RewriteCond %{REQUEST_FILENAME} !-f
        RewriteCond %{REQUEST_FILENAME} !-d
        RewriteRule . /index.html [L]
    </Directory>
</VirtualHost>
```

#### Nginx Configuration

Create `/etc/nginx/sites-available/vigil`:
```nginx
server {
    listen 80;
    server_name yourdomain.com;
    root /var/www/html/vigil;
    index index.html;

    # SPA routing
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
}
```

#### Deploy:
```bash
# Build locally
npm run build

# Copy to server
scp -r dist/* user@yourserver:/var/www/html/vigil/

# Or use rsync
rsync -avz --delete dist/ user@yourserver:/var/www/html/vigil/
```

---

## 🔒 Security Setup

### 1. HTTPS/SSL

#### Netlify/Vercel:
- Automatic (nothing to do!)

#### Your Own Server:
```bash
# Install Certbot
sudo apt install certbot python3-certbot-apache

# Get certificate
sudo certbot --apache -d yourdomain.com

# Auto-renewal
sudo systemctl enable certbot.timer
```

### 2. Security Headers

Add to your server config:

**Nginx**:
```nginx
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "no-referrer-when-downgrade" always;
add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
```

**Apache**:
```apache
Header always set X-Frame-Options "SAMEORIGIN"
Header always set X-Content-Type-Options "nosniff"
Header always set X-XSS-Protection "1; mode=block"
Header always set Referrer-Policy "no-referrer-when-downgrade"
```

### 3. Environment Variables

**Never commit**:
- API keys
- Database credentials
- Secret tokens
- Private configuration

**Always use**:
- Environment variables
- Secrets management (AWS Secrets Manager, etc.)
- `.env.production` (gitignored)

---

## 📊 Post-Deployment

### 1. Monitoring

#### Set up:
- [ ] Error tracking (Sentry, LogRocket)
- [ ] Analytics (Google Analytics, Plausible)
- [ ] Uptime monitoring (UptimeRobot, Pingdom)
- [ ] Performance monitoring (Lighthouse CI)

#### Sentry Setup:
```bash
npm install @sentry/react

# In src/index.js
import * as Sentry from '@sentry/react';

Sentry.init({
  dsn: process.env.REACT_APP_SENTRY_DSN,
  environment: process.env.REACT_APP_ENV,
});
```

### 2. Analytics

#### Google Analytics:
```html
<!-- In public/index.html -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

### 3. Performance

#### Check:
```bash
# Using Lighthouse
npx lighthouse https://yourdomain.com --view

# Or use https://web.dev/measure/
```

#### Optimize if needed:
- [ ] Image compression
- [ ] Code splitting
- [ ] Lazy loading
- [ ] CDN for static assets
- [ ] Browser caching
- [ ] Gzip/Brotli compression

---

## ✅ Go-Live Checklist

### Final Verification
- [ ] Production build created
- [ ] All environment variables set
- [ ] API endpoints configured
- [ ] HTTPS enabled
- [ ] Custom domain configured
- [ ] DNS records updated
- [ ] Security headers set
- [ ] Error tracking configured
- [ ] Analytics set up
- [ ] Backups configured

### Testing on Production
- [ ] Homepage loads
- [ ] Login works
- [ ] All features functional
- [ ] No console errors
- [ ] No 404 errors
- [ ] Mobile responsive
- [ ] Performance acceptable
- [ ] All roles tested

### Documentation
- [ ] Deployment documented
- [ ] Environment variables documented
- [ ] API endpoints documented
- [ ] Monitoring access shared
- [ ] Credentials stored securely

---

## 🔄 Continuous Deployment

### Git-Based Deployment

#### Netlify:
1. Connect GitHub/GitLab
2. Configure build command: `npm run build`
3. Set publish directory: `dist`
4. Add environment variables
5. Auto-deploy on push to main

#### Vercel:
1. `vercel link`
2. `git push` auto-deploys

### GitHub Actions

Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Node
        uses: actions/setup-node@v2
        with:
          node-version: '16'
      
      - name: Install dependencies
        run: npm install
      
      - name: Build
        run: npm run build
      
      - name: Deploy to Netlify
        run: npx netlify-cli deploy --prod --dir=dist
        env:
          NETLIFY_AUTH_TOKEN: ${{ secrets.NETLIFY_AUTH_TOKEN }}
          NETLIFY_SITE_ID: ${{ secrets.NETLIFY_SITE_ID }}
```

---

## 📱 Mobile Considerations

### PWA (Progressive Web App)

Add to `public/index.html`:
```html
<link rel="manifest" href="/manifest.json">
<meta name="theme-color" content="#0a0f1e">
```

Create `public/manifest.json`:
```json
{
  "name": "Vigil Surveillance System",
  "short_name": "Vigil",
  "description": "AI-Powered Smart Surveillance",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#0a0f1e",
  "theme_color": "#0a0f1e",
  "icons": [
    {
      "src": "/icon-192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "/icon-512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ]
}
```

---

## 🎉 Launch Day!

### Before Launch
- [ ] Notify team
- [ ] Prepare support channels
- [ ] Have rollback plan ready
- [ ] Monitor logs/errors
- [ ] Be ready for issues

### Launch
- [ ] Deploy to production
- [ ] Verify deployment successful
- [ ] Test critical paths
- [ ] Monitor error rates
- [ ] Watch performance metrics

### After Launch
- [ ] Monitor for 24 hours
- [ ] Address any issues quickly
- [ ] Collect user feedback
- [ ] Document any issues
- [ ] Plan next iteration

---

## 🆘 Rollback Plan

### Quick Rollback

**Netlify/Vercel**:
- Go to dashboard
- Select previous deployment
- Click "Publish"

**Your Server**:
```bash
# Keep previous builds
mv dist dist.backup
cp -r dist.backup dist

# Or use Git tags
git checkout v1.0.0
npm run build
deploy
```

---

## 📞 Support After Deployment

### Monitoring
- Check error tracking daily
- Review analytics weekly
- Monitor uptime 24/7
- Watch performance metrics

### Updates
- Security patches: Immediately
- Bug fixes: Within 24-48 hours
- New features: Regular releases
- Dependencies: Monthly updates

---

<div align="center">

# 🚀 Ready to Deploy!

Follow this checklist step by step for a successful deployment.

---

**Good luck with your launch!** 🎉

</div>
