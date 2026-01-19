# Pre-Download Checklist ✅

## Before You Download

Use this checklist to ensure your system is ready and the Vigil app will work perfectly when you download it.

---

## System Requirements

### ✅ Node.js Installation

- [ ] **Node.js 18+ installed**
  ```bash
  node --version
  # Should show: v18.x.x or higher
  ```

- [ ] **npm 8+ installed**
  ```bash
  npm --version
  # Should show: 8.x.x or higher
  ```

**If Not Installed:**
1. Download from https://nodejs.org/
2. Install LTS version (recommended)
3. Restart your computer after installation

---

### ✅ Code Editor

- [ ] **VS Code installed** (recommended)
  - Download: https://code.visualstudio.com/

**Alternative Editors:**
- WebStorm
- Sublime Text
- Atom
- Notepad++ (not recommended for React)

---

### ✅ Terminal Access

- [ ] **Know how to open terminal**
  - **Windows:** `Win+R` → type `cmd` → Enter
  - **Mac:** `Cmd+Space` → type `terminal` → Enter
  - **Linux:** `Ctrl+Alt+T`

- [ ] **Know how to navigate folders**
  ```bash
  cd folder-name        # Enter folder
  cd ..                 # Go up one level
  ls                    # List files (Mac/Linux)
  dir                   # List files (Windows)
  ```

---

### ✅ Browser

- [ ] **Modern browser installed**
  - Chrome 90+ ✅ (Recommended)
  - Firefox 88+
  - Safari 14+
  - Edge 90+

---

## Space & Performance Requirements

- [ ] **2 GB free disk space** (for project + node_modules)
- [ ] **4 GB RAM minimum** (8 GB recommended for smooth development)
- [ ] **Stable internet connection** (for initial npm install)

---

## Download Preparation

### ✅ Folder Location

- [ ] **Decide where to save the project**
  
  **Recommended Locations:**
  - Windows: `C:\Projects\vigil`
  - Mac: `~/Projects/vigil`
  - Linux: `~/projects/vigil`

  **Avoid:**
  - ❌ Desktop (gets cluttered)
  - ❌ Downloads folder (temporary location)
  - ❌ OneDrive/Dropbox sync folders (causes issues with node_modules)
  - ❌ Paths with spaces or special characters

---

## File System Permissions

### Windows

- [ ] **Not using OneDrive or network drive for project folder**
- [ ] **Antivirus won't block Node.js** (add exception if needed)

### Mac/Linux

- [ ] **Folder has write permissions**
  ```bash
  # Test by creating a file in your chosen folder
  cd ~/Projects
  touch test.txt
  rm test.txt
  ```

---

## Network Configuration

- [ ] **No corporate firewall blocking npm**
  ```bash
  # Test npm registry access
  npm ping
  # Should return: Ping success: {npm: '200'}
  ```

- [ ] **Proxy configured** (if behind corporate proxy)
  ```bash
  npm config set proxy http://proxy-server:port
  npm config set https-proxy http://proxy-server:port
  ```

---

## Post-Download Steps Preview

After downloading, you'll need to:

1. ✅ Extract the ZIP file
2. ✅ Open folder in VS Code
3. ✅ Run `npm install` (5-10 minutes first time)
4. ✅ Run `npm start`
5. ✅ Open browser to http://localhost:3000

**Total Time:** ~15 minutes for complete setup

---

## Expected File Structure (After Download)

```
vigil-surveillance-system/
├── 📁 public/
├── 📁 src/
│   ├── 📁 components/
│   ├── 📁 styles/
│   ├── 📄 index.js
│   └── 📄 App.jsx
├── 📁 components/       (legacy TypeScript)
├── 📄 package.json
├── 📄 webpack.config.js
├── 📄 PROJECT_README.md
├── 📄 INSTALLATION_GUIDE.md
└── 📄 .gitignore

Total Files: ~200
Total Size: ~500 KB (before npm install)
After npm install: ~350 MB (with node_modules)
```

---

## Quick Test After Download

Run these commands in sequence:

```bash
# 1. Navigate to project
cd vigil-surveillance-system

# 2. Verify package.json exists
ls package.json          # Mac/Linux
dir package.json         # Windows

# 3. Install dependencies
npm install

# 4. Start development server
npm start

# 5. Verify it opens in browser
# Should auto-open at http://localhost:3000
```

**Expected Results:**
- ✅ No errors during `npm install`
- ✅ Dev server starts successfully
- ✅ Browser opens automatically
- ✅ Login screen displays correctly
- ✅ Can login and see dashboard

---

## Troubleshooting Preparation

### Common Issues & Pre-emptive Solutions

#### Issue: `npm: command not found`

**Pre-check:**
```bash
which npm    # Mac/Linux
where npm    # Windows
```

**Solution:**
- Node.js not installed or not in PATH
- Restart terminal after Node.js installation

---

#### Issue: `EACCES: permission denied`

**Pre-check:**
- Don't use `sudo npm install`
- Check folder permissions

**Solution:**
```bash
# Fix npm permissions (Mac/Linux)
sudo chown -R $USER ~/.npm
```

---

#### Issue: `Port 3000 already in use`

**Pre-check:**
```bash
# Check what's using port 3000
lsof -i :3000          # Mac/Linux
netstat -ano | findstr :3000    # Windows
```

**Solution:**
- Kill the process using port 3000
- Or use different port: `PORT=3001 npm start`

---

#### Issue: Firewall/Antivirus Blocking

**Pre-check:**
- Temporarily disable firewall
- Add Node.js to antivirus exceptions

**Solution:**
- Whitelist `node.exe` and `npm.exe`
- Allow network access for webpack-dev-server

---

## Pre-Download Knowledge Check

Answer these questions to verify readiness:

1. **Do you know what Node.js is?**
   - [ ] Yes - it's a JavaScript runtime
   - [ ] No - read: https://nodejs.org/en/about/

2. **Do you know what npm is?**
   - [ ] Yes - it's the Node.js package manager
   - [ ] No - read: https://docs.npmjs.com/about-npm

3. **Do you understand how to use the terminal/command line?**
   - [ ] Yes - basic commands (cd, ls, etc.)
   - [ ] No - watch: YouTube "Terminal basics tutorial"

4. **Do you know what React is?**
   - [ ] Yes - it's a JavaScript UI library
   - [ ] No - it's okay, you'll learn! Read: https://react.dev/

5. **Do you have a plan for where to save the project?**
   - [ ] Yes - I've chosen a folder location
   - [ ] No - decide now (see "Folder Location" above)

---

## Final Pre-Flight Check

- [ ] ✅ Node.js 18+ installed and working
- [ ] ✅ npm working (tested with `npm --version`)
- [ ] ✅ VS Code (or editor) installed
- [ ] ✅ Modern browser installed
- [ ] ✅ 2+ GB free disk space
- [ ] ✅ Stable internet connection
- [ ] ✅ Folder location decided
- [ ] ✅ Basic terminal knowledge
- [ ] ✅ No network restrictions (proxy configured if needed)
- [ ] ✅ Antivirus won't block Node.js

---

## You're Ready! 🚀

If you checked all the boxes above, you're 100% ready to download and run Vigil.

### Next Steps:

1. **Download** the project ZIP file
2. **Extract** to your chosen location
3. **Follow** `/INSTALLATION_GUIDE.md`
4. **Enjoy** your Vigil surveillance system!

---

## Quick Start Command Reference

```bash
# 1. Navigate to project
cd vigil-surveillance-system

# 2. Install dependencies (first time only)
npm install

# 3. Start development server
npm start

# 4. Build for production
npm run build

# 5. Preview production build
npm run serve

# 6. Clean (if needed)
npm run clean
```

---

## Support

If you encounter issues after download:

1. ✅ Check `/INSTALLATION_GUIDE.md` troubleshooting section
2. ✅ Review browser console for errors (F12)
3. ✅ Verify all prerequisites are met
4. ✅ Try `npm run clean && npm install`

---

## Time Estimates

- **Download:** 1-2 minutes
- **Extract:** 30 seconds
- **npm install:** 3-10 minutes (first time)
- **npm start:** 30 seconds
- **Testing:** 5 minutes

**Total:** ~20 minutes from download to working app

---

## What to Expect

### On First Run

You'll see:
- ✅ Login screen with three role options (Admin, Officer, Security Authority)
- ✅ Dark theme by default (toggle available)
- ✅ Modern glassmorphism UI
- ✅ Cyan and amber accent colors

### After Login

You'll get:
- ✅ 9-camera DVR grid (3x3 layout)
- ✅ Real-time notifications (demo mode)
- ✅ Multiple dashboard tabs
- ✅ Analytics charts
- ✅ Map view with incident markers
- ✅ Fullscreen camera playback

### Everything Works Immediately

- ✅ No backend required (frontend-only)
- ✅ Demo data included
- ✅ All features functional
- ✅ Ready for customization

---

## Final Checklist Before Download

I have:
- [ ] ✅ Verified Node.js is installed (`node --version`)
- [ ] ✅ Verified npm is installed (`npm --version`)
- [ ] ✅ Decided where to save the project
- [ ] ✅ Verified I have 2+ GB free space
- [ ] ✅ VS Code or alternative editor ready
- [ ] ✅ Read the expected time requirements (~20 min total)
- [ ] ✅ Understood this is a frontend-only application
- [ ] ✅ Ready to follow `/INSTALLATION_GUIDE.md`

---

**If all boxes are checked, you're ready to download! 🎉**

See you on the other side with a working Vigil surveillance system!

---

*Pre-Download Checklist - Last Updated: January 2026*
