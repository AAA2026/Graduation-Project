# Camera-Focused Dashboard Layout

## Overview
The Vigil surveillance system has been **completely redesigned** to mirror a traditional DVR/security monitor system. When you log in, cameras are IMMEDIATELY in your face - edge-to-edge, tightly packed like a real security control room.

## New DVR-Style Layout

### 1. **DVR Camera Grid** (Primary - Full Screen)
- **What You See First**: 4-6 camera feeds filling the screen on login
- **Layout Options**: 
  - 2x2 (4 cameras) - Default mobile
  - 2x3 (6 cameras) - Default desktop
  - 3x3 (9 cameras) - Medium density
  - 4x4 (16 cameras) - Maximum density
- **DVR Features**:
  - Edge-to-edge monitors with minimal gaps (2px borders)
  - Dark monitor frames like real security equipment
  - Scanline effects for authentic CRT/monitor feel
  - Live recording indicators ("REC" badges)
  - Real-time timestamps on each feed
  - Pulsing green status dots for online cameras
  - Red alert borders for incidents
  - Click any camera to select/highlight it
  - Hover for fullscreen action

### 2. **Minimal Control Bar** (Top - Compact)
- **Location**: Thin bar at the very top
- **Elements**:
  - System status (pulsing green indicator)
  - Role title (COMMAND CENTER / SURVEILLANCE CONTROL)
  - Collapsible "Stats" and "Incidents" buttons
  - Current time
- **Philosophy**: Don't distract from cameras

### 3. **Collapsible Panels** (Hidden by Default)
- **Stats Panel**: Click "Stats" to expand metrics - hides again to show cameras
- **Incidents Panel**: Click "Incidents" to show alerts sidebar
- **Default State**: CLOSED - cameras take 100% of view

### 4. **Bottom Info Bar**
- Shows camera count (e.g., "Displaying 6 of 12 cameras")
- Selected camera ID when clicked
- Current date

## Role-Specific Dashboards

### Admin Dashboard
- **Title**: "COMMAND CENTER"
- **Default Grid**: 2x3 (6 cameras)
- **Quick Access**: Stats and Incidents collapsed by default

### Officer Dashboard
- **Title**: "SURVEILLANCE CONTROL"
- **Default Grid**: 2x3 (6 cameras)
- **Quick Access**: Stats and Incidents collapsed by default

### Security Authority Dashboard
- **Title**: "SECURITY COMMAND"
- **Default Grid**: 2x3 (6 cameras)
- **Additional**: 4 quick action buttons (compact, 1 row)
- **Quick Access**: Emergency dispatch, team status

## DVR Visual Design

### Monitor Aesthetics
- **Background**: Deep black gradient (realistic monitor screens)
- **Borders**: 2px dark gray separators between feeds
- **Effects**:
  - Subtle scanline animation (8s loop)
  - Grid pattern overlay (cyan tint at 5% opacity)
  - Recording indicator (red dot + "REC" badge)
  - Monospace timestamps (HH:MM:SS format)
  
### Alert States
- **Normal**: Dark gray borders, cyan accents
- **Alert**: Red borders (2px), pulsing alert badge
- **Offline/Maintenance**: Amber status indicator

### Camera Information Overlay
- **Top**: Camera ID + Status dot + Alert badge (if active)
- **Bottom**: Location + Live timestamp
- **Hover**: Black overlay with "Fullscreen" button

## Navigation Changes
All dashboards now start with "Live Control" emphasizing the camera-first DVR approach:
- Admin: "Live Control" (not "Dashboard")
- Officer: "Live Control" (not "Dashboard")  
- Security Authority: "Live Control" (default view)

## Design Philosophy (Revised)
1. **Cameras DOMINATE**: Fills 100% of screen on login, like walking into a security operations center
2. **DVR Authenticity**: Dark monitors, scanlines, recording indicators, timestamp overlays
3. **Everything Else Hides**: Stats/incidents collapsed until needed
4. **Tight Grid**: Minimal spacing (2px gaps) - cameras side-by-side
5. **Professional Feel**: Looks like equipment security personnel actually use

## Mobile Responsiveness
- 2x3 grid becomes 2 columns on mobile
- Control buttons stack vertically
- Stats/Incidents remain collapsible
- Maintains DVR aesthetic even on small screens

## Technical Features
- **Scanline Animation**: CSS keyframe animation for CRT effect
- **Grid Layouts**: Responsive from 4 to 16 cameras
- **Click to Select**: Border highlight on selected camera
- **Mute Toggle**: Audio control for all feeds
- **Smooth Transitions**: Collapsible panels with Motion animations