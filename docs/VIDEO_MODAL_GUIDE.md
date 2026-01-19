# 🎬 Video Modal - How It Works

## What's New

### Video Playback Modal
A professional fullscreen video player that opens when you click **"View Video"** button.

**Dimensions:**
- Width: `min(1000px, 90vw)` - Responsive from 900px to fullscreen
- Height: `min(700px, 85vh)` - Responsive viewport height
- Always centered on screen

---

## Features Inside Video Modal

### 1. **Download Button** (Inside Modal)
Located in the top header, next to volume/mute controls
- Click to download the current video
- Works in parallel with the camera grid download button
- Uses same backend video fetch logic

### 2. **Mute/Unmute Toggle**
Click volume icon to toggle sound on/off during playback

### 3. **Play/Pause**
- Video has built-in HTML5 controls
- Hover overlay with large play/pause button
- Video starts playing automatically

### 4. **Video Information**
- **Top header:** Shows incident type, camera ID, timestamp
- **Bottom footer:** Camera ID and current time
- Professional gradient backdrop with transparency

### 5. **Close Button**
Click X button to close modal and return to dashboard

---

## Where to Use It

### Incidents Tab
```
Incidents view → Click "View Video" button on any incident
→ Opens full-screen video modal with download button
```

### Live Control Dashboard
```
Live Control → Hover over camera feed
→ Click "View" button (blue button, leftmost)
→ Opens video in large modal with download
```

---

## Video URL Handling

### For Incidents:
```
videoUrl: `http://127.0.0.1:5000/videos/{camera_id}_sample.mp4`
```

### For Live Cameras:
```
videoUrl: comes from backend `/api/live-status` response
```

---

## Modal Styling

### Professional Design Elements:
- **Dark theme** with gradient backgrounds
- **Cyan accent border** (1px, 30% opacity)
- **Backdrop blur** effect on header/footer
- **Smooth animations** with Framer Motion
- **Responsive sizing** - scales on mobile/tablet/desktop

### Colors:
- Background: Black with dark gray gradient
- Border: Cyan/Blue accent
- Buttons: Color-coded (Blue for View, Green for Download)
- Text: White on dark background

### Animations:
- Entry: Scale + fade (spring physics)
- Exit: Reverse animation
- Transitions: Smooth 200-300ms

---

## Complete Usage Flow

**Scenario 1: Reviewing Incident**
```
1. Incidents tab visible
2. Hover over incident card
3. See "View Video" button (cyan)
4. Click "View Video"
5. Modal opens with video
6. Click "Download" button in header to save MP4
7. Click X to close modal
```

**Scenario 2: Monitoring Live Camera**
```
1. Live Control dashboard visible
2. See 12 camera feeds in grid
3. Hover over any camera (e.g., CAM-042)
4. Three buttons appear: View | Screenshot | Download
5. Click "View" (blue button)
6. Large fullscreen modal opens
7. Click "Download" button to save current stream
8. Video has HTML5 controls (timeline, volume, fullscreen)
```

---

## Browser Support

✅ Works in:
- Chrome/Edge (Chromium-based)
- Firefox
- Safari
- All modern mobile browsers

Requires:
- HTML5 video support
- ES6+ JavaScript
- CSS3 transforms & animations

---

## Technical Details

### Component Files:
- `src/components/VideoModal.tsx` - Modal component
- `src/components/IncidentFeed.tsx` - Uses modal for incidents
- `src/components/DVRCameraGrid.tsx` - Uses modal for live cameras

### Dependencies:
- React (hooks: useState, useRef)
- Framer Motion (AnimatePresence, motion.div)
- Lucide React (icons)
- Custom exportUtils (downloadVideoClip function)
- Sonner (toast notifications)

### Key Functions:
```typescript
// In VideoModal.tsx
- handleDownload() - Downloads video via exportUtils
- togglePlayPause() - Controls HTML5 video element
- toggleMute() - Mutes/unmutes video

// In IncidentFeed.tsx & DVRCameraGrid.tsx
- setSelectedVideoIncident() - Opens modal with incident video
- setVideoModalCamera() - Opens modal with camera video
```

---

## What's Responsive?

✅ **Mobile** (< 600px)
- Modal width: 90vw (near-full screen)
- Modal height: 85vh (full viewport)
- Buttons stack vertically if needed
- Touch-friendly controls

✅ **Tablet** (600px - 1200px)
- Modal width: 900px
- Modal height: 600px
- Everything readable and clickable

✅ **Desktop** (> 1200px)
- Modal width: 1000px (max)
- Modal height: 700px (max)
- All controls visible and spaced

---

## Download Button Notes

**Location inside modal:** Top-right header area
**Icon:** Green download icon
**Behavior:** 
- Fetches video from backend
- Shows success toast when complete
- Shows error toast if download fails
- Works offline (uses blob + URL.createObjectURL)

**File naming:**
- Pattern: `{cameraId}_timestamp.mp4`
- Example: `CAM-042_1704298800.mp4`

---

## Known Limitations

⚠️ Note: If backend returns video at 30+ MB size:
- Download may take a few seconds
- Browser shows download progress
- No progress bar in modal (standard browser behavior)

⚠️ If video URL 404:
- Modal shows "Video not available"
- Download button still appears but will fail
- Check backend is running and video exists

---

## Future Enhancements (Optional)

Could add:
- [ ] Timeline scrubber for frame-by-frame review
- [ ] Frame capture feature (extract specific frame as image)
- [ ] Playback speed controls
- [ ] Recording indicator if still recording
- [ ] AI annotation overlay (draws boxes around detected objects)
- [ ] Replay in slow-motion option
