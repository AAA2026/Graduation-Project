# Vigil - Professional Features Guide (Vite Version)

## ✨ New Features Implemented (Frontend-Only)

This guide covers the professional features that have been added to your Vigil surveillance system. All features work **without a backend** using mock data and frontend state management.

---

## 1. 📊 Status Dashboard Widget

### Overview
A real-time system health monitor that appears in the **bottom-right corner** of all dashboards.

### Features
- **Mock System Health Monitoring** using simulated data:
  - API Status (healthy/degraded/down)
  - API Response Time (100-400ms simulated)
  - Database connection status (always connected in mock)
  - System uptime (tracked from app start)
  - Active cameras count (5-6 cameras simulated)
  - Total cameras (6 cameras)

- **Auto-updates every 5 seconds** with new mock data
- **Compact/Expanded View** - Click to toggle detailed metrics
- **Color-coded Status Indicators**:
  - 🟢 Green = Healthy (API < 500ms)
  - 🟡 Yellow = Degraded (API 500-1500ms)
  - 🔴 Red = Down (API > 1500ms)

### Location
```tsx
// Always visible in ModernSecurityLayout
import StatusWidget from './components/StatusWidget';

// Rendered at bottom of layout
<StatusWidget />
```

### Usage
- Widget is **always visible** in bottom-right corner
- Click the **Activity icon** to expand/collapse details
- Shows simulated metrics - perfect for demos!

---

## 2. 📥 Export & Download Features

### Overview
Professional export capabilities for incident reports, analytics, and camera data.

### Supported Formats

#### **CSV Export**
- Spreadsheet format with all incident data
- Fields included:
  - ID, Timestamp, Camera ID, Camera Name
  - Type, Confidence %, Location, Status
  - Assigned Officer, Response Time, AI Model, Video URL
- File naming: `vigil_incidents_[timestamp].csv`

#### **PDF Export**
- Professional formatted reports
- Includes:
  - Header with generation date
  - Total incident count
  - Formatted table with all data
  - Footer with page numbers
- File naming: `vigil_incident_report_[timestamp].pdf`

### How to Use

#### In Incidents Tab:
1. Navigate to **Incidents** tab
2. Apply filters if needed (All/Active/Resolved)
3. Click **"Export CSV"** or **"Export PDF"** button
4. File downloads automatically to your Downloads folder
5. Toast notification confirms successful export

#### Export Buttons Available:
```tsx
<Button onClick={handleExportCSV}>
  <Download className="w-4 h-4 mr-2" />
  Export CSV
</Button>

<Button onClick={handleExportPDF}>
  <FileText className="w-4 h-4 mr-2" />
  Export PDF
</Button>
```

### Available Export Functions

```typescript
import {
  exportIncidentsToCSV,
  exportIncidentsToPDF,
  exportAnalyticsToCSV,
  exportAnalyticsToPDF,
  exportCameraDataToCSV,
  captureScreenshot,
  captureAllCameraFeeds,
  captureMapScreenshot,
} from './utils/exportUtils';

// Export incidents
exportIncidentsToCSV(incidents);
exportIncidentsToPDF(incidents);

// Export analytics
exportAnalyticsToCSV(analyticsData);
exportAnalyticsToPDF(analyticsData);

// Export camera data
exportCameraDataToCSV(cameras);

// Capture screenshots
captureScreenshot('elementId', 'filename.png');
captureAllCameraFeeds(); // Captures entire camera grid
captureMapScreenshot(); // Captures map view
```

---

## 3. 🍞 Breadcrumb Navigation

### Overview
Professional navigation breadcrumbs showing your current location in the app.

### Features
- Shows current navigation path
- Home icon for quick dashboard access
- Clickable navigation items to go back
- Auto-updates based on current view
- Format: `🏠 Dashboard > Incidents`

### Locations
- **Incidents View** - Shows "Dashboard > Incidents"
- Easily extensible to other views

### Usage

```tsx
import Breadcrumb, { BreadcrumbItem } from './components/Breadcrumb';

const breadcrumbItems: BreadcrumbItem[] = [
  { label: 'Dashboard', onClick: () => navigate('dashboard') },
  { label: 'Incidents' },
];

<Breadcrumb items={breadcrumbItems} />
```

### Visual Design
- Home icon as first item
- Chevron separators (>)
- Current page highlighted in white
- Previous pages are clickable and colored cyan
- Responsive design for mobile

---

## 4. ⏸️ Auto-Pause Mode (Coming Soon)

This feature can be easily added to pause real-time incident updates when you need to review data without interruption.

### Planned Features:
- Pause/Resume button in header
- Missed incident counter
- No notifications while paused
- Resume catches up on missed incidents

### Implementation Guide:

```typescript
// In useRealtimeIncidents hook
const [isPaused, setIsPaused] = useState(false);
const [missedIncidents, setMissedIncidents] = useState(0);

const togglePause = () => {
  setIsPaused(!isPaused);
  if (isPaused) {
    toast.success(`Resumed • ${missedIncidents} new incidents while paused`);
    setMissedIncidents(0);
  } else {
    toast.info('Real-time updates paused');
  }
};

// In component
<Button onClick={togglePause}>
  {isPaused ? '▶️ LIVE' : '⏸️ PAUSED'}
  {missedIncidents > 0 && ` (${missedIncidents})`}
</Button>
```

---

## 🛠️ Technical Implementation

### Dependencies Added

```json
{
  "dependencies": {
    "jspdf": "^2.5.2",
    "jspdf-autotable": "^3.8.3",
    "html2canvas": "^1.4.1"
  },
  "devDependencies": {
    "@types/jspdf": "^2.0.0"
  }
}
```

### File Structure

```
src/
├── components/
│   ├── StatusWidget.tsx          # System health monitor (NEW)
│   ├── Breadcrumb.tsx             # Navigation breadcrumbs (NEW)
│   ├── ModernSecurityLayout.tsx   # Updated with StatusWidget
│   └── IncidentsView.tsx          # Updated with exports + breadcrumbs
├── utils/
│   └── exportUtils.ts             # Export utilities (NEW)
└── hooks/
    └── useRealtimeIncidents.ts    # Real-time incident management
```

---

## 📝 Usage Examples

### Example 1: Export Incidents to PDF
```typescript
import { exportIncidentsToPDF } from './utils/exportUtils';
import { toast } from 'sonner';

const handleExportPDF = () => {
  try {
    const exportData = incidents.map(inc => ({
      id: inc.id,
      timestamp: inc.timestamp,
      cameraId: inc.cameraId,
      cameraName: inc.cameraId,
      type: inc.type,
      confidence: inc.confidence / 100,
      location: inc.location,
      status: inc.status,
      aiModel: 'YOLO-v8',
    }));
    
    exportIncidentsToPDF(exportData);
    toast.success('Incidents exported to PDF successfully');
  } catch (error) {
    toast.error('Failed to export PDF');
  }
};
```

### Example 2: Monitor System Health
```typescript
// StatusWidget automatically manages its own state
// Just render it in your layout

import StatusWidget from './components/StatusWidget';

function Layout() {
  return (
    <div>
      {/* Your content */}
      <StatusWidget />
    </div>
  );
}
```

### Example 3: Add Breadcrumbs to View
```typescript
import Breadcrumb from './components/Breadcrumb';

function AnalyticsView() {
  const breadcrumbItems = [
    { label: 'Dashboard', onClick: () => navigate('dashboard') },
    { label: 'Analytics' },
  ];

  return (
    <div>
      <Breadcrumb items={breadcrumbItems} />
      {/* Rest of analytics content */}
    </div>
  );
}
```

---

## 🎨 UI/UX Enhancements

### Visual Consistency
- All components use Vigil theme variables
- Dark/Light mode support
- Consistent color coding (cyan/amber accents)
- Smooth Framer Motion animations
- Professional typography and spacing

### User Feedback
- Toast notifications for all actions
- Loading states for async operations
- Error handling with friendly messages
- Accessibility-friendly design (ARIA labels, keyboard nav)

### Responsive Design
- Mobile-friendly layouts (< 640px)
- Tablet optimized (640px - 1024px)
- Desktop optimized (> 1024px)
- Touch-optimized controls
- Adaptive component sizing

---

## 🚀 Installation & Setup

### 1. Install Dependencies
```bash
npm install jspdf jspdf-autotable html2canvas
npm install --save-dev @types/jspdf
```

### 2. Copy New Files
Ensure you have these files in your Vite project:
- `/src/components/StatusWidget.tsx`
- `/src/components/Breadcrumb.tsx`
- `/src/utils/exportUtils.ts`

### 3. Update Existing Files
The following files have been updated:
- `/src/components/ModernSecurityLayout.tsx` (added StatusWidget)
- `/src/components/IncidentsView.tsx` (added exports + breadcrumbs)

### 4. Start Development Server
```bash
npm run dev
```

### 5. Test Features
1. ✅ Login to the app
2. ✅ Check bottom-right corner for StatusWidget
3. ✅ Go to Incidents tab
4. ✅ See breadcrumb navigation
5. ✅ Click "Export CSV" and "Export PDF" buttons
6. ✅ Check your Downloads folder for exported files

---

## 📊 Mock Data vs Real Data

### Current Implementation (Mock Data)
All features currently use **frontend mock data**:

- **StatusWidget**: Simulates API calls with setTimeout
- **Exports**: Uses current incident state from React
- **Breadcrumbs**: Navigation state only

### Future Integration (Real Backend)
To connect to real backend later:

1. **Replace mock data in StatusWidget**:
```typescript
// Instead of setTimeout simulation
const response = await fetch('/api/live-status');
const data = await response.json();
setStatus(data);
```

2. **Exports already work** - they just need real incident data passed in
3. **Breadcrumbs work** - no backend needed

---

## 🔍 Troubleshooting

### Issue: PDF export not working
**Solution**: 
1. Check browser console for errors
2. Ensure `jspdf` and `jspdf-autotable` are installed
3. Try disabling browser PDF viewer extensions

### Issue: CSV download not starting
**Solution**:
1. Check if browser is blocking downloads
2. Verify data array is not empty
3. Check browser console for errors

### Issue: StatusWidget not appearing
**Solution**:
1. Check if `StatusWidget` is imported in layout
2. Verify z-index (should be 50)
3. Check CSS for `fixed` positioning

### Issue: Toast notifications not showing
**Solution**:
1. Ensure `<Toaster />` is in App.tsx
2. Import `toast` from `sonner`
3. Check if toast container is rendered

---

## 💡 Customization Guide

### Customize StatusWidget Colors
```typescript
// In StatusWidget.tsx
const getStatusColor = () => {
  switch (status.apiStatus) {
    case 'healthy':
      return 'text-green-500'; // Change to your color
    case 'degraded':
      return 'text-yellow-500';
    case 'down':
      return 'text-red-500';
  }
};
```

### Customize Export File Names
```typescript
// In exportUtils.ts
const filename = `vigil_incident_report_${Date.now()}.pdf`;
// Change to:
const filename = `MyCompany_Report_${Date.now()}.pdf`;
```

### Customize Breadcrumb Styling
```typescript
// In Breadcrumb.tsx
className="text-[var(--vigil-primary)]" // Change color
className="text-lg" // Change size
```

---

## 🎯 Performance Optimization

### Recommendations:
1. **Lazy load export libraries**:
```typescript
const exportPDF = async () => {
  const { exportIncidentsToPDF } = await import('./utils/exportUtils');
  exportIncidentsToPDF(data);
};
```

2. **Debounce status updates**:
```typescript
const debouncedUpdate = useMemo(
  () => debounce(updateStatus, 5000),
  []
);
```

3. **Virtual scrolling for large exports**:
Use `react-window` for exporting 1000+ rows

---

## 📚 Additional Resources

- **jsPDF Documentation**: https://github.com/parallax/jsPDF
- **html2canvas Guide**: https://html2canvas.hertzen.com/
- **Framer Motion**: https://motion.dev
- **Sonner (Toast)**: https://sonner.emilkowal.ski/

---

## 🎉 You're All Set!

All professional features are now fully integrated and working with mock data. The UI/UX matches your Vigil design system perfectly.

### Quick Feature Checklist:
- ✅ Status Widget - Real-time system health
- ✅ Export to CSV - Download incident data
- ✅ Export to PDF - Professional reports
- ✅ Breadcrumb Navigation - Easy navigation
- ✅ Screenshot Capture - Save camera feeds
- ✅ Toast Notifications - User feedback
- ✅ Dark/Light Mode - Full theme support

**Happy coding!** 🚀
