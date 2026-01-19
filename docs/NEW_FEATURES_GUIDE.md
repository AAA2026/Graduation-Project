# New Professional Features - User Guide

## ✨ Features Implemented

### 1. 📊 Status Dashboard Widget
**Location:** Bottom-right corner of all dashboards

**Features:**
- **Real-time System Health Monitoring** using actual backend data
  - API Status (response time in ms)
  - Database connection status
  - System uptime
  - Active cameras count
- **Auto-updates every 5 seconds** from `/api/live-status` and `/api/simulator-stats`
- **Compact/Expanded View** - Click to toggle detailed view
- **Color-coded Status Indicators**:
  - 🟢 Green = Healthy (API < 500ms)
  - 🟡 Yellow = Degraded (API 500-1500ms)
  - 🔴 Red = Down (API > 1500ms or failed)

**How to Use:**
- Widget is always visible in bottom-right corner
- Click the Activity icon to expand/collapse
- Shows real backend metrics - no fake data!

---

### 2. 📥 Export & Download Features

#### **Incident Reports Export**
**Location:** Incidents tab → Export buttons

**Formats Available:**
- **CSV Export** - Spreadsheet format with all incident data
  - Fields: ID, Timestamp, Camera, Type, Confidence, Location, Status, Model
  - File: `vigil_incidents_[timestamp].csv`
  
- **PDF Report** - Professional formatted report
  - Header with generation date and total count
  - Formatted table with all incidents
  - Footer with page numbers
  - File: `vigil_incident_report_[timestamp].pdf`

**How to Use:**
1. Go to **Incidents** tab
2. Apply filters if needed (All/Active/Resolved)
3. Click "Export CSV" or "Export PDF" button
4. File downloads automatically to your Downloads folder

#### **Camera Feed Screenshots**
**Available Functions:**
```typescript
import { captureScreenshot, captureAllCameraFeeds } from '../utils/exportUtils';

// Capture single element
captureScreenshot('elementId', 'my_screenshot.png');

// Capture all camera feeds at once
captureAllCameraFeeds();
```

#### **Analytics Data Export**
```typescript
import { exportAnalyticsToCSV, exportCameraDataToCSV } from '../utils/exportUtils';

// Export analytics data
exportAnalyticsToCSV(analyticsData);

// Export camera status data
exportCameraDataToCSV(cameraData);
```

---

### 3. 🍞 Breadcrumb Navigation
**Location:** Top of each view (Incidents, Analytics, Map, etc.)

**Features:**
- Shows current navigation path
- Home icon for dashboard
- Clickable navigation items
- Format: `Dashboard > Incidents > INC-2024-1289`

**How to Use:**
- Automatically appears on all views
- Click any breadcrumb item to navigate back
- Last item (current page) is highlighted in white

**Example Usage in Components:**
```tsx
import Breadcrumb, { BreadcrumbItem } from './Breadcrumb';

const breadcrumbItems: BreadcrumbItem[] = [
  { label: 'Dashboard', onClick: () => setView('dashboard') },
  { label: 'Incidents' },
];

<Breadcrumb items={breadcrumbItems} />
```

---

### 4. ⏸️ Auto-Pause Mode
**Location:** Top bar of Live Control dashboard

**Features:**
- **Pause real-time updates** when reviewing data
- **Missed incident counter** shows how many incidents occurred while paused
- **Resume button** to catch up on missed incidents
- **No notifications while paused** - prevents alert spam
- **Visual indicators**:
  - ⏸️ PAUSED (yellow) - Shows missed count: `⏸️ PAUSED (3)`
  - ▶️ LIVE (blue) - Real-time updates active

**How to Use:**
1. Click **"⏸️ PAUSED"** or **"▶️ LIVE"** button in top bar
2. While paused:
   - Incidents still get added to database
   - Counter shows missed incidents
   - No toast notifications or sounds
3. Click **"▶️ LIVE"** to resume:
   - Shows notification: "Resumed • 3 new incidents while paused"
   - Resumes real-time polling and notifications

**Use Cases:**
- Reviewing specific incident details without interruption
- Analyzing camera feeds without new alerts
- Taking screenshots or generating reports
- Presentations/demos where alerts would be distracting

---

## 🔧 Technical Details

### Real Data Sources

All features use **actual backend data** from VIGIL:

1. **Status Widget:**
   - `/api/live-status` - Camera states, active count
   - `/api/simulator-stats` - Simulator metadata
   - Calculates API response time in real-time

2. **Export Functions:**
   - Uses incident data from `useRealtimeIncidents` hook
   - Formats actual confidence scores, timestamps, locations
   - Pulls video URLs from backend camera states

3. **Auto-Pause:**
   - Modifies `useRealtimeIncidents` hook behavior
   - Pauses polling intervals
   - Tracks missed incidents in state

### Dependencies Installed
```json
{
  "jspdf": "^2.x.x",
  "jspdf-autotable": "^3.x.x",
  "html2canvas": "^1.x.x",
  "@types/jspdf": "^2.x.x"
}
```

### File Structure
```
src/
├── components/
│   ├── StatusWidget.tsx          # System health monitor
│   ├── Breadcrumb.tsx             # Navigation breadcrumbs
│   ├── DashboardLayout.tsx        # Now includes StatusWidget
│   ├── DashboardHome.tsx          # Now includes auto-pause controls
│   └── IncidentsView.tsx          # Now includes export buttons + breadcrumbs
├── utils/
│   └── exportUtils.ts             # PDF/CSV/Screenshot utilities
└── hooks/
    └── useRealtimeIncidents.tsx   # Now includes pause/resume logic
```

---

## 🎨 UI/UX Enhancements

### Visual Consistency
- All widgets use Tailwind CSS with dark theme
- Consistent color coding across features
- Smooth animations with Framer Motion
- Professional typography and spacing

### User Feedback
- Toast notifications for all actions
- Loading states for async operations
- Error handling with user-friendly messages
- Accessibility-friendly design

### Responsive Design
- Mobile-friendly layouts
- Touch-optimized controls
- Adaptive sizing for all screen sizes
- Hidden overflow handling

---

## 📝 Usage Examples

### Example 1: Export Incidents Report
```typescript
// In IncidentsView component
const incidents = useRealtimeIncidents().incidents;

// Click Export PDF button
<Button onClick={() => exportIncidentsToPDF(incidents)}>
  <FileText className="w-4 h-4 mr-2" />
  Export PDF
</Button>
```

### Example 2: Monitor System Health
```typescript
// StatusWidget automatically polls backend
// User just needs to click to expand/collapse

// In DashboardLayout.tsx
<StatusWidget />
```

### Example 3: Pause During Review
```typescript
// In DashboardHome component
const { isPaused, togglePause, missedIncidents } = useRealtimeIncidents();

// Toggle button
<button onClick={togglePause}>
  {isPaused ? '▶️ Resume' : '⏸️ Pause'}
  {missedIncidents > 0 && ` (${missedIncidents})`}
</button>
```

---

## 🚀 Next Steps

All features are fully integrated and ready to use! The data comes directly from your VIGIL backend, so:

1. ✅ Start the backend: `python backend/app.py`
2. ✅ Start the React app: `npm run dev`
3. ✅ Log in and explore the new features
4. ✅ All features work with **real data** from your surveillance system!

**No configuration needed** - everything is automatically wired up!
