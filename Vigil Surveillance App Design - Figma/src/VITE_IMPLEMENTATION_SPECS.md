# Vigil - Vite App Implementation Specifications

## Development Environment Specs
- **Target Resolution**: 1920px × 1080px (Full HD) - primary development viewport
- **Minimum Resolution**: 1366px × 768px (laptop standard)
- **Testing Viewports**: 
  - Mobile: 375px × 812px (iPhone)
  - Tablet: 768px × 1024px (iPad)
  - Desktop: 1920px × 1080px (Full HD)
  - Large Desktop: 2560px × 1440px (2K)

---

## Global Layout Structure

### Application Container
```css
.app-container {
  min-height: 100vh;
  width: 100vw;
  background: theme-dependent;
}
```

### Header (Sticky)
```css
.header {
  height: 64px; /* base height */
  width: 100%;
  position: sticky;
  top: 0;
  z-index: 50;
  border-bottom: 1px solid var(--border);
}

.header-expanded {
  height: 160px; /* with Quick Stats panel */
}

.navigation-tabs-bar {
  height: 48px;
  border-top: 1px solid var(--border);
}
```

### Main Content Area
```css
.main-content {
  padding: 24px; /* desktop: p-6 */
  min-height: calc(100vh - 112px); /* 64px header + 48px nav */
}

/* Mobile */
@media (max-width: 640px) {
  .main-content {
    padding: 16px; /* p-4 */
  }
}
```

**Available Content Height**: `calc(100vh - 112px)` (after header + nav)

---

## 1. LIVE MONITOR TAB (Camera Grid)

### Container Dimensions
```css
.camera-grid-container {
  width: 100%;
  height: calc(100vh - 160px); /* 112px header/nav + 48px controls */
  padding: 0;
}

.camera-grid-controls {
  height: 48px;
  margin-bottom: 8px;
  padding: 0 8px;
}
```

### Grid Layouts

#### 2×2 Grid (4 cameras)
```css
.grid-2x2 {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  grid-template-rows: repeat(2, 1fr);
  gap: 8px;
  height: calc(100vh - 160px);
}

.camera-cell {
  aspect-ratio: 16 / 9;
  border-radius: 12px;
  overflow: hidden;
}
```
- **Each camera dimensions** (at 1920px viewport):
  - Width: `(1920px - 48px - 8px) / 2 = 932px`
  - Height: `524px` (maintains 16:9)

#### 2×3 Grid (6 cameras) - DEFAULT
```css
.grid-2x3 {
  display: grid;
  grid-template-columns: repeat(2, 1fr); /* mobile */
  grid-template-rows: repeat(3, 1fr);
  gap: 8px;
  height: calc(100vh - 160px);
}

@media (min-width: 768px) {
  .grid-2x3 {
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: repeat(2, 1fr);
  }
}
```
- **Each camera dimensions** (at 1920px viewport, desktop 3-col):
  - Width: `(1920px - 48px - 16px) / 3 = 618px`
  - Height: `348px` (maintains 16:9)

#### 3×3 Grid (9 cameras)
```css
.grid-3x3 {
  display: grid;
  grid-template-columns: repeat(2, 1fr); /* mobile */
  grid-template-rows: repeat(5, auto);
  gap: 8px;
  height: auto;
  min-height: calc(100vh - 160px);
}

@media (min-width: 768px) {
  .grid-3x3 {
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: repeat(3, 1fr);
  }
}
```
- **Each camera dimensions** (at 1920px viewport):
  - Width: `(1920px - 48px - 16px) / 3 = 618px`
  - Height: `348px`

#### 4×4 Grid (16 cameras)
```css
.grid-4x4 {
  display: grid;
  grid-template-columns: repeat(2, 1fr); /* mobile */
  gap: 8px;
  height: auto;
}

@media (min-width: 768px) {
  .grid-4x4 {
    grid-template-columns: repeat(4, 1fr);
    grid-template-rows: repeat(4, auto);
  }
}
```
- **Each camera dimensions** (at 1920px viewport):
  - Width: `(1920px - 48px - 24px) / 4 = 462px`
  - Height: `260px`

### Individual Camera Card
```css
.camera-card {
  position: relative;
  border: 2px solid var(--border);
  border-radius: 12px;
  background: #000;
  overflow: hidden;
  aspect-ratio: 16 / 9;
}

.camera-card.alert {
  border-color: #ef4444;
  animation: pulse 2s infinite;
}

.camera-overlay-top {
  position: absolute;
  top: 8px;
  left: 8px;
  right: 8px;
  display: flex;
  justify-content: space-between;
  z-index: 10;
}

.camera-overlay-bottom {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 8px;
  background: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(4px);
}
```

---

## 2. INCIDENTS TAB

### Container Dimensions
```css
.incidents-container {
  width: 100%;
  max-width: 1920px;
  margin: 0 auto;
  padding: 0;
}

.incidents-header {
  margin-bottom: 24px;
  height: auto; /* ~80px */
}

.filter-tabs {
  height: 48px;
  margin-bottom: 24px;
  display: flex;
  gap: 4px;
}
```

### Incident Cards Grid
```css
.incidents-grid {
  display: grid;
  grid-template-columns: 1fr; /* mobile */
  gap: 16px;
  width: 100%;
}

@media (min-width: 768px) {
  .incidents-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (min-width: 1024px) {
  .incidents-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (min-width: 1536px) {
  .incidents-grid {
    grid-template-columns: repeat(4, 1fr);
  }
}
```

### Individual Incident Card
```css
.incident-card {
  width: 100%;
  height: auto; /* ~280px content-based */
  min-height: 260px;
  padding: 20px;
  border-radius: 12px;
  border: 1px solid var(--border);
  background: var(--card);
}
```

**Card Dimensions at Different Viewports:**
- **1920px (4-col)**: `(1920px - 48px - 48px) / 4 = 456px width`
- **1440px (3-col)**: `(1440px - 48px - 32px) / 3 = 453px width`
- **1024px (3-col)**: `(1024px - 48px - 32px) / 3 = 315px width`
- **768px (2-col)**: `(768px - 48px - 16px) / 2 = 352px width`

### Incident Detail Modal
```css
.incident-modal {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 98vw;
  height: 98vh;
  max-width: none;
  border-radius: 16px;
  background: var(--background);
  z-index: 50;
  overflow: hidden;
}

.modal-header {
  height: 80px;
  padding: 24px;
  border-bottom: 1px solid var(--border);
}

.modal-content {
  height: calc(98vh - 160px); /* subtract header + footer */
  overflow-y: auto;
  padding: 24px;
  display: grid;
  gap: 24px;
}

@media (min-width: 1024px) {
  .modal-content {
    grid-template-columns: 1.5fr 1fr; /* 60% video, 40% details */
  }
}

.modal-footer {
  height: 80px;
  padding: 16px 24px;
  border-top: 1px solid var(--border);
}
```

**Modal Dimensions at 1920px viewport:**
- Width: `1920px × 0.98 = 1881px`
- Height: `1080px × 0.98 = 1058px`
- Video Section: `1881px × 0.6 = 1128px width`
- Details Section: `1881px × 0.4 = 752px width`

#### Video Player in Modal
```css
.video-player-container {
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #000;
  border-radius: 12px;
  position: relative;
}

.video-controls {
  height: 56px;
  padding: 12px;
  background: rgba(0, 0, 0, 0.8);
  border-radius: 0 0 12px 12px;
}
```

---

## 3. MAP VIEW TAB

### Container Dimensions
```css
.map-container {
  width: 100%;
  height: calc(100vh - 136px); /* 112px header/nav + 24px padding */
  min-height: 600px;
  border-radius: 12px;
  overflow: hidden;
  position: relative;
}
```

**Map Dimensions at Different Viewports:**
- **1920×1080**: `1920px × 944px` (100% width × calculated height)
- **1440×900**: `1440px × 740px`
- **1366×768**: `1366px × 608px` (reaches min-height 600px)

### Map Elements
```css
.map-marker {
  width: 32px;
  height: 32px;
  position: absolute;
  transform: translate(-50%, -100%); /* center on point */
}

.map-marker.incident {
  width: 40px;
  height: 40px;
}

.map-popup {
  width: 280px;
  max-width: 90vw;
  padding: 16px;
  border-radius: 12px;
  background: var(--card);
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
}

.map-legend {
  position: absolute;
  bottom: 20px;
  left: 20px;
  width: 200px;
  padding: 12px;
  background: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(8px);
  border-radius: 8px;
}
```

---

## 4. ANALYTICS TAB

### Container Dimensions
```css
.analytics-container {
  width: 100%;
  max-width: 1920px;
  margin: 0 auto;
}

.analytics-header {
  height: auto; /* ~80px */
  margin-bottom: 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
```

### Key Metrics Cards
```css
.metrics-grid {
  display: grid;
  grid-template-columns: 1fr; /* mobile */
  gap: 16px;
  margin-bottom: 24px;
}

@media (min-width: 768px) {
  .metrics-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (min-width: 1024px) {
  .metrics-grid {
    grid-template-columns: repeat(4, 1fr);
  }
}

.metric-card {
  height: 140px;
  padding: 24px;
  border-radius: 12px;
  border: 1px solid var(--border);
}
```

**Metric Card Dimensions at 1920px (4-col):**
- Width: `(1920px - 48px - 48px) / 4 = 456px`
- Height: `140px` (fixed)

### Charts Grid
```css
.charts-grid {
  display: grid;
  grid-template-columns: 1fr; /* mobile */
  gap: 24px;
  margin-bottom: 24px;
}

@media (min-width: 1024px) {
  .charts-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

.chart-card {
  width: 100%;
  height: auto;
  padding: 24px;
  border-radius: 12px;
  background: var(--card);
}

.chart-container {
  width: 100%;
  height: 300px; /* fixed chart height */
}
```

**Chart Card Dimensions at 1920px (2-col):**
- Width: `(1920px - 48px - 24px) / 2 = 924px`
- Height: `~400px` (300px chart + 100px padding/header)

### Alert Resolution Status
```css
.resolution-card {
  width: 100%;
  padding: 24px;
  border-radius: 12px;
}

.progress-bar-container {
  height: 12px;
  width: 100%;
  border-radius: 9999px;
  margin-top: 8px;
  margin-bottom: 16px;
}
```

---

## 5. CAMERAS TAB (Admin)

### Container Dimensions
```css
.cameras-container {
  width: 100%;
  max-width: 1920px;
  margin: 0 auto;
}

.cameras-header {
  height: 64px;
  margin-bottom: 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.search-filter-bar {
  height: 56px;
  margin-bottom: 16px;
  display: flex;
  gap: 8px;
  align-items: center;
}
```

### Camera Table (Desktop)
```css
.camera-table {
  width: 100%;
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
}

.table-header-row {
  height: 48px;
  background: var(--muted);
}

.table-row {
  height: 60px;
  border-bottom: 1px solid var(--border);
}

.table-cell {
  padding: 12px;
}
```

**Table Column Widths (at 1872px available width after padding):**
1. **Camera ID**: `280px` (15%)
2. **Name**: `468px` (25%)
3. **Location**: `468px` (25%)
4. **Status**: `224px` (12%)
5. **Zone**: `243px` (13%)
6. **Actions**: `187px` (10%)

### Camera Cards (Mobile/Tablet)
```css
.cameras-card-grid {
  display: grid;
  grid-template-columns: 1fr; /* mobile */
  gap: 12px;
}

@media (min-width: 768px) {
  .cameras-card-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

.camera-card {
  height: auto; /* ~120px */
  padding: 16px;
  border-radius: 12px;
  border: 1px solid var(--border);
}
```

### Add/Edit Camera Modal
```css
.camera-modal {
  width: 600px;
  max-width: 95vw;
  padding: 24px;
  border-radius: 12px;
}

.form-field {
  height: 40px;
  margin-bottom: 16px;
}

.form-label {
  height: 20px;
  font-size: 14px;
  margin-bottom: 8px;
}
```

---

## 6. SYSTEM HEALTH TAB (Admin)

### Container Dimensions
```css
.health-container {
  width: 100%;
  max-width: 1920px;
  margin: 0 auto;
}

.health-header {
  height: 64px;
  margin-bottom: 24px;
}
```

### Health Metrics Cards
```css
.health-metrics-grid {
  display: grid;
  grid-template-columns: 1fr; /* mobile */
  gap: 16px;
  margin-bottom: 24px;
}

@media (min-width: 768px) {
  .health-metrics-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (min-width: 1024px) {
  .health-metrics-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (min-width: 1536px) {
  .health-metrics-grid {
    grid-template-columns: repeat(4, 1fr);
  }
}

.health-card {
  height: 160px;
  padding: 20px;
  border-radius: 12px;
  border: 1px solid var(--border);
}
```

**Health Card Dimensions at 1920px (4-col):**
- Width: `(1920px - 48px - 48px) / 4 = 456px`
- Height: `160px`

### System Logs Table
```css
.logs-container {
  width: 100%;
  max-height: 400px;
  overflow-y: auto;
  border: 1px solid var(--border);
  border-radius: 12px;
  margin-bottom: 24px;
}

.log-row {
  height: 48px;
  padding: 12px;
  border-bottom: 1px solid var(--border);
}
```

### Resource Usage Charts
```css
.resource-charts-grid {
  display: grid;
  grid-template-columns: 1fr; /* mobile */
  gap: 24px;
}

@media (min-width: 1024px) {
  .resource-charts-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

.resource-chart {
  width: 100%;
  height: 200px;
  padding: 16px;
  border-radius: 12px;
  background: var(--card);
}
```

---

## 7. AI MODELS TAB (Admin)

### Container Dimensions
```css
.models-container {
  width: 100%;
  max-width: 1920px;
  margin: 0 auto;
}

.models-header {
  height: 64px;
  margin-bottom: 24px;
}
```

### Model Cards Grid
```css
.models-grid {
  display: grid;
  grid-template-columns: 1fr; /* mobile */
  gap: 16px;
}

@media (min-width: 768px) {
  .models-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (min-width: 1024px) {
  .models-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (min-width: 1536px) {
  .models-grid {
    grid-template-columns: repeat(4, 1fr);
  }
}

.model-card {
  height: auto; /* ~240px */
  min-height: 220px;
  padding: 20px;
  border-radius: 12px;
  border: 1px solid var(--border);
}
```

**Model Card Dimensions at Different Viewports:**
- **1920px (4-col)**: `(1920px - 48px - 48px) / 4 = 456px width`
- **1440px (3-col)**: `(1440px - 48px - 32px) / 3 = 453px width`
- **1024px (3-col)**: `(1024px - 48px - 32px) / 3 = 315px width`
- **768px (2-col)**: `(768px - 48px - 16px) / 2 = 352px width`

### Model Card Elements
```css
.model-status-badge {
  position: absolute;
  top: 16px;
  right: 16px;
  height: 24px;
  padding: 4px 8px;
  border-radius: 6px;
}

.model-name {
  font-size: 18px;
  line-height: 24px;
  margin-bottom: 8px;
}

.model-version {
  font-size: 14px;
  font-family: monospace;
  margin-bottom: 16px;
}

.accuracy-gauge {
  height: 8px;
  width: 100%;
  border-radius: 4px;
  margin-bottom: 16px;
}

.metrics-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin-bottom: 16px;
}

.action-buttons {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.action-button {
  height: 36px;
  width: 100%;
}
```

---

## 8. USER MANAGEMENT TAB (Admin)

### Container Dimensions
```css
.users-container {
  width: 100%;
  max-width: 1920px;
  margin: 0 auto;
}

.users-header {
  height: 64px;
  margin-bottom: 16px;
}
```

### Users Table
```css
.users-table {
  width: 100%;
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
}

.users-header-row {
  height: 48px;
  background: var(--muted);
}

.user-row {
  height: 64px;
  border-bottom: 1px solid var(--border);
}
```

**Table Column Widths (at 1872px available):**
1. **Name**: `468px` (25%) - includes avatar
2. **Email**: `561px` (30%)
3. **Role**: `280px` (15%)
4. **Status**: `224px` (12%)
5. **Last Login**: `337px` (18%)

### User Avatar
```css
.user-avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  margin-right: 12px;
}
```

---

## 9. AI FEEDBACK TAB (Admin)

### Container Dimensions
```css
.feedback-container {
  width: 100%;
  max-width: 1920px;
  margin: 0 auto;
}

.feedback-header {
  height: 64px;
  margin-bottom: 24px;
}
```

### Feedback Cards Grid
```css
.feedback-grid {
  display: grid;
  grid-template-columns: 1fr; /* mobile */
  gap: 16px;
}

@media (min-width: 768px) {
  .feedback-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (min-width: 1280px) {
  .feedback-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

.feedback-card {
  height: auto; /* ~320px */
  min-height: 300px;
  padding: 16px;
  border-radius: 12px;
  border: 1px solid var(--border);
}

.feedback-thumbnail {
  width: 100%;
  aspect-ratio: 16 / 9;
  border-radius: 8px;
  margin-bottom: 12px;
}

.feedback-textarea {
  width: 100%;
  height: 80px;
  margin-bottom: 12px;
}

.feedback-buttons {
  display: flex;
  gap: 8px;
}

.feedback-button {
  flex: 1;
  height: 36px;
}
```

---

## 10. REPORTS TAB (Officer)

### Container Dimensions
```css
.reports-container {
  width: 100%;
  max-width: 1920px;
  margin: 0 auto;
}

.reports-header {
  height: 64px;
  margin-bottom: 24px;
}
```

### Report Type Cards
```css
.report-types-grid {
  display: grid;
  grid-template-columns: 1fr; /* mobile */
  gap: 16px;
  margin-bottom: 24px;
}

@media (min-width: 768px) {
  .report-types-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

.report-type-card {
  height: 140px;
  padding: 20px;
  border: 2px solid var(--amber-500);
  border-radius: 12px;
  cursor: pointer;
  transition: transform 0.2s;
}

.report-type-card:hover {
  transform: scale(1.02);
}
```

### Reports Filters Bar
```css
.reports-filters {
  height: 56px;
  padding: 12px;
  margin-bottom: 16px;
  border-radius: 12px;
  background: var(--card);
  display: flex;
  gap: 12px;
  align-items: center;
}

.date-picker {
  width: auto;
  height: 40px;
}

.search-input {
  flex: 1;
  height: 40px;
}

.generate-button {
  height: 40px;
  padding: 0 20px;
}
```

### Reports Table
```css
.reports-table {
  width: 100%;
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
}

.reports-header-row {
  height: 48px;
  background: var(--muted);
}

.report-row {
  height: 64px;
  border-bottom: 1px solid var(--border);
}
```

**Table Column Widths (at 1872px):**
1. **Report ID**: `280px` (15%)
2. **Type**: `280px` (15%)
3. **Date Range**: `374px` (20%)
4. **Generated By**: `374px` (20%)
5. **Status**: `280px` (15%)
6. **Actions**: `280px` (15%)

---

## 11. NOTIFICATIONS TAB

### Container Dimensions
```css
.notifications-container {
  width: 100%;
  max-width: 800px; /* centered, narrower container */
  margin: 0 auto;
}

.notifications-header {
  height: 64px;
  margin-bottom: 16px;
}

.filter-tabs {
  height: 40px;
  margin-bottom: 16px;
  display: flex;
  gap: 4px;
}
```

### Notification Cards
```css
.notifications-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.notification-card {
  height: auto; /* ~100px */
  min-height: 90px;
  padding: 16px;
  border-radius: 12px;
  border-left: 4px solid var(--accent-color);
  background: var(--card);
}

.notification-icon {
  width: 32px;
  height: 32px;
  margin-right: 12px;
}

.unread-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--cyan-500);
}
```

---

## Responsive Breakpoints Summary

```css
/* Mobile First Approach */

/* Small devices (portrait phones) */
@media (max-width: 639px) {
  .main-content { padding: 16px; }
  .grid { grid-template-columns: 1fr; }
  .camera-grid { grid-template-columns: repeat(2, 1fr); }
}

/* Medium devices (tablets) */
@media (min-width: 640px) and (max-width: 767px) {
  .main-content { padding: 20px; }
  .grid { grid-template-columns: repeat(2, 1fr); }
}

/* Large devices (laptops) */
@media (min-width: 768px) and (max-width: 1023px) {
  .main-content { padding: 24px; }
  .grid { grid-template-columns: repeat(2, 1fr); }
  .camera-grid { grid-template-columns: repeat(3, 1fr); }
}

/* Extra large devices (desktops) */
@media (min-width: 1024px) and (max-width: 1535px) {
  .main-content { padding: 24px; }
  .grid { grid-template-columns: repeat(3, 1fr); }
  .camera-grid { grid-template-columns: repeat(3, 1fr); }
}

/* 2XL devices (large desktops) */
@media (min-width: 1536px) {
  .main-content { padding: 24px; }
  .grid { grid-template-columns: repeat(4, 1fr); }
  .camera-grid { grid-template-columns: repeat(4, 1fr); }
}
```

---

## Quick Reference Table: Content Area Sizes

| Viewport | Available Width | Available Height | Grid Columns | Card Width |
|----------|----------------|------------------|--------------|------------|
| **1920×1080** | 1872px | 944px | 3-4 cols | 456-618px |
| **1440×900** | 1392px | 764px | 3 cols | 453px |
| **1366×768** | 1318px | 632px | 3 cols | 427px |
| **1024×768** | 976px | 632px | 2-3 cols | 315-472px |
| **768×1024** | 720px | 888px | 2 cols | 352px |
| **375×812** | 343px | 676px | 1 col | 343px |

**Calculation Formula:**
- Available Width = `Viewport Width - (Padding × 2)`
- Available Height = `Viewport Height - Header(112px) - Padding(24px)`
- Card Width (n-col) = `(Available Width - (Gap × (n-1))) / n`

---

## Vite Configuration Notes

### Recommended vite.config.ts settings:
```typescript
export default defineConfig({
  server: {
    port: 3000,
    open: true,
    host: true, // Expose on network for testing
  },
  build: {
    target: 'esnext',
    minify: 'terser',
    cssMinify: true,
  },
  css: {
    devSourcemap: true,
  },
})
```

### Testing Viewports in Dev Tools:
1. **1920×1080** - Primary development
2. **1366×768** - Minimum laptop support
3. **375×812** - Mobile (iPhone X)
4. **768×1024** - Tablet (iPad portrait)

---

## Performance Targets

- **Initial Load**: < 2s
- **Tab Switch**: < 100ms
- **Grid Render**: < 300ms
- **Modal Open**: < 200ms
- **Chart Render**: < 500ms

### Recommended Optimizations:
- Lazy load chart components
- Virtual scrolling for tables with >50 rows
- Memoize camera grid components
- Use CSS transforms for animations (not margin/padding)
- Debounce search inputs (300ms)

---

This specification document provides exact, implementation-ready dimensions for building the Vigil app in a Vite environment!
