# Vigil Application - Component Specifications & Dimensions

This document provides detailed specifications for all tabs, views, and components in the Vigil surveillance system.

---

## Global Layout Specifications

### Screen Layout
- **Application Type**: Full-screen web application
- **Minimum Viewport**: 320px width
- **Optimal Viewport**: 1920px × 1080px (Full HD)
- **Responsive Breakpoints**:
  - `sm`: 640px (Small devices)
  - `md`: 768px (Tablets)
  - `lg`: 1024px (Desktops)
  - `xl`: 1280px (Large desktops)
  - `2xl`: 1536px (Ultra-wide displays)

### Color Scheme
- **Background**: 
  - Light Mode: `#ffffff` (white) with subtle gray tones
  - Dark Mode: `#0a0a0a` (deep charcoal black)
- **Primary Accent**: `#06b6d4` (Cyan 500)
- **Secondary Accent**: `#f59e0b` (Amber 500)
- **Alert Color**: `#ef4444` (Red 500)
- **Success Color**: `#10b981` (Green 500)

---

## 1. HEADER COMPONENT

### Dimensions
- **Height**: 
  - Collapsed: `64px` (4rem padding top/bottom)
  - Expanded (with Quick Stats): `160px`
- **Width**: `100%` full-width
- **Position**: Sticky top, `z-index: 50`

### Header Sections

#### Logo Area
- **Width**: `120px`
- **Height**: `40px`
- **Spacing**: `gap-4` (1rem between logo and role title)

#### System Status Badge
- **Badge Size**: `auto × 24px`
- **Icon Size**: `12px × 12px`
- **Pulse Animation**: Green dot 8px diameter
- **Text**: "All Systems Active" - text-sm (14px)

#### Quick Stats Panel (Collapsible)
- **Trigger Button**: `auto × 32px`
- **Panel Grid**: 
  - Mobile: `grid-cols-2` (2 columns)
  - Desktop: `grid-cols-4` (4 columns)
- **Gap Between Cards**: `12px` (gap-3)
- **Each Stat Card**:
  - Padding: `12px` (p-3)
  - Label: `text-xs` (12px)
  - Value: `text-xl` (20px)

### Navigation Tabs Bar
- **Height**: `48px`
- **Tab Button**:
  - Padding: `px-4 py-2` (16px horizontal, 8px vertical)
  - Border Radius: `8px` (rounded-lg)
  - Active Indicator: `0.5px` height cyan line at bottom
  - Icon Size: `16px × 16px`
  - Font Size: `14px` (text-sm)
  - Gap: `8px` (gap-2)
- **Badge on Tabs**:
  - Height: `20px` (h-5)
  - Padding: `6px` horizontal
  - Animate: Pulse animation for alerts

### User Menu Section
- **Avatar/Name Area**: `auto × 40px`
- **Username**: `text-sm` (14px)
- **Role Label**: `text-xs` (12px)
- **Logout Button**: `32px × 32px`

---

## 2. MAIN CONTENT AREA

### Container
- **Padding**: 
  - Mobile: `16px` (p-4)
  - Desktop: `24px` (p-6)
- **Background**: Inherits from theme (transparent/semi-transparent)
- **Max Content Width**: `100%` (no max-width restriction)

---

## 3. LIVE MONITOR TAB (DVR Camera Grid)

### Control Bar
- **Height**: `40px`
- **Spacing**: `gap-2` (8px)
- **Padding**: `px-2` (8px horizontal)

### Grid Layout Options

#### 2×2 Grid
- **Columns**: 2
- **Rows**: 2
- **Total Cameras**: 4
- **Grid Template**: `grid-cols-2`
- **Camera Aspect Ratio**: `16:9`
- **Gap Between Cameras**: `8px` (gap-2)

#### 2×3 Grid (Default)
- **Columns**: 
  - Mobile: 2 columns
  - Desktop: 3 columns
- **Rows**: 2 (mobile), 2 (desktop)
- **Total Cameras**: 6
- **Grid Template**: `grid-cols-2 md:grid-cols-3`
- **Camera Aspect Ratio**: `16:9`
- **Gap Between Cameras**: `8px` (gap-2)

#### 3×3 Grid
- **Columns**: 
  - Mobile: 2 columns
  - Desktop: 3 columns
- **Rows**: 3
- **Total Cameras**: 9
- **Grid Template**: `grid-cols-2 md:grid-cols-3`
- **Camera Aspect Ratio**: `16:9`
- **Gap Between Cameras**: `8px` (gap-2)

#### 4×4 Grid
- **Columns**: 
  - Mobile: 2 columns
  - Desktop: 4 columns
- **Rows**: 4
- **Total Cameras**: 16
- **Grid Template**: `grid-cols-2 md:grid-cols-4`
- **Camera Aspect Ratio**: `16:9`
- **Gap Between Cameras**: `8px` (gap-2)

### Individual Camera Card
- **Border Radius**: `12px` (rounded-xl)
- **Border Width**: `2px`
- **Border Color**: 
  - Normal: `border-border` (theme border)
  - Alert: `border-red-500` (red border with animation)
- **Padding**: `0` (video fills card)
- **Background**: Black (#000000) placeholder
- **Overlay Elements**:
  - **Top-left Badge** (Camera ID):
    - Size: `auto × 24px`
    - Font: `text-xs` (12px)
    - Position: `top-2 left-2`
  - **Top-right Status**:
    - Icon: `12px × 12px`
    - Position: `top-2 right-2`
  - **Bottom Overlay** (Camera Info):
    - Height: `auto`
    - Padding: `8px` (p-2)
    - Background: Semi-transparent black `bg-black/60`
    - Backdrop Blur: `backdrop-blur-sm`
    - Camera Name: `text-sm` (14px)
    - Location: `text-xs` (12px)

### Alert Animation
- **Border Pulse**: Red border with pulse effect
- **Alert Badge**:
  - Position: `top-1/2 left-1/2` (centered)
  - Transform: `translate(-50%, -50%)`
  - Size: `auto × auto`
  - Icon: `48px × 48px`
  - Background: Red with opacity
  - Animation: Pulse

---

## 4. INCIDENTS TAB

### Page Layout
- **Header Section**:
  - Title: `text-2xl` (24px) font-bold
  - Description: `text-sm` (14px) muted
  - Spacing: `mb-6` (24px margin bottom)

### Filter Tabs
- **Container Height**: `40px`
- **Tab Button**:
  - Padding: `px-4 py-2` (16px × 8px)
  - Border Radius: `8px`
  - Active: Gradient background cyan-to-blue
  - Font: `text-sm` (14px)
  - Gap: `4px` (gap-1)

### Incident Cards Grid
- **Grid Layout**:
  - Mobile: `grid-cols-1` (1 column)
  - Tablet: `grid-cols-2` (2 columns)
  - Desktop: `grid-cols-3` (3 columns)
- **Gap**: `16px` (gap-4)
- **Card Dimensions**:
  - Width: `100%` (fills grid cell)
  - Height: `auto` (content-based)
  - Padding: `20px` (p-5)
  - Border Radius: `12px` (rounded-xl)
  - Border: `1px` solid border-color
  - Box Shadow: Subtle shadow

### Individual Incident Card
- **Header Row**:
  - Height: `auto`
  - Flex: Space-between alignment
  - **Incident ID Badge**:
    - Font: `text-xs` (12px) font-mono
    - Padding: `4px 8px`
    - Border Radius: `6px`
  - **Severity Badge**:
    - Size: `auto × 20px`
    - Font: `text-xs` (12px)
    - Colors: Critical=Red, High=Orange, Medium=Amber, Low=Blue

- **Content Section**:
  - **Type Icon**: `40px × 40px`
  - **Description**: `text-base` (16px)
  - **Location**: `text-sm` (14px) with MapPin icon
  - **Camera ID**: `text-sm` (14px) with Camera icon
  - **Timestamp**: `text-xs` (12px) muted
  - **Confidence Bar**:
    - Height: `8px`
    - Width: `100%`
    - Border Radius: `4px`
    - Fill Color: Based on confidence level

- **Action Buttons**:
  - Container: Flex row with gap-2 (8px)
  - Button Height: `36px`
  - Button Padding: `px-3 py-2`
  - Font: `text-sm` (14px)
  - Icons: `16px × 16px`
  - View Details: Primary button (full width on mobile)
  - Resolve: Success color (green)
  - Dismiss: Destructive color (red)

---

## 5. INCIDENT DETAIL MODAL

### Modal Dimensions
- **Width**: `98vw` (98% of viewport width)
- **Height**: `98vh` (98% of viewport height)
- **Max Width**: `none` (no max-width restriction)
- **Border Radius**: `16px` (rounded-2xl)
- **Position**: Centered on screen
- **Backdrop**: Semi-transparent black overlay
- **z-index**: `50`

### Modal Layout Sections

#### Header Section
- **Height**: `auto` (~80px)
- **Padding**: `24px` (p-6)
- **Border Bottom**: `1px` solid border-color
- **Elements**:
  - Incident ID: `text-2xl` (24px) font-bold
  - Type Badge: `auto × 28px`
  - Severity Badge: `auto × 28px`
  - Close Button: `32px × 32px` (top-right)

#### Main Content Area
- **Layout**: Two-column grid
  - Left Column: Video player (60% width)
  - Right Column: Details tabs (40% width)
- **Grid Template**: `grid-cols-1 lg:grid-cols-[1.5fr,1fr]`
- **Gap**: `24px` (gap-6)
- **Padding**: `24px` (p-6)
- **Max Height**: `calc(98vh - 200px)` (subtract header and footer)
- **Overflow**: Auto scroll

### Video Player Section (Left Column)
- **Aspect Ratio**: `16:9`
- **Background**: Black (#000000)
- **Border Radius**: `12px`
- **Video Container Height**: `auto` (maintains aspect ratio)
- **Overlay Elements**:
  - Play/Pause Overlay: Centered, `80px × 80px` icon
  - Timestamp: Top-left, `text-sm` (14px)
  - Camera ID: Top-right, `text-sm` (14px)

#### Video Controls Bar
- **Height**: `56px`
- **Padding**: `12px` (p-3)
- **Background**: Semi-transparent dark `bg-black/80`
- **Border Radius**: `8px` (bottom of video)
- **Elements**:
  - **Play/Pause Button**: `40px × 40px`
  - **Skip Buttons**: `32px × 32px`
  - **Progress Bar**:
    - Height: `6px`
    - Width: `100%`
    - Border Radius: `3px`
    - Fill Color: Cyan
  - **Time Display**: `text-xs` (12px)
  - **Speed Selector**: `auto × 32px` dropdown
  - **Volume Toggle**: `32px × 32px`
  - **Download Button**: `32px × 32px`
  - **Fullscreen Button**: `32px × 32px`

### Details Tabs Section (Right Column)
- **Tab List Height**: `44px`
- **Tab Button**:
  - Padding: `px-4 py-2` (16px × 8px)
  - Font: `text-sm` (14px)
  - Border Radius: `8px 8px 0 0`
  - Active: Cyan gradient background

#### Tab 1: Video Evidence
- **Thumbnail Grid**: `grid-cols-2`
- **Gap**: `12px` (gap-3)
- **Each Thumbnail**:
  - Aspect Ratio: `16:9`
  - Border Radius: `8px`
  - Border: `2px` solid
  - Hover: Scale 1.05
  - Overlay Badge: Timestamp `text-xs` (12px)

#### Tab 2: Details
- **Info Rows**: Stacked vertically with `12px` gap
- **Each Info Row**:
  - Icon: `20px × 20px`
  - Label: `text-sm` (14px) muted
  - Value: `text-base` (16px) bold
  - Spacing: `gap-3` (12px)

#### Tab 3: Nearby Cameras
- **Camera List**: Stacked cards
- **Gap**: `12px` (gap-3)
- **Each Camera Card**:
  - Padding: `12px` (p-3)
  - Border Radius: `8px`
  - Border: `1px` solid
  - Hover: Cyan border
  - Camera ID: `text-sm` (14px) font-mono
  - Location: `text-sm` (14px)
  - Distance: `text-xs` (12px) badge
  - Thumbnail: `80px × 45px` (16:9 mini)

#### Tab 4: Activity Log
- **Timeline Container**: Vertical list
- **Gap**: `16px` (gap-4)
- **Each Log Entry**:
  - Left Border: `2px` colored line
  - Padding Left: `16px` (pl-4)
  - Dot Indicator: `12px × 12px` circle
  - Timestamp: `text-xs` (12px) muted
  - Action: `text-sm` (14px)
  - User: `text-xs` (12px) badge

### Action Buttons (Footer)
- **Container Height**: `72px`
- **Padding**: `16px` (p-4)
- **Border Top**: `1px` solid border-color
- **Button Sizes**: 
  - Primary (Resolve/Dismiss): `auto × 40px`
  - Padding: `px-6 py-2.5`
  - Font: `text-sm` (14px)
  - Gap: `12px` (gap-3)

---

## 6. ANALYTICS TAB

### Page Layout
- **Header Section**:
  - Title: `text-2xl` (24px) font-bold
  - Time Range Selector: `width: 192px` (w-48)
  - Spacing: `mb-6` (24px)

### Key Metrics Cards
- **Grid Layout**: `grid-cols-1 md:grid-cols-2 lg:grid-cols-4`
- **Gap**: `16px` (gap-4)
- **Each Metric Card**:
  - Width: `100%` (fills grid cell)
  - Height: `auto` (~140px)
  - Padding: `24px` (p-6)
  - Border Radius: `12px`
  - Gradient Background: Based on metric type
  - Border: `1px` with opacity
  - **Icon**: `32px × 32px` (w-8 h-8)
  - **Value**: `text-3xl` (30px)
  - **Label**: `text-sm` (14px) muted
  - **Badge**: `auto × 20px` with trend icon

### Chart Cards
- **Grid Layout**: `grid-cols-1 lg:grid-cols-2`
- **Gap**: `24px` (gap-6)
- **Each Chart Card**:
  - Width: `100%`
  - Height: `auto`
  - Padding: `24px` (p-6)
  - Border Radius: `12px`
  - Background: Card with glassmorphism
  - Border: `1px` solid border-color

#### Chart Specifications
- **Chart Height**: `300px`
- **Chart Width**: `100%` (ResponsiveContainer)
- **Font Sizes**:
  - Axis Labels: `12px`
  - Legend: `12px`
  - Tooltip: `14px`
- **Colors**:
  - Violence: `#ef4444` (Red)
  - Crash: `#f59e0b` (Amber)
  - Accuracy: `#10b981` (Green)
  - Response Time: `#06b6d4` (Cyan)

### Alert Resolution Status
- **Card Width**: `100%` (full-width below charts)
- **Progress Bars**:
  - Height: `12px` (h-3)
  - Border Radius: `9999px` (rounded-full)
  - Gap between bars: `16px` (gap-4)
  - Label Row Height: `24px`
  - Label Font: `text-sm` (14px)
  - Value Font: `text-sm` (14px) muted

---

## 7. MAP VIEW TAB

### Map Container
- **Width**: `100%`
- **Height**: `calc(100vh - 200px)` (full viewport minus header)
- **Min Height**: `600px`
- **Border Radius**: `12px`
- **Background**: Map tiles or placeholder

### Map Markers
- **Camera Marker**:
  - Size: `32px × 32px`
  - Icon: Camera icon
  - Color: Cyan (online), Red (offline), Amber (maintenance)
  - Pulse Animation: For cameras with alerts

- **Incident Marker**:
  - Size: `40px × 40px`
  - Icon: Alert triangle
  - Color: Red (critical), Orange (high), Amber (medium)
  - Badge: Incident count if multiple at location

### Map Popup/Tooltip
- **Width**: `280px`
- **Max Width**: `90vw` (mobile responsive)
- **Padding**: `16px` (p-4)
- **Border Radius**: `12px`
- **Elements**:
  - Title: `text-base` (16px) font-bold
  - Details: `text-sm` (14px)
  - Status Badge: `auto × 20px`
  - Action Button: `100% × 36px`

### Map Legend
- **Position**: Bottom-left
- **Width**: `200px`
- **Padding**: `12px` (p-3)
- **Background**: Semi-transparent card
- **Backdrop Blur**: Yes
- **Item Height**: `28px`
- **Icon Size**: `16px × 16px`
- **Font**: `text-xs` (12px)

---

## 8. CAMERA MANAGEMENT TAB (Admin Only)

### Page Layout
- **Header**: Title + Add Camera button
- **Search Bar**: `100% × 40px`, margin-bottom: `16px`
- **Filter Pills**: `auto × 32px`, gap: `8px`

### Camera Table
- **Width**: `100%`
- **Row Height**: `60px`
- **Header Row Height**: `48px`
- **Header Font**: `text-sm` (14px) font-medium
- **Cell Font**: `text-sm` (14px)
- **Cell Padding**: `12px` (px-3 py-4)

#### Table Columns
1. **Camera ID**: `width: 15%`, font-mono
2. **Name**: `width: 25%`
3. **Location**: `width: 25%`
4. **Status**: `width: 12%`, badge
5. **Zone**: `width: 13%`, badge
6. **Actions**: `width: 10%`, icon buttons

### Camera Card View (Mobile)
- **Card Height**: `auto`
- **Padding**: `16px` (p-4)
- **Border Radius**: `12px`
- **Gap Between Cards**: `12px` (gap-3)
- **Thumbnail**: `80px × 45px`
- **Info Section**: Stacked text with icons

### Add/Edit Camera Modal
- **Width**: `600px`
- **Max Width**: `95vw`
- **Padding**: `24px` (p-6)
- **Form Field Height**: `40px`
- **Label Font**: `text-sm` (14px)
- **Input Font**: `text-base` (16px)
- **Field Gap**: `16px` (gap-4)

---

## 9. USER MANAGEMENT TAB (Admin Only)

### User Table
- **Width**: `100%`
- **Row Height**: `64px`
- **Header Row Height**: `48px`

#### Table Columns
1. **Name**: `width: 25%`
2. **Email**: `width: 30%`
3. **Role**: `width: 15%`, badge
4. **Status**: `width: 12%`, badge
5. **Last Login**: `width: 18%`, text-sm muted

### User Avatar
- **Size**: `40px × 40px`
- **Border Radius**: `50%` (rounded-full)
- **Fallback**: Initials with colored background

### Add/Edit User Modal
- **Width**: `600px`
- **Max Width**: `95vw`
- **Form Layout**: Stacked fields with `16px` gap
- **Field Height**: `40px`
- **Select Dropdown Height**: `auto` (max 200px scroll)

---

## 10. AI MODEL MANAGEMENT TAB (Admin Only)

### Model Cards Grid
- **Grid Layout**: `grid-cols-1 md:grid-cols-2 lg:grid-cols-3`
- **Gap**: `16px` (gap-4)
- **Card Dimensions**:
  - Width: `100%`
  - Height: `auto` (~240px)
  - Padding: `20px` (p-5)
  - Border Radius: `12px`

### Model Card Elements
- **Status Badge**: Top-right, `auto × 24px`
- **Model Name**: `text-lg` (18px) font-bold
- **Version**: `text-sm` (14px) font-mono
- **Accuracy Gauge**:
  - Height: `8px`
  - Width: `100%`
  - Border Radius: `4px`
  - Percentage Label: `text-2xl` (24px) bold
- **Metrics Row**: 
  - Grid: `grid-cols-3`
  - Each Metric: `text-center`
  - Value: `text-base` (16px)
  - Label: `text-xs` (12px) muted
- **Action Buttons**: 
  - Button: `100% × 36px`
  - Gap: `8px` (gap-2)

### Deploy Model Modal
- **Width**: `500px`
- **Max Width**: `95vw`
- **Confirmation Text**: `text-base` (16px)
- **Warning Badge**: Full-width, `auto × auto`, padding: `12px`

---

## 11. SYSTEM HEALTH TAB (Admin Only)

### Health Metrics Cards
- **Grid Layout**: `grid-cols-1 md:grid-cols-2 lg:grid-cols-3`
- **Gap**: `16px` (gap-4)
- **Card Height**: `auto` (~160px)
- **Status Indicator**:
  - Size: `12px × 12px` circle
  - Animation: Pulse for healthy systems
  - Colors: Green (healthy), Amber (warning), Red (critical)

### System Logs Table
- **Width**: `100%`
- **Max Height**: `400px`
- **Overflow**: Auto scroll
- **Row Height**: `48px`
- **Log Level Badges**: 
  - Size: `auto × 20px`
  - Colors: Info=Blue, Warning=Amber, Error=Red, Success=Green
- **Timestamp**: `text-xs` (12px) font-mono

### Resource Usage Charts
- **Chart Height**: `200px`
- **Chart Width**: `100%`
- **Real-time Update**: Every 5 seconds
- **Data Points Shown**: Last 20 measurements

---

## 12. AI FEEDBACK TAB (Admin Only)

### Feedback Cards Grid
- **Grid Layout**: `grid-cols-1 md:grid-cols-2 xl:grid-cols-3`
- **Gap**: `16px` (gap-4)
- **Card Dimensions**:
  - Width: `100%`
  - Height: `auto` (~320px)
  - Padding: `16px` (p-4)

### Feedback Card Elements
- **Video Thumbnail**: 
  - Aspect Ratio: `16:9`
  - Border Radius: `8px`
  - Height: `auto`
- **Detection Badge**: 
  - Position: Overlay on thumbnail top-right
  - Size: `auto × 24px`
  - Confidence: Displayed as percentage
- **Feedback Section**:
  - Textarea Height: `80px`
  - Button Row: Flex gap-2
  - Approve Button: `50% × 36px` green
  - Reject Button: `50% × 36px` red

---

## 13. REPORTS TAB (Officer Dashboard)

### Report Type Cards
- **Grid Layout**: `grid-cols-1 md:grid-cols-3`
- **Gap**: `16px` (gap-4)
- **Card Dimensions**:
  - Width: `100%`
  - Height: `auto` (~140px)
  - Padding: `20px` (p-5)
  - Border: `2px` solid
  - Border Color: Amber/golden accent
  - Hover: Scale 1.02

### Report Filters Bar
- **Height**: `56px`
- **Padding**: `12px` (p-3)
- **Background**: Card background
- **Border Radius**: `12px`
- **Elements**:
  - Date Pickers: `auto × 40px`
  - Status Filter: `auto × 40px`
  - Search: `flex-1 × 40px`
  - Generate Button: `auto × 40px` amber accent

### Reports Table
- **Width**: `100%`
- **Row Height**: `64px`
- **Header Height**: `48px`
- **Columns**:
  - Report ID: `15%` font-mono
  - Type: `15%` badge
  - Date Range: `20%`
  - Generated By: `20%`
  - Status: `15%` badge (amber/golden for complete)
  - Actions: `15%` (Download + View icons)

---

## 14. NOTIFICATIONS TAB

### Notification List
- **Width**: `100%`
- **Max Width**: `800px` (centered)
- **Gap Between Items**: `12px` (gap-3)

### Notification Card
- **Height**: `auto` (~100px)
- **Padding**: `16px` (p-4)
- **Border Radius**: `12px`
- **Border Left**: `4px` colored bar (type-based)
- **Elements**:
  - Icon: `32px × 32px`
  - Title: `text-base` (16px) font-bold
  - Message: `text-sm` (14px)
  - Timestamp: `text-xs` (12px) muted
  - Action Buttons: `auto × 32px`
  - Unread Indicator: `8px × 8px` cyan dot

### Filter Tabs
- **Height**: `40px`
- **Tab Width**: `auto`
- **Padding**: `px-4 py-2`
- **Gap**: `4px`

---

## 15. RESPONSIVE BEHAVIOR

### Mobile (< 640px)
- **Header**: Simplified, logo smaller (100px width)
- **Navigation Tabs**: Horizontal scroll, touch-friendly
- **Camera Grid**: 2 columns max
- **Incident Cards**: 1 column
- **Modal**: `95vw × 95vh`, single column layout
- **Tables**: Convert to card view
- **Charts**: Full-width, reduced height (250px)

### Tablet (640px - 1024px)
- **Camera Grid**: 2-3 columns
- **Incident Cards**: 2 columns
- **Analytics Grid**: 2 columns
- **Modal**: 2-column layout maintained
- **Navigation**: Full tab bar visible

### Desktop (> 1024px)
- **Full Layout**: All features visible
- **Camera Grid**: Up to 4 columns
- **Incident Cards**: 3 columns
- **Analytics**: 4-column metrics, 2-column charts
- **Modal**: Optimized 60/40 split layout
- **Hover Effects**: Enabled for all interactive elements

---

## 16. ANIMATION SPECIFICATIONS

### Transitions
- **Duration**: `200ms` (fast), `300ms` (normal), `500ms` (slow)
- **Easing**: `ease-in-out` (default), `cubic-bezier(0.4, 0, 0.2, 1)` (smooth)

### Animations
- **Pulse**: Alert badges, status indicators, live badges
- **Slide In**: Modals, panels (from top/right)
- **Fade In/Out**: Tooltips, notifications, overlays
- **Scale**: Hover effects on cards and buttons (1.02-1.05)
- **Shimmer**: Loading states, skeleton screens

### Loading States
- **Skeleton Height**: Matches content height
- **Skeleton Border Radius**: Matches content radius
- **Animation**: Shimmer left-to-right, 1.5s duration

---

## 17. TYPOGRAPHY

### Font Families
- **Primary**: `Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`
- **Monospace**: `'Fira Code', 'Courier New', monospace` (for IDs, codes)
- **Display**: `Urbanist, Poppins` (optional for headings)

### Font Sizes
- **xs**: `12px` / `0.75rem`
- **sm**: `14px` / `0.875rem`
- **base**: `16px` / `1rem`
- **lg**: `18px` / `1.125rem`
- **xl**: `20px` / `1.25rem`
- **2xl**: `24px` / `1.5rem`
- **3xl**: `30px` / `1.875rem`

### Font Weights
- **Normal**: `400`
- **Medium**: `500`
- **Semibold**: `600`
- **Bold**: `700`

### Line Heights
- **Tight**: `1.25`
- **Normal**: `1.5`
- **Relaxed**: `1.75`

---

## 18. SPACING SYSTEM

Based on Tailwind's 4px base unit:

- **0**: `0px`
- **1**: `4px` / `0.25rem`
- **2**: `8px` / `0.5rem`
- **3**: `12px` / `0.75rem`
- **4**: `16px` / `1rem`
- **5**: `20px` / `1.25rem`
- **6**: `24px` / `1.5rem`
- **8**: `32px` / `2rem`
- **10**: `40px` / `2.5rem`
- **12**: `48px` / `3rem`
- **16**: `64px` / `4rem`
- **20**: `80px` / `5rem`

---

## 19. BORDER RADII

- **sm**: `4px` / `0.25rem`
- **DEFAULT**: `6px` / `0.375rem`
- **md**: `8px` / `0.5rem`
- **lg**: `12px` / `0.75rem`
- **xl**: `16px` / `1rem`
- **2xl**: `20px` / `1.25rem`
- **3xl**: `24px` / `1.5rem`
- **full**: `9999px` (perfect circle/pill)

---

## 20. Z-INDEX LAYERS

- **Base**: `0` (normal content)
- **Dropdown**: `10`
- **Sticky**: `20`
- **Fixed**: `30`
- **Modal Backdrop**: `40`
- **Modal Content**: `50`
- **Popover**: `60`
- **Tooltip**: `70`
- **Toast/Notification**: `80`

---

## Summary Table: All Tabs & Key Dimensions

| Tab Name | Grid Layout | Card Size | Content Height | Special Features |
|----------|------------|-----------|----------------|------------------|
| **Live Monitor** | 2×2, 2×3, 3×3, 4×4 | 16:9 aspect | Auto | Real-time video feeds |
| **Incidents** | 1-2-3 cols responsive | ~280px height | Auto scroll | Live incident alerts |
| **Map View** | Full-width | N/A | calc(100vh - 200px) | Interactive markers |
| **Analytics** | 1-2-4 cols responsive | Various | 300px charts | Real-time metrics |
| **Cameras** | Table/Cards | 60px row / auto card | Auto scroll | CRUD operations |
| **Users** | Table | 64px row | Auto scroll | Role management |
| **System Health** | 1-2-3 cols | ~160px height | Auto | Real-time monitoring |
| **AI Models** | 1-2-3 cols | ~240px height | Auto | Version control |
| **AI Feedback** | 1-2-3 cols | ~320px height | Auto scroll | Review & approve |
| **Reports** | Table + filters | 64px row | Auto scroll | PDF generation |
| **Notifications** | Single column | ~100px height | Auto scroll | Real-time alerts |

---

This comprehensive specification document ensures consistent design implementation across all components and provides clear guidelines for responsive behavior, animations, and user interactions.
