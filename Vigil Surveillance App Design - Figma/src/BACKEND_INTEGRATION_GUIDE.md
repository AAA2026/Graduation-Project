# Backend Integration Guide

## Overview

This guide shows you how to connect Vigil (frontend) to your backend API for real data, authentication, and persistence.

Currently, Vigil uses:
- ✅ **Mock/Demo Data** - Simulated incidents, cameras, analytics
- ✅ **Local State** - React useState/useEffect
- ✅ **No Persistence** - Data resets on refresh

After Backend Integration:
- ✅ **Real Data** - From your database
- ✅ **User Authentication** - Login with real accounts
- ✅ **Data Persistence** - Saves across sessions
- ✅ **Real-time Updates** - WebSocket or polling
- ✅ **AI Model Integration** - Connect to actual AI services

---

## Architecture Overview

### Current Frontend-Only Architecture

```
┌─────────────────────┐
│   React Frontend    │
│   (Vigil App)       │
│                     │
│  • Mock data        │
│  • Local state      │
│  • No API calls     │
└─────────────────────┘
```

### After Backend Integration

```
┌─────────────────────┐         ┌─────────────────────┐
│   React Frontend    │────────▶│   Backend API       │
│   (Vigil App)       │         │   (Your Server)     │
│                     │◀────────│                     │
│  • Real data        │         │  • Authentication   │
│  • API calls        │         │  • Database         │
│  • WebSockets       │         │  • AI Services      │
└─────────────────────┘         └─────────────────────┘
                                          │
                                          ▼
                                ┌─────────────────────┐
                                │    Database         │
                                │    (MongoDB/SQL)    │
                                └─────────────────────┘
                                          │
                                          ▼
                                ┌─────────────────────┐
                                │   AI Models         │
                                │   (Detection APIs)  │
                                └─────────────────────┘
```

---

## Backend Options

### Option 1: Node.js + Express (Recommended)

**Pros:**
- Same language as frontend (JavaScript)
- Easy to share code/types
- Large ecosystem (npm packages)
- Good for real-time (Socket.io)

**Cons:**
- Need to manage server hosting
- Scaling requires setup

**Tech Stack:**
```
- Runtime: Node.js
- Framework: Express.js
- Database: MongoDB (Mongoose) or PostgreSQL (Sequelize)
- Real-time: Socket.io
- Authentication: JWT (jsonwebtoken)
```

---

### Option 2: Python + Flask/FastAPI

**Pros:**
- Great for AI/ML integration
- Easy to connect Python AI models
- Simple REST API setup

**Cons:**
- Different language from frontend
- Less real-time support

**Tech Stack:**
```
- Runtime: Python 3.10+
- Framework: FastAPI or Flask
- Database: PostgreSQL (SQLAlchemy)
- Real-time: WebSockets (FastAPI) or polling
- Authentication: JWT
```

---

### Option 3: Backend-as-a-Service (BaaS)

**Firebase (Google)**
- Pros: No server management, real-time database, authentication built-in
- Cons: Vendor lock-in, pricing at scale

**Supabase (Open Source Firebase Alternative)**
- Pros: PostgreSQL-based, open source, great real-time features
- Cons: Less mature than Firebase

**AWS Amplify**
- Pros: Full AWS integration, scalable
- Cons: Complex setup, steeper learning curve

---

## Step-by-Step Integration

### Phase 1: Authentication

#### Current (Demo Login)

File: `/src/components/LoginScreen.jsx`

```javascript
const handleLogin = (role) => {
  // Demo - no actual verification
  if (email && password) {
    onLogin(role);
  }
};
```

#### After Backend Integration

```javascript
const handleLogin = async (role) => {
  try {
    // Call your API
    const response = await fetch('https://your-api.com/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, role })
    });

    const data = await response.json();
    
    if (response.ok) {
      // Store JWT token
      localStorage.setItem('authToken', data.token);
      localStorage.setItem('userRole', data.role);
      
      // Call parent callback
      onLogin(data.role);
    } else {
      toast.error(data.message || 'Login failed');
    }
  } catch (error) {
    toast.error('Network error. Please try again.');
  }
};
```

#### Backend API Endpoint Example

```javascript
// Node.js + Express
app.post('/auth/login', async (req, res) => {
  const { email, password, role } = req.body;
  
  // Find user in database
  const user = await User.findOne({ email, role });
  
  if (!user) {
    return res.status(401).json({ message: 'Invalid credentials' });
  }
  
  // Verify password (use bcrypt)
  const isValid = await bcrypt.compare(password, user.hashedPassword);
  
  if (!isValid) {
    return res.status(401).json({ message: 'Invalid credentials' });
  }
  
  // Generate JWT token
  const token = jwt.sign(
    { userId: user._id, role: user.role },
    process.env.JWT_SECRET,
    { expiresIn: '24h' }
  );
  
  res.json({ token, role: user.role, user: user.name });
});
```

---

### Phase 2: Fetch Real Incidents

#### Current (Mock Data)

File: `/src/components/hooks/useRealtimeIncidents.jsx`

```javascript
// Simulated incidents
useEffect(() => {
  const interval = setInterval(() => {
    // Create fake incident
    const newIncident = {
      id: Date.now(),
      type: 'violence',
      // ... mock data
    };
    setIncidents(prev => [newIncident, ...prev]);
  }, 8000);
}, []);
```

#### After Backend Integration (REST API)

```javascript
import { useState, useEffect } from 'react';

export function useRealtimeIncidents() {
  const [incidents, setIncidents] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch initial incidents
    fetchIncidents();

    // Poll for new incidents every 5 seconds
    const interval = setInterval(fetchIncidents, 5000);

    return () => clearInterval(interval);
  }, []);

  const fetchIncidents = async () => {
    try {
      const token = localStorage.getItem('authToken');
      
      const response = await fetch('https://your-api.com/incidents', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setIncidents(data.incidents);
      }
    } catch (error) {
      console.error('Error fetching incidents:', error);
    } finally {
      setLoading(false);
    }
  };

  return { incidents, loading };
}
```

#### After Backend Integration (WebSocket - Real-time)

```javascript
import { useState, useEffect } from 'react';
import io from 'socket.io-client';

export function useRealtimeIncidents() {
  const [incidents, setIncidents] = useState([]);
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    // Connect to WebSocket server
    const token = localStorage.getItem('authToken');
    const newSocket = io('https://your-api.com', {
      auth: { token }
    });

    setSocket(newSocket);

    // Listen for new incidents
    newSocket.on('new_incident', (incident) => {
      setIncidents(prev => [incident, ...prev]);
    });

    // Fetch initial incidents
    newSocket.emit('get_incidents', (data) => {
      setIncidents(data.incidents);
    });

    return () => newSocket.disconnect();
  }, []);

  return { incidents, socket };
}
```

#### Backend WebSocket Server Example

```javascript
// Node.js + Socket.io
const io = require('socket.io')(server, {
  cors: { origin: 'http://localhost:3000' }
});

io.use((socket, next) => {
  // Verify JWT token
  const token = socket.handshake.auth.token;
  // ... verify token
  next();
});

io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);

  // Send incidents on request
  socket.on('get_incidents', async (callback) => {
    const incidents = await Incident.find().sort('-createdAt').limit(50);
    callback({ incidents });
  });

  // Broadcast new incident to all connected clients
  function broadcastIncident(incident) {
    io.emit('new_incident', incident);
  }
});

// When AI detects incident
function onAIDetection(incident) {
  // Save to database
  Incident.create(incident);
  
  // Broadcast to all clients
  io.emit('new_incident', incident);
}
```

---

### Phase 3: Camera Feeds

#### Current (Mock Streams)

File: `/src/../components/DVRCameraGrid.jsx`

```javascript
const cameras = Array.from({ length: 9 }, (_, i) => ({
  id: i + 1,
  streamUrl: `https://images.unsplash.com/photo-...`,
  status: 'online',
  // ... mock data
}));
```

#### After Backend Integration

```javascript
const [cameras, setCameras] = useState([]);

useEffect(() => {
  fetchCameras();
}, []);

const fetchCameras = async () => {
  try {
    const token = localStorage.getItem('authToken');
    const response = await fetch('https://your-api.com/cameras', {
      headers: { 'Authorization': `Bearer ${token}` }
    });

    const data = await response.json();
    setCameras(data.cameras);
  } catch (error) {
    console.error('Error fetching cameras:', error);
  }
};
```

#### Backend API Response

```json
{
  "cameras": [
    {
      "id": 1,
      "name": "Main Entrance",
      "location": "Building A",
      "streamUrl": "rtsp://camera1.local:554/stream",
      "hlsUrl": "https://your-cdn.com/camera1/playlist.m3u8",
      "status": "online",
      "aiModel": {
        "type": "violence",
        "accuracy": 0.95,
        "enabled": true
      },
      "resolution": "1920x1080",
      "fps": 30
    }
  ]
}
```

**For Live Streaming:**

Use HLS (HTTP Live Streaming) for web compatibility:

```javascript
import Hls from 'hls.js';

useEffect(() => {
  if (camera.hlsUrl && videoRef.current) {
    const hls = new Hls();
    hls.loadSource(camera.hlsUrl);
    hls.attachMedia(videoRef.current);
  }
}, [camera]);

return <video ref={videoRef} autoPlay muted />;
```

---

### Phase 4: Analytics Data

#### Current (Mock Data)

File: `/src/../components/AnalyticsView.jsx`

```javascript
const mockData = [
  { date: 'Mon', incidents: 12 },
  { date: 'Tue', incidents: 19 },
  // ... hardcoded
];
```

#### After Backend Integration

```javascript
const [analyticsData, setAnalyticsData] = useState(null);

useEffect(() => {
  fetchAnalytics();
}, [dateRange]); // Re-fetch when date range changes

const fetchAnalytics = async () => {
  try {
    const token = localStorage.getItem('authToken');
    const response = await fetch(
      `https://your-api.com/analytics?start=${dateRange.start}&end=${dateRange.end}`,
      { headers: { 'Authorization': `Bearer ${token}` } }
    );

    const data = await response.json();
    setAnalyticsData(data);
  } catch (error) {
    console.error('Error fetching analytics:', error);
  }
};
```

#### Backend API Response

```json
{
  "timeRange": {
    "start": "2026-01-01",
    "end": "2026-01-11"
  },
  "summary": {
    "totalIncidents": 156,
    "violenceDetections": 23,
    "carCrashes": 8,
    "peopleCounted": 15420
  },
  "dailyIncidents": [
    { "date": "2026-01-01", "violence": 2, "crashes": 1, "people": 1250 },
    { "date": "2026-01-02", "violence": 3, "crashes": 0, "people": 1180 }
  ],
  "topLocations": [
    { "location": "Main Entrance", "count": 34 },
    { "location": "Parking Lot", "count": 28 }
  ],
  "hourlyDistribution": [...]
}
```

---

### Phase 5: AI Model Configuration

#### After Backend Integration

File: `/src/../components/AIModelManagement.jsx`

```javascript
const [models, setModels] = useState([]);

// Fetch AI models from backend
useEffect(() => {
  fetchModels();
}, []);

const fetchModels = async () => {
  const response = await fetch('https://your-api.com/ai-models', {
    headers: { 'Authorization': `Bearer ${localStorage.getItem('authToken')}` }
  });
  const data = await response.json();
  setModels(data.models);
};

// Toggle model on/off
const toggleModel = async (modelId) => {
  await fetch(`https://your-api.com/ai-models/${modelId}/toggle`, {
    method: 'PATCH',
    headers: {
      'Authorization': `Bearer ${localStorage.getItem('authToken')}`,
      'Content-Type': 'application/json'
    }
  });
  
  fetchModels(); // Refresh
};
```

---

## API Endpoint Structure

### Recommended Backend Routes

```
Authentication
  POST   /auth/login           - User login
  POST   /auth/logout          - User logout
  POST   /auth/refresh-token   - Refresh JWT
  GET    /auth/me              - Get current user

Incidents
  GET    /incidents            - List all incidents
  GET    /incidents/:id        - Get single incident
  POST   /incidents            - Create incident (AI webhook)
  PATCH  /incidents/:id        - Update incident status
  DELETE /incidents/:id        - Delete incident

Cameras
  GET    /cameras              - List all cameras
  GET    /cameras/:id          - Get single camera
  POST   /cameras              - Add new camera
  PATCH  /cameras/:id          - Update camera settings
  DELETE /cameras/:id          - Remove camera
  GET    /cameras/:id/stream   - Get camera stream URL

AI Models
  GET    /ai-models            - List configured models
  GET    /ai-models/:id        - Get model details
  PATCH  /ai-models/:id        - Update model settings
  POST   /ai-models/:id/train  - Trigger model training

Analytics
  GET    /analytics            - Get analytics data
  GET    /analytics/export     - Export report (PDF/CSV)

Users
  GET    /users                - List users (admin only)
  POST   /users                - Create user
  PATCH  /users/:id            - Update user
  DELETE /users/:id            - Delete user

System
  GET    /system/health        - System health check
  GET    /system/stats         - System statistics
  POST   /system/logs          - Submit frontend error logs
```

---

## Database Schema Examples

### MongoDB (Mongoose)

```javascript
// Incident Schema
const incidentSchema = new mongoose.Schema({
  type: {
    type: String,
    enum: ['violence', 'car_crash', 'people_counter'],
    required: true
  },
  cameraId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Camera',
    required: true
  },
  location: String,
  severity: {
    type: String,
    enum: ['low', 'medium', 'high', 'critical']
  },
  confidence: Number,
  videoUrl: String,
  thumbnailUrl: String,
  timestamp: {
    type: Date,
    default: Date.now
  },
  status: {
    type: String,
    enum: ['active', 'resolved', 'false_alarm'],
    default: 'active'
  },
  aiModelId: String,
  metadata: Object
});

// Camera Schema
const cameraSchema = new mongoose.Schema({
  name: String,
  location: String,
  streamUrl: String,
  hlsUrl: String,
  status: {
    type: String,
    enum: ['online', 'offline', 'maintenance'],
    default: 'online'
  },
  aiModels: [{
    type: String,
    enabled: Boolean
  }],
  resolution: String,
  fps: Number,
  settings: Object
});

// User Schema
const userSchema = new mongoose.Schema({
  name: String,
  email: {
    type: String,
    unique: true,
    required: true
  },
  hashedPassword: String,
  role: {
    type: String,
    enum: ['admin', 'officer', 'security-authority'],
    required: true
  },
  permissions: [String],
  lastLogin: Date
});
```

---

## Environment Variables

Create `.env` file (backend):

```bash
# Server
PORT=5000
NODE_ENV=production

# Database
MONGODB_URI=mongodb://localhost:27017/vigil
# OR
DATABASE_URL=postgresql://user:pass@localhost:5432/vigil

# Authentication
JWT_SECRET=your-super-secret-key-change-this
JWT_EXPIRE=24h

# Frontend URL (CORS)
FRONTEND_URL=http://localhost:3000

# AI Services
AI_API_KEY=your-ai-service-api-key
AI_API_URL=https://ai-service.com/api

# File Storage (S3, Cloudinary, etc.)
S3_BUCKET=vigil-videos
S3_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret

# Email (for alerts)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
```

Create `.env.local` file (frontend):

```bash
REACT_APP_API_URL=http://localhost:5000
REACT_APP_WS_URL=ws://localhost:5000
REACT_APP_ENV=development
```

Update webpack config to use environment variables:

```javascript
// webpack.config.js
const webpack = require('webpack');
const dotenv = require('dotenv');

const env = dotenv.config().parsed;

module.exports = {
  plugins: [
    new webpack.DefinePlugin({
      'process.env': JSON.stringify(env)
    })
  ]
};
```

---

## Security Considerations

### 1. Protect API Keys

```javascript
// ❌ Don't do this
const apiKey = 'hardcoded-key-in-frontend';

// ✅ Do this
const apiKey = process.env.REACT_APP_API_KEY;
```

### 2. Use HTTPS in Production

```javascript
const API_URL = process.env.NODE_ENV === 'production'
  ? 'https://api.vigil.com'
  : 'http://localhost:5000';
```

### 3. Validate JWT Tokens

```javascript
// Frontend: Include token in requests
const token = localStorage.getItem('authToken');
fetch(url, {
  headers: {
    'Authorization': `Bearer ${token}`
  }
});

// Backend: Verify token middleware
const verifyToken = (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1];
  
  if (!token) {
    return res.status(401).json({ message: 'No token provided' });
  }
  
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded;
    next();
  } catch (error) {
    res.status(401).json({ message: 'Invalid token' });
  }
};
```

### 4. Implement Rate Limiting

```javascript
// Backend
const rateLimit = require('express-rate-limit');

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});

app.use('/api/', limiter);
```

---

## Testing Backend Integration

### 1. Test Authentication

```javascript
// Test login
const testLogin = async () => {
  const response = await fetch('http://localhost:5000/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      email: 'admin@vigil.com',
      password: 'admin123',
      role: 'admin'
    })
  });
  
  const data = await response.json();
  console.log('Token:', data.token);
};
```

### 2. Test WebSocket Connection

```javascript
import io from 'socket.io-client';

const socket = io('http://localhost:5000', {
  auth: { token: 'your-jwt-token' }
});

socket.on('connect', () => {
  console.log('Connected to WebSocket');
});

socket.on('new_incident', (incident) => {
  console.log('New incident:', incident);
});
```

---

## Deployment

### Frontend (Vigil)

**Netlify:**
```bash
npm run build
netlify deploy --prod --dir=dist
```

**Vercel:**
```bash
npm run build
vercel --prod
```

**Traditional Hosting:**
```bash
npm run build
# Upload /dist folder to your server
```

### Backend

**Heroku:**
```bash
git push heroku main
heroku config:set JWT_SECRET=your-secret
```

**AWS EC2:**
```bash
# SSH into server
npm install -g pm2
pm2 start server.js
pm2 startup
pm2 save
```

**Docker:**
```dockerfile
FROM node:18
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 5000
CMD ["node", "server.js"]
```

---

## Next Steps

1. ✅ Choose your backend technology (Node.js/Python/BaaS)
2. ✅ Set up database (MongoDB/PostgreSQL)
3. ✅ Implement authentication API
4. ✅ Create incident endpoints
5. ✅ Set up WebSocket server
6. ✅ Connect AI detection services
7. ✅ Test integration locally
8. ✅ Deploy to production

---

*Backend Integration Guide - Last Updated: January 2026*
