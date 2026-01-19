# Vigil Surveillance System - Deployment Guide

This guide provides step-by-step instructions for deploying the Vigil Surveillance System, consisting of a **Python Flask Backend** (with AI models) and a **React Frontend**.

## Prerequisites

Ensure the following are installed on the deployment machine:

1.  **Python 3.10+**: [Download Python](https://www.python.org/downloads/)
2.  **Node.js 18+ & npm**: [Download Node.js](https://nodejs.org/)
3.  **Git**: [Download Git](https://git-scm.com/)
4.  **wkhtmltopdf** (Optional, for PDF reports): [Download](https://wkhtmltopdf.org/downloads.html)
    *   *Note: Add the `bin` folder to your System PATH.*

---

## 1. Backend Setup (API & AI)

The backend handles video processing, AI inference (Violence/Crash/People Counting), and serves the API.

### 1.1. Navigate to Backend
```bash
cd backend
```

### 1.2. Create Virtual Environment
It is recommended to use a virtual environment to manage dependencies.
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 1.3. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: If you have a GPU (NVIDIA), ensure you install the CUDA-supported version of PyTorch separately if `requirements.txt` installs the CPU version by default. Visit [pytorch.org](https://pytorch.org/) for specific commands.*

### 1.4. Configure Environment
The backend uses in-memory storage for incidents and JSON files for reports by default, so **no external database (SQL/Mongo) is required** for the initial setup.

**Key Configuration Files:**
*   `backend/config.py`: Contains camera definitions, model path settings, and sensitivity thresholds.
*   `backend/app.py`: Main entry point.

### 1.5. Prepare Video Data (CRITICAL)
The system simulates camera feeds using video files processed by specific AI models based on their folder location. 

**You MUST organize your videos EXACTLY as follows for the system to detect incidents correctly:**

```text
/VIGIL
  /Videos
    ├── /violence       # Videos containing violence (Triggers Siren Alerts)
    ├── /no_violence    # Normal activity videos (monitored for violence)
    ├── /crash          # Videos containing car accidents (Triggers Crash Alerts)
    └── /no_crash       # Normal traffic videos (monitored for crashes)
```

*   **Supported Formats**: `.mp4`
*   **Behavior**: The simulator rotates through videos in these folders.
    *   Videos in `violence` will trigger violence alerts.
    *   Videos in `crash` will trigger accident alerts.
    *   Videos in `no_...` folders will show as "Normal" status.

### 1.6. Run the Backend
```bash
# Development
python -m backend.app

# Production (using Gunicorn on Linux)
# gunicorn -k eventlet -w 1 --bind 0.0.0.0:5000 backend.app:app
```
The server will start on `http://localhost:5000`.

---

## 2. Frontend Setup (React User Interface)

The frontend is a modern React application built with Vite.

### 2.1. Navigate to Frontend
```bash
cd "Vigil Surveillance App Design - Figma"
```

### 2.2. Install Dependencies
```bash
npm install
```

### 2.3. Configure API Connection
Create a `.env` file in the frontend root if it doesn't exist, and point it to your backend:

**File:** `.env`
```env
VITE_API_URL=http://localhost:5000
```
*Replace `http://localhost:5000` with your production backend IP/URL if deploying to a remote server.*

### 2.4. Run Local Development
```bash
npm run dev
```
Access the app at `http://localhost:5173`.

### 2.5. Build for Production
To generate static files for deployment (e.g., via Nginx, Vercel, or Netlify):
```bash
npm run build
```
The output will be in the `dist` folder.

---

## 3. Database & Storage Info

*   **Incidents**: Currently stored in-memory (RAM) for high-speed demo performance. They reset via the "Dismiss All" feature or restart.
*   **Reports**: Saved as JSON files in `backend/reports/`.
*   **Users**: Hardcoded demo users in `app.py`.
    *   Admin: `admin@vigil.com` / `admin123`
    *   Officer: `officer@vigil.com` / `officer123`
    *   Security: `security@vigil.com` / `security123`

To persist data permanently, modifications to `backend/services/incident_storage.py` to use SQLite or PostgreSQL would be required.

---

## 4. Troubleshooting

*   **"Model not found"**: Ensure AI models (`.pt`, `.pth` files) are present in `backend/models/`.
*   **Connection Error**: Verify the frontend `.env` matches the backend's running port.
*   **Video Playback Fails**: Ensure video paths in `backend/config.py` logic match your actual `Videos` directory structure.
