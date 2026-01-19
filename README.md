# VIGIL – AI-Powered Surveillance System

VIGIL is a real-time surveillance demonstration platform that uses advanced AI models to detect violence, car crashes, and count people in video feeds. It features a Python/Flask backend that runs AI inference and a modern React/Vite frontend for live monitoring.

---

## 🚀 Quick Start (Fresh Install)
This guide is for setting up the project from scratch (e.g., after cloning from GitHub).

### Prerequisites
*   **Python 3.8+** (Ensure added to PATH)
*   **Node.js 16+** (Ensure added to PATH)
*   **PowerShell** (for the unified startup script)

### 1️⃣ Setup & Run (One-Click)
The easiest way is to run the included **automated script**. It will verify your environment, install all python/node dependencies if missing, and start the system.

```powershell
.\start-vigil.ps1
```

### 2️⃣ Manual Setup (If Script Fails)
If you prefer to run things manually:

**Backend:**
```bash
pip install -r requirements.txt
python -m backend.app
```

**Frontend:**
```bash
cd "Vigil Surveillance App Design - Figma"
npm install  # Run this once to install dependencies
npm run dev
```

### Access Points
*   **Backend API**: `http://127.0.0.1:5000`
*   **Frontend Dashboard**: `http://localhost:3000`

### 🌍 Deployment
For detailed instructions on deploying to a production environment (including database and server setup), see the **[Deployment Guide](DEPLOYMENT.md)**.

---

## 🏗️ Architecture

### AI & Backend (`/backend`)
*   **Framework**: Flask with Socket.IO for real-time updates.
*   **Inference Engine** (`backend/ai/inference.py`):
    *   **Violence Detection**: MobileNet-Clip based model. Integrated with **YOLOv8 People Counting** for crowd analytics.
    *   **Crash Detection**: MobileNetV2-LSTM hybrid model specialized for vehicle accidents.
*   **Simulator** (`backend/services/camera_simulator.py`): Simulates live patterns by rotating video clips on a loop.

### Frontend (`/Vigil Surveillance App Design - Figma`)
*   **Design**: Modern dark-mode UI with "glassmorphism" aesthetics.
*   **Features**:
    *   Live camera grid.
    *   Real-time incident alerts.
    *   Interactive dashboard for security personnel.

---

## ⚙️ Configuration

You can tune the system's sensitivity in `backend/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `VIOLENCE_THRESHOLD` | 0.70 | Confidence required to flag violence. High precision mode. |
| `ACCIDENT_THRESHOLD` | 0.30 | Confidence required to flag crashes. Tuned for demo sensitivity. |
| `DEFAULT_CAMERAS` | 12 Cams | List of active camera IDs. |
| `MODEL_PATHS` | - | Paths to `.pt` and `.pth` model weights. |

Cameras are assigned specific detection roles in `config.py`:
*   **Violence Cameras** (`CAM-042`, `CAM-128`, etc.): Equipped with the **Violence Model** and **YOLO People Counter**. They process videos from the `violence` and `no_violence` folders.
*   **Crash Cameras** (`CAM-283`, `CAM-074`, etc.): Equipped with the **Crash Model**. They process videos from the `crash` and `no_crash` folders.

---

## 🛠️ Troubleshooting

**"Frontend folder not found" error:**
Ensure the folder `Vigil Surveillance App Design - Figma` exists in the root. The startup script expects this exact name.

**Backend starts but no videos play:**
Check that the `Videos/` directory contains standard MP4 files. The system expects videos to be organized in subfolders or directly in the root of `Videos/`.

**"Module not found" errors:**
Manually install backend dependencies:
```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
├── backend/                  # Flask API & AI Logic
│   ├── app.py                # Main entry point
│   ├── ai/                   # Inference scripts (inference.py, etc.)
│   ├── services/             # Simulator, Incident Management
│   └── config.py             # System constants
├── Vigil Surveillance.../    # React/Vite Frontend
│   ├── src/                  # React components
│   └── package.json          # Frontend dependencies
├── start-vigil.ps1           # Verified startup script
└── requirements.txt          # Python dependencies
```
