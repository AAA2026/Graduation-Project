# VIGIL System Startup Script
# Starts both backend and frontend servers

Write-Host "🚀 Starting VIGIL Surveillance System..." -ForegroundColor Cyan
Write-Host ""

# Check if Node.js is available
$nodeAvailable = Get-Command node -ErrorAction SilentlyContinue
if (-not $nodeAvailable) {
    Write-Host "❌ Node.js not found in PATH" -ForegroundColor Red
    Write-Host "Adding Node.js to PATH for this session..." -ForegroundColor Yellow
    $env:Path = "$env:Path;C:\Program Files\nodejs"
    
    # Verify again
    $nodeAvailable = Get-Command node -ErrorAction SilentlyContinue
    if (-not $nodeAvailable) {
        Write-Host "❌ Failed to add Node.js to PATH. Please install Node.js or add it manually." -ForegroundColor Red
        exit 1
    }
}

Write-Host "✅ Node.js found: $(node -v)" -ForegroundColor Green
Write-Host "✅ npm found: $(npm -v)" -ForegroundColor Green
Write-Host ""

# Check if Python is available
$pythonAvailable = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonAvailable) {
    Write-Host "❌ Python not found in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and add it to PATH" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Python found: $(python --version)" -ForegroundColor Green
Write-Host ""

# Project paths
$projectRoot = "S:\GRAD PROJECT\VIGIL"
$backendPath = Join-Path $projectRoot "backend"
$frontendPath = Join-Path $projectRoot "Vigil Surveillance App Design - Figma"

# Check if paths exist
if (-not (Test-Path $backendPath)) {
    Write-Host "❌ Backend folder not found at: $backendPath" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $frontendPath)) {
    Write-Host "❌ Frontend folder not found at: $frontendPath" -ForegroundColor Red
    exit 1
}

Write-Host "📁 Backend: $backendPath" -ForegroundColor Gray
Write-Host "📁 Frontend: $frontendPath" -ForegroundColor Gray
Write-Host ""

# Install Python dependencies (if needed)
Write-Host "📦 Checking Python dependencies..." -ForegroundColor Cyan
$requirementsFile = Join-Path $projectRoot "requirements.txt"
if (Test-Path $requirementsFile) {
    Write-Host "Installing Python packages (this may take a while)..." -ForegroundColor Yellow
    python -m pip install -q -r $requirementsFile
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Python dependencies installed" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Some Python packages may have failed to install" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  requirements.txt not found, skipping Python dependencies" -ForegroundColor Yellow
}
Write-Host ""

# Install Node dependencies (if needed)
Write-Host "📦 Checking Node.js dependencies..." -ForegroundColor Cyan
Push-Location $frontendPath
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing Node.js packages..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Node.js dependencies installed" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to install Node.js dependencies" -ForegroundColor Red
        Pop-Location
        exit 1
    }
} else {
    Write-Host "✅ Node.js dependencies already installed" -ForegroundColor Green
}
Pop-Location
Write-Host ""

# Start Backend Server
Write-Host "🔧 Starting Backend Server..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectRoot'; Write-Host '🐍 VIGIL Backend Server' -ForegroundColor Green; Write-Host ''; python -m backend.app" -WindowStyle Normal

Write-Host "⏳ Waiting for backend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check if backend is running
try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:5000/" -Method GET -TimeoutSec 3
    Write-Host "✅ Backend is running at http://127.0.0.1:5000" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Backend may not be running yet. Check the backend terminal." -ForegroundColor Yellow
}
Write-Host ""

# Start Frontend Server
Write-Host "🎨 Starting Frontend Server..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$frontendPath'; Write-Host '⚛️  VIGIL Frontend Server' -ForegroundColor Green; Write-Host ''; npm run dev" -WindowStyle Normal

Write-Host "⏳ Waiting for frontend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5
Write-Host ""

Write-Host "✨ VIGIL System Started!" -ForegroundColor Green
Write-Host ""
Write-Host "📍 Backend API:  http://127.0.0.1:5000" -ForegroundColor Cyan
Write-Host "📍 Frontend App: http://localhost:3000" -ForegroundColor Cyan

Write-Host ""
Write-Host "Press Ctrl+C to stop this script (servers will continue running)" -ForegroundColor Gray
Write-Host "To stop servers, close their respective PowerShell windows" -ForegroundColor Gray
Write-Host ""

# Keep script running
Write-Host "Script is running. Press Ctrl+C to exit..." -ForegroundColor Gray
while ($true) {
    Start-Sleep -Seconds 10
}
