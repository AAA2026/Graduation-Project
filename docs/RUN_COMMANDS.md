
# VIGIL RUN COMMANDS

## Frontend (React/Vite)

1. Open a terminal.
2. Change directory to the React app:

	```powershell
	cd "S:\GRAD PROJECT\VIGIL\Vigil Surveillance App Design - Figma"
	npm install  # Only needed once if dependencies change
	```

3. Start the frontend:

	```powershell
	npm run dev
	```

	- App will be available at: http://localhost:5173/ (or similar, check output)
	- Output should show: `VITE vX.X.X` and `Local: http://localhost:5173/`

---

## Backend (Flask)

**CRITICAL:** Must be run from the PROJECT ROOT (`S:\GRAD PROJECT\VIGIL`) as a module.

1. Open a new terminal.
2. Change directory to the **ROOT** folder:

	```powershell
	cd "S:\GRAD PROJECT\VIGIL"
	```

3. Activate your Python environment (if needed):

	```powershell
	# Example for local venv
	.\venv\Scripts\Activate.ps1
	```

4. Start the backend as a module:

	```powershell
	python -m backend.app
	```

	- Output should show: `Simulator started` and Flask running on port 5000.
	- API available at: http://localhost:5000/
	- **DO NOT** run `python app.py` from inside the `backend` folder. This will fail imports.

---

## Health Checks

- Frontend: http://localhost:5173/ (or port shown in terminal)
- Backend Status: http://localhost:5000/api/live-status
- Incidents API: http://localhost:5000/api/incidents

---

## Troubleshooting

- **Backend Module Error:** If you see `ModuleNotFoundError`, make sure you are in `S:\GRAD PROJECT\VIGIL` and running `python -m backend.app`.
- **Port Conflicts:** Ensure port 5000 (backend) and 5173 (frontend) are free.
- **Environment:** Ensure `.env` files are correctly placed if used.

---

## Quick Reference

- **Frontend:**
  - `cd "Vigil Surveillance App Design - Figma"`
  - `npm run dev`

- **Backend:**
  - `cd "S:\GRAD PROJECT\VIGIL"` (Root)
  - `python -m backend.app`

