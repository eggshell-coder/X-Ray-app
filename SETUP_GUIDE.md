# Local setup guide

## Requirements

- Python 3.10 or 3.11
- Node.js 18 or newer
- Git

## Backend

```powershell
git clone https://github.com/eggshell-coder/X-Ray-app.git
cd X-Ray-app\backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

Verify the service:

```text
http://localhost:8080/api/health
```

## Frontend

Open a second terminal:

```powershell
cd X-Ray-app\frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

## Production deployment

The repository includes `render.yaml` for deployment configuration. Configure
the Render frontend variable `VITE_API_URL` with the Railway backend's public
URL, rebuild the frontend, and then verify the backend health endpoint.

Tunneling is only needed when a phone must access a backend running on your
local computer. It is not needed when using the deployed Railway URL.
