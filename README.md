# CXR-GNN

Chest X-ray classification with a FastAPI backend and React frontend. The
application uses a frozen ResNet18 image encoder, a GATv2 graph classifier, an
independent chest-X-ray reference gate, and uncertainty checks before showing a
disease prediction.

> **Research and education only.** This project is not a medical device and
> must not be used as a substitute for a qualified radiologist.

## Live demo

[Open the deployed application](https://x-ray-app-ropd.onrender.com/)

## Features

- Chest-X-ray input gate based on an approved reference-image bank.
- Fail-closed behavior for non-medical images and unsupported uploads.
- Disease prediction with a 50% confidence floor; uncertain cases are marked
  for review instead of displaying a disease label.
- React/Vite interface with mobile-friendly upload compression.
- FastAPI health and prediction endpoints.

## Project layout

```text
backend/
  app/                    FastAPI routes and model service
  cxr_gnn/                preprocessing, graph creation, model and validation
  checkpoints/            model checkpoint and reference bank
  requirements.txt
frontend/
  src/                    React application
  dist/                   production build served by the backend
render.yaml               Render deployment configuration
```

## Local development

### Backend

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

Check readiness at `http://localhost:8080/api/health`.

### Frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. The Vite development proxy forwards `/api` to
the backend on port `8080`.

## API

### `GET /api/health`

Returns model and validator readiness.

### `POST /api/predict`

Accepts a multipart form upload named `file`. A successful response contains
the predicted class, confidence, and ranked class probabilities. Rejected
uploads return `status: "rejected"` and do not expose disease probabilities.

## Deployment

- **Backend:** Railway or another Python container host. Set the public
  backend URL and expose the service port supplied by `$PORT`.
- **Frontend:** Render static site. Set `VITE_API_URL` to the public backend
  URL without a trailing `/api`, then rebuild after changes.
- Keep `backend/checkpoints/best_gatv2.pt` and
  `backend/checkpoints/xray_reference_bank.npz` available to the backend.

After pushing to `main`, wait for the Railway/Render deployment to finish and
verify `/api/health` before testing an upload.

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `CXR_GNN_CKPT` | Model checkpoint path | `backend/checkpoints/best_gatv2.pt` |
| `CXR_REFERENCE_BANK` | Reference gate bank path | `backend/checkpoints/xray_reference_bank.npz` |
| `CXR_REFERENCE_THRESHOLD` | Reference similarity threshold | `0.7287` |
| `CXR_REFERENCE_PHASH_MAX_DISTANCE` | Approved-image pHash tolerance | `6` |
| `CXR_REQUIRE_DICOM` | Require verified chest DICOM metadata | `false` |

## License

This project is released under the [MIT License](LICENSE).
