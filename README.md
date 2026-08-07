# CXR·GNN — Chest X-ray Disease Classifier (FastAPI + GATv2)

Full app for your `cxr_gnn` project: a FastAPI backend that loads your trained
`best_gatv2.pt` and runs the exact same image→superpixel-graph→GATv2 pipeline
from your notebook, plus a single-page frontend to upload an X-ray and see
the prediction.

```
xray-app/
├── backend/
│   ├── app/
│   │   ├── main.py            ← FastAPI app (routes)
│   │   └── model_service.py   ← loads checkpoint, runs inference
│   ├── cxr_gnn/                ← trimmed copy of your training package
│   │   ├── config.py
│   │   ├── utils.py
│   │   ├── models/{encoder,gatv2}.py
│   │   └── data/{dataset,graph}.py
│   ├── checkpoints/            ← put best_gatv2.pt here
│   └── requirements.txt
└── frontend/
    └── index.html               ← single-file UI, no build step
```

## 1. Install & run the backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
# torch-geometric needs matching wheels for your torch/CUDA version — if the
# plain pip install fails, follow: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

cp /path/to/your/best_gatv2.pt checkpoints/best_gatv2.pt

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** — FastAPI serves the frontend directly from
`/`, so backend and UI run as one app. No separate frontend server needed.

Check the model loaded correctly:
```bash
curl http://localhost:8000/api/health
```

## 2. How prediction works

`POST /api/predict` (multipart file upload) →

1. Image is decoded, converted to grayscale, resized to `cfg.img_size`,
   intensity-rescaled — identical to `load_gray()` in your notebook.
2. SLIC superpixel segmentation builds a region-adjacency graph.
3. Each node gets a 140-d feature vector: 128-d frozen-ResNet18 deep
   features (average-pooled per region) + 12-d hand-crafted texture/shape
   features. Edges get a 2-d intensity/texture-difference feature.
3. The graph is passed through your trained `GATv2Classifier`.
4. Softmax probabilities for every class are returned, ranked.

```json
{
  "prediction": "TB",
  "confidence": 0.842,
  "probabilities": [
    {"label": "TB", "probability": 0.842},
    {"label": "ChronicLung", "probability": 0.091},
    {"label": "Normal", "probability": 0.041},
    {"label": "Cardiac", "probability": 0.018},
    {"label": "Pleural", "probability": 0.008}
  ],
  "n_superpixels": 176
}
```

Class names are read straight from the checkpoint's `class2idx`, so this
adapts automatically if you retrain with a different label set.

## 3. Deploying

- **Docker**: wrap `backend/` in a `python:3.11-slim` image, `pip install -r
  requirements.txt`, `COPY frontend /frontend`, run uvicorn. Mount or bake in
  `checkpoints/best_gatv2.pt`.
- **Hosting**: Render, Railway, Fly.io, or a GPU box on AWS/GCP all work fine
  for FastAPI + PyTorch. CPU inference is fine for single-image requests;
  GPU only matters if you expect heavy concurrent traffic.
- Lock down CORS in `app/main.py` (`allow_origins`) once you have a real
  frontend domain — it's wide open (`*`) for local development right now.

## 4. Notes / things you may want to change

- **Preprocessing must stay in sync.** If you ever change SLIC parameters,
  feature extraction, or normalization in the notebook, mirror the change in
  `backend/cxr_gnn/data/graph.py` and `dataset.py` — inference will silently
  degrade otherwise, since it's re-implemented here rather than imported
  from the notebook.
- **`graph_cache.pt`** from your project folder is a training-time cache of
  precomputed graphs — it is not needed for inference and isn't used here.
- The frontend is one static HTML file (no npm/build step) so it's easy to
  swap for a React app later if you want a fancier UI — the API contract
  above stays the same either way.
