<div align="center">

<img src="https://i.ibb.co/xS72Y4nT/Chat-GPT-Image-Jul-22-2026-12-39-45-AM.png" alt="SignSpot AI Logo" width="220" />

# SignSpot AI Detection Server

**Automated PDF Signature-Field Localizer & Image Background Extractor**

Developed by **Adedoyin Ifeoluwa James**
In collaboration with **NORA Research Lab**

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688)
![Docker](https://img.shields.io/badge/docker-ready-2496ED)
![Status](https://img.shields.io/badge/status-live-brightgreen)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)

**🔗 Live API: [https://signspot-frontend.vercel.app](https://signspot-frontend.vercel.app)**

</div>

---

## Overview

**SignSpot AI Detection Server** is a lightweight microservice for document workflow automation. It exposes REST API endpoints — plus a minimal Gradio UI for manual testing — that solve two problems:

1. **PDF signature-field detection.** Renders each page of a PDF, runs OCR across it, looks for role/signature keywords (`Signature`, `Signed`, `Applicant`, `Witness`, `Authorized`, `Owner`, `Director`, `Employee`, `Customer`, `Manager`), and pairs each match with the nearest horizontal line to estimate a signature bounding box.
2. **Signature background removal.** Strips the background from a signature photo or scan, crops to the transparent bounds, and returns a base64-encoded PNG ready to stamp onto a document.

The REST API is the primary interface — the Gradio tabs exist for quick manual checks, not as the main product.

## Live Demo

The API is live at **https://signspot.onrender.com**.

```bash
curl https://signspot.onrender.com/health
# {"status": "online"}
```

- Interactive OpenAPI docs: **https://signspot.onrender.com/docs**
- Gradio UI (manual testing): **https://signspot.onrender.com**

> **Note:** this is hosted on Render's free tier, which sleeps after a period of inactivity. The first request after idle can take 20–50 seconds while the instance wakes up — that's expected, not a bug. A `GET /health` call is a good way to "warm" it before a user's first real request.

## Features

- Two REST endpoints (`/analyze`, `/signature`) plus `/health` and auto-generated OpenAPI docs at `/docs`
- Multi-page PDF support, page-by-page streaming (doesn't hold every page image in memory at once)
- OCR model loaded once and reused across requests, not reloaded per call
- CPU-only inference — no GPU required
- CORS enabled by default for cross-origin frontend calls
- Heuristic-only detection — no signature-specific ML model to train, host, or version

## API Reference

### `GET /health`

```bash
curl https://signspot.onrender.com/health
```

```json
{"status": "online"}
```

### `POST /analyze`

`multipart/form-data`, field name `pdf`. Optional `?dpi=150` query param (default `150`) to trade OCR accuracy against speed.

```bash
curl -X POST "https://signspot.onrender.com/analyze" \
  -F "pdf=@contract.pdf"
```

```json
{
  "status": "success",
  "pages": 1,
  "signature_fields": [
    {
      "id": "field1",
      "label": "Applicant Signature",
      "page": 1,
      "x": 151,
      "y": 1448,
      "width": 475,
      "height": 43
    }
  ]
}
```

When nothing is detected:

```json
{"status": "error", "message": "No signature field found."}
```

### `POST /signature`

`multipart/form-data`, field name `signature`.

```bash
curl -X POST "https://signspot.onrender.com/signature" \
  -F "signature=@signature.jpg"
```

```json
{
  "status": "success",
  "image": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

`image` is a base64-encoded transparent PNG — use directly as `data:image/png;base64,<value>`.

All error responses across both endpoints follow `{"status": "error", "message": "..."}`.

### Calling it from a browser frontend

CORS is open (`allow_origins=["*"]`), so any frontend can call these endpoints directly:

```javascript
const API_BASE = "https://signspot.onrender.com";

async function analyzePdf(pdfFile) {
  const formData = new FormData();
  formData.append("pdf", pdfFile);
  const res = await fetch(`${API_BASE}/analyze`, { method: "POST", body: formData });
  return res.json();
}

async function cleanSignature(imageFile) {
  const formData = new FormData();
  formData.append("signature", imageFile);
  const res = await fetch(`${API_BASE}/signature`, { method: "POST", body: formData });
  const data = await res.json();
  if (data.status === "success") return `data:image/png;base64,${data.image}`;
  throw new Error(data.message);
}
```

## Architecture & Performance Notes

- **Lazy model loading.** EasyOCR and rembg are imported and initialized on first use, not at boot, guarded by a double-checked lock. This keeps `/health` and the Gradio UI available quickly; the one-time model-load cost lands on whichever request triggers it first.
- **Threadpool offloading.** OCR and background-removal work runs via `run_in_threadpool`, so a slow `/analyze` call doesn't block `/health` or other concurrent requests.
- **CPU-only PyTorch.** `requirements.txt` points at PyTorch's CPU wheel index so EasyOCR doesn't pull in unused CUDA/cuDNN packages.
- **Lightweight background-removal model.** Uses rembg's `u2netp` (~4.5MB) instead of the default `u2net` (~176MB) — sufficient for isolating ink on paper.
- **Streamed PDF rendering.** Pages are rendered and processed one at a time via a generator, not all materialized in memory upfront.

## Tech Stack

| Component | Library |
|---|---|
| Web framework | FastAPI + Uvicorn |
| UI | Gradio (mounted into the FastAPI app) |
| PDF rendering | PyMuPDF (fitz) |
| OCR | EasyOCR (CPU) |
| Line detection | OpenCV |
| Background removal | rembg (u2netp) |
| Image handling | Pillow, NumPy |

## Project Structure

```
.
├── app.py            # all application logic — routes, OCR, detection heuristics, UI
├── requirements.txt
├── Dockerfile
└── README.md
```

## Deployment

### Currently live on: Render

This service is deployed on [Render](https://render.com)'s free web service tier, built directly from the `Dockerfile` in this repo.

1. Create a new **Web Service** on Render and connect this repository.
2. Render detects the `Dockerfile` automatically — no build command needed.
3. No environment variables are required.

### Alternative: Google Cloud Run (free tier, no cold-start RAM ceiling)

```bash
gcloud run deploy signspot \
  --source . \
  --port 7860 \
  --memory 2Gi \
  --allow-unauthenticated \
  --region us-central1
```

Cloud Run's free monthly quota (requests, GB-seconds, vCPU-seconds) comfortably covers a low-traffic API, and you can allocate more memory per instance than Render's free tier allows — worth moving to if `/analyze` ever gets memory-killed under real load.

### Alternative: Hugging Face Spaces (Docker SDK)

Works with the same `Dockerfile` and `app_port: 7860` frontmatter, but Hugging Face now requires a **PRO** subscription to create Gradio or Docker SDK Spaces — no longer available on the free tier.

### Local (Docker)

```bash
docker build -t signspot .
docker run -p 7860:7860 signspot
```

### Local (no Docker)

```bash
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:7860` for the UI, or `http://localhost:7860/docs` for the API reference.

## Known Limitations

- **Render's free tier caps at 512MB RAM**, which is tight for EasyOCR + PyTorch under real load — if `/analyze` starts returning 502s or timing out on genuine multi-page PDFs, that's the likely cause. Google Cloud Run (above) doesn't have this ceiling.
- **Cold start after inactivity.** The free instance sleeps when idle; the first request afterward can take 20–50 seconds.
- **Model weights download on first use** unless pre-baked into the image, adding to that first-request delay after any redeploy or restart.
- Signature-field detection is **heuristic** (OCR + line geometry), not a trained detection model — accuracy depends on document layout and scan quality.

## License

No license has been specified yet — all rights reserved by default. Add a `LICENSE` file (MIT and Apache-2.0 are common choices for a tool like this) if you want to open it up.

## Author

**Adedoyin Ifeoluwa James**
NORA Research Lab
