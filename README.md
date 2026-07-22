
<img src="https://i.ibb.co/xS72Y4nT/Chat-GPT-Image-Jul-22-2026-12-39-45-AM.png" alt="SignSpot AI Logo" width="220" />

# SignSpot AI Detection Server

**Automated PDF Signature-Field Localizer & Image Background Extractor**

Developed by **Adedoyin Ifeoluwa James**
In collaboration with **NORA Research Lab**

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688)
![Docker](https://img.shields.io/badge/docker-ready-2496ED)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)

</div>

---

> **Note for GitHub readers:** the YAML block at the very top of this file is Hugging Face Spaces configuration, not part of the document body. GitHub will render it as plain text above the horizontal rule — that's expected, not a formatting bug.

## Overview

**SignSpot AI Detection Server** is a lightweight microservice for document workflow automation. It exposes REST API endpoints — plus a minimal Gradio UI for manual testing — that solve two problems:

1. **PDF signature-field detection.** Renders each page of a PDF, runs OCR across it, looks for role/signature keywords (`Signature`, `Signed`, `Applicant`, `Witness`, `Authorized`, `Owner`, `Director`, `Employee`, `Customer`, `Manager`), and pairs each match with the nearest horizontal line to estimate a signature bounding box.
2. **Signature background removal.** Strips the background from a signature photo or scan, crops to the transparent bounds, and returns a base64-encoded PNG ready to stamp onto a document.

The REST API is the primary interface — the Gradio tabs exist for quick manual checks, not as the main product.

## Features

- Two REST endpoints (`/analyze`, `/signature`) plus `/health` and auto-generated OpenAPI docs at `/docs`
- Multi-page PDF support, page-by-page streaming (doesn't hold every page image in memory at once)
- OCR model loaded once and reused across requests, not reloaded per call
- CPU-only inference — no GPU required
- CORS enabled by default for cross-origin frontend calls
- Heuristic-only detection — no signature-specific ML model to train, host, or version

## API Reference

### `GET /health`

```json
{"status": "online"}
```

### `POST /analyze`

`multipart/form-data`, field name `pdf`. Optional `?dpi=150` query param (default `150`) to trade OCR accuracy against speed.

```bash
curl -X POST "https://your-space.hf.space/analyze" \
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
curl -X POST "https://your-space.hf.space/signature" \
  -F "signature=@signature.jpg"
```

```json
{
  "status": "success",
  "image": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

All error responses across both endpoints follow `{"status": "error", "message": "..."}`.

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

### Hugging Face Spaces (Docker SDK)

1. Create a Space with SDK **Docker** (or keep this repo's frontmatter as-is — `sdk: docker` above does that for you).
2. Push `app.py`, `requirements.txt`, `Dockerfile`, and this `README.md` to the Space repo.
3. HF builds the image and starts the container on port `7860` automatically.

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

- **Cold start on the first OCR/background-removal request** can exceed a few seconds — EasyOCR's PyTorch dependency is inherently heavier than a pure-CPU OCR engine like Tesseract. Swap `easyocr` for `pytesseract` in `app.py`/`requirements.txt` if a hard low-latency/low-RAM target matters more than OCR accuracy.
- **Model weights download on first use** unless the Space has persistent storage or the weights are pre-baked into the image; expect a slower first request after any redeploy or restart.
- Signature-field detection is **heuristic** (OCR + line geometry), not a trained detection model — accuracy depends on document layout and scan quality.

## License

No license has been specified yet — all rights reserved by default. Add a `LICENSE` file (MIT and Apache-2.0 are common choices for a tool like this) if you want to open it up.

## Author

**Adedoyin Ifeoluwa James**
NORA Research Lab
