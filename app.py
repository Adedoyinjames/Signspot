"""
SignSpot AI Detection Server
-----------------------------
Lightweight Hugging Face Gradio Space exposing REST endpoints for:
  - PDF signature-field detection (OCR + horizontal-line heuristics)
  - Signature image background removal

Design notes:
  - Heavy ML deps (EasyOCR/torch, rembg) are imported lazily, on first use,
    and cached as module-level singletons. This keeps the Space's boot time
    (Gradio UI + FastAPI routes available) fast; the one-time model-load
    cost is paid by whichever request triggers it first, not by startup.
  - CPU-bound work runs in FastAPI's threadpool (via run_in_threadpool) so
    /health and other requests stay responsive while a page is being OCR'd.
"""

import os
import io
import time
import base64
import traceback
import threading

import numpy as np
from PIL import Image
import cv2
import fitz  # PyMuPDF

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

import gradio as gr

# ============================================================
# Tunable heuristics
# ============================================================
RENDER_DPI = 150                  # page render resolution for OCR
LINE_MIN_WIDTH_RATIO = 0.03       # min horizontal line length, as a fraction of page width
LINE_MAX_HEIGHT = 8               # max thickness (px) to count as a "line" not a filled box
LINE_SEARCH_MAX_GAP_RATIO = 3.0   # search window below a label, in multiples of label height
SIGNATURE_BOX_PADDING = 40        # extra height (px) reserved above a detected line
FALLBACK_BOX_WIDTH = 180          # used when no line is found near a label
FALLBACK_BOX_HEIGHT = 45

KEYWORDS = [
    "signature", "signed", "applicant", "witness", "authorized",
    "owner", "director", "employee", "customer", "manager",
]
ROLE_WORDS = {"applicant", "witness", "owner", "director", "employee", "customer", "manager"}
SIGNATURE_WORDS = {"signature", "signed", "authorized"}

# ============================================================
# Lazy-loaded, globally-reused models
# ============================================================
_ocr_reader = None
_ocr_lock = threading.Lock()

_rembg_session = None
_rembg_lock = threading.Lock()


def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        with _ocr_lock:
            if _ocr_reader is None:
                print("[INIT] Loading OCR model (first use)...")
                t0 = time.time()
                import easyocr
                _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                print(f"[INIT] OCR model ready in {time.time() - t0:.2f}s")
    return _ocr_reader


def get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        with _rembg_lock:
            if _rembg_session is None:
                print("[INIT] Loading background-removal model (first use)...")
                t0 = time.time()
                from rembg import new_session
                _rembg_session = new_session("u2netp")  # small model, ~4.5MB
                print(f"[INIT] rembg model ready in {time.time() - t0:.2f}s")
    return _rembg_session


# ============================================================
# PDF -> page images (streamed, not all materialized at once)
# ============================================================
def iter_pdf_pages_as_images(pdf_bytes: bytes, dpi: int = RENDER_DPI):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")  # raises immediately on bad input
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    def _gen():
        try:
            for page in doc:
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                yield img
        finally:
            doc.close()

    return _gen()


# ============================================================
# Horizontal line detection
# ============================================================
def detect_horizontal_lines(image_np: np.ndarray):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    width = image_np.shape[1]
    kernel_len = max(20, int(width * LINE_MIN_WIDTH_RATIO))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= kernel_len and h <= LINE_MAX_HEIGHT:
            lines.append((x, y, w, h))
    return lines


# ============================================================
# Keyword hits -> signature fields
# ============================================================
def find_keyword_hits(ocr_results):
    hits = []
    for bbox, text, conf in ocr_results:
        low = text.lower()
        matched = [k for k in KEYWORDS if k in low]
        if matched:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            hits.append({
                "text": text.strip(),
                "keywords": matched,
                "x_min": min(xs), "x_max": max(xs),
                "y_min": min(ys), "y_max": max(ys),
            })
    return hits


def find_nearest_line(hit, lines):
    hit_h = max(1.0, hit["y_max"] - hit["y_min"])
    hit_cy = (hit["y_min"] + hit["y_max"]) / 2
    best, best_score = None, None

    for (lx, ly, lw, lh) in lines:
        line_cy = ly + lh / 2
        vgap = line_cy - hit["y_max"]  # positive => line is below the label

        same_row = abs(line_cy - hit_cy) < hit_h * 1.5 and lx >= hit["x_min"] - 5
        below = 0 <= vgap <= hit_h * LINE_SEARCH_MAX_GAP_RATIO and (lx + lw) >= hit["x_min"] - 10

        if same_row or below:
            score = abs(vgap) + (0 if same_row else 5) + abs(lx - hit["x_max"]) * 0.1
            if best_score is None or score < best_score:
                best, best_score = (lx, ly, lw, lh), score

    return best


def build_label(texts, keywords):
    role = next((w.title() for w in ROLE_WORDS if w in keywords), None)
    combined = " ".join(dict.fromkeys(t.strip() for t in texts if t.strip()))
    low = combined.lower()
    if any(w in low for w in SIGNATURE_WORDS):
        return combined.title()
    if role:
        return f"{role} Signature"
    return f"{combined.title()} Signature" if combined else "Signature"


def build_signature_fields(ocr_results, lines, page_idx, start_id, page_w, page_h):
    hits = find_keyword_hits(ocr_results)
    grouped = {}
    unmatched = []

    for hit in hits:
        line = find_nearest_line(hit, lines)
        if line:
            key = (round(line[0] / 10), round(line[1] / 10))
            g = grouped.setdefault(key, {"line": line, "texts": [], "keywords": set()})
            g["texts"].append(hit["text"])
            g["keywords"].update(hit["keywords"])
        else:
            unmatched.append(hit)

    fields = []
    fid = start_id

    for g in grouped.values():
        lx, ly, lw, lh = g["line"]
        height = max(30, lh + SIGNATURE_BOX_PADDING)
        fields.append({
            "id": f"field{fid}",
            "label": build_label(g["texts"], g["keywords"]),
            "page": page_idx,
            "x": int(lx),
            "y": int(ly - height + lh),
            "width": int(lw),
            "height": int(height),
        })
        fid += 1

    for hit in unmatched:
        x = int(hit["x_max"] + 10)
        y = int(hit["y_min"] - 5)
        if x + FALLBACK_BOX_WIDTH > page_w:
            x = int(hit["x_min"])
            y = int(hit["y_max"] + 8)
        fields.append({
            "id": f"field{fid}",
            "label": build_label([hit["text"]], set(hit["keywords"])),
            "page": page_idx,
            "x": x,
            "y": max(0, y),
            "width": FALLBACK_BOX_WIDTH,
            "height": FALLBACK_BOX_HEIGHT,
        })
        fid += 1

    return fields


# ============================================================
# Signature background removal
# ============================================================
def crop_transparent_border(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    if arr.shape[-1] < 4:
        return img
    alpha = arr[:, :, 3]
    coords = cv2.findNonZero((alpha > 10).astype(np.uint8))
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img.crop((x, y, x + w, y + h))


# ============================================================
# Core logic (shared by REST routes and the Gradio UI)
# ============================================================
def analyze_pdf_core(pdf_bytes: bytes, dpi: int = RENDER_DPI):
    try:
        page_iter = iter_pdf_pages_as_images(pdf_bytes, dpi=dpi)
    except Exception as e:
        return {"status": "error", "message": f"Could not read PDF: {e}"}, 400

    try:
        reader = get_ocr_reader()
    except Exception as e:
        print(f"[ERROR] OCR model failed to load: {e}")
        return {"status": "error", "message": "OCR model failed to load."}, 500

    all_fields = []
    next_id = 1
    page_count = 0

    try:
        for page_idx, pil_img in enumerate(page_iter, start=1):
            page_count = page_idx
            try:
                print(f"[PROCESS] OCR on page {page_idx}")
                page_w, page_h = pil_img.size
                np_img = np.array(pil_img)
                ocr_results = reader.readtext(np_img)
                lines = detect_horizontal_lines(np_img)
                page_fields = build_signature_fields(ocr_results, lines, page_idx, next_id, page_w, page_h)
                print(f"[DETECT] page {page_idx}: {len(page_fields)} signature field(s)")
                all_fields.extend(page_fields)
                next_id += len(page_fields)
            except Exception as page_err:
                print(f"[ERROR] page {page_idx} failed: {page_err}")
                continue
    except Exception as e:
        print(f"[ERROR] Failed while iterating PDF pages: {e}")
        traceback.print_exc()
        if page_count == 0:
            return {"status": "error", "message": "Failed while processing the PDF."}, 500
        # else: fall through and return whatever we already found

    if not all_fields:
        return {"status": "error", "message": "No signature field found."}, 200

    return {"status": "success", "pages": page_count, "signature_fields": all_fields}, 200


def remove_background_core(img_bytes: bytes):
    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    except Exception as e:
        return {"status": "error", "message": f"Invalid image file: {e}"}, 400

    try:
        session = get_rembg_session()
        from rembg import remove
        result_img = remove(pil_img, session=session)
    except Exception as e:
        print(f"[ERROR] Background removal failed: {e}")
        traceback.print_exc()
        return {"status": "error", "message": "Background removal failed."}, 500

    cropped = crop_transparent_border(result_img)
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"status": "success", "image": b64_str}, 200


# ============================================================
# FastAPI app + REST routes
# ============================================================
app = FastAPI(
    title="SignSpot AI Detection Server",
    description="REST API for PDF signature-field detection and signature background removal.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"[ERROR] Unhandled exception on {request.url.path}: {exc}")
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"status": "error", "message": "Internal server error."})


@app.get("/health")
def health():
    return {"status": "online"}


@app.post("/analyze")
async def api_analyze(pdf: UploadFile = File(...), dpi: int = RENDER_DPI):
    t0 = time.time()
    print(f"[REQUEST] POST /analyze - file={pdf.filename}")
    pdf_bytes = await pdf.read()
    result, status_code = await run_in_threadpool(analyze_pdf_core, pdf_bytes, dpi)
    print(f"[DONE] /analyze in {time.time() - t0:.2f}s - status={result['status']}")
    return JSONResponse(status_code=status_code, content=result)


@app.post("/signature")
async def api_signature(signature: UploadFile = File(...)):
    t0 = time.time()
    print(f"[REQUEST] POST /signature - file={signature.filename}")
    img_bytes = await signature.read()
    result, status_code = await run_in_threadpool(remove_background_core, img_bytes)
    print(f"[DONE] /signature in {time.time() - t0:.2f}s - status={result['status']}")
    return JSONResponse(status_code=status_code, content=result)


# ============================================================
# Gradio UI (secondary - the REST API above is the primary interface)
# ============================================================
def gradio_analyze(pdf_file):
    if pdf_file is None:
        return {"status": "error", "message": "No PDF uploaded."}
    with open(pdf_file, "rb") as f:
        pdf_bytes = f.read()
    result, _ = analyze_pdf_core(pdf_bytes)
    return result


def gradio_remove_bg(image_path):
    if image_path is None:
        return None
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    result, _ = remove_background_core(img_bytes)
    if result["status"] == "success":
        return Image.open(io.BytesIO(base64.b64decode(result["image"])))
    gr.Warning(result.get("message", "Background removal failed."))
    return None


with gr.Blocks(title="SignSpot AI Detection Server") as demo:
    gr.Markdown(
        "# SignSpot AI Detection Server\n"
        "This Space is built to be used as a REST API — see **/docs** for the full "
        "OpenAPI reference (`/health`, `/analyze`, `/signature`). "
        "The tabs below are for quick manual testing only."
    )
    with gr.Tab("Analyze PDF"):
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        analyze_btn = gr.Button("Analyze", variant="primary")
        json_output = gr.JSON(label="Detected Signature Fields")
        analyze_btn.click(fn=gradio_analyze, inputs=pdf_input, outputs=json_output)
    with gr.Tab("Remove Signature Background"):
        sig_input = gr.Image(label="Upload Signature", type="filepath")
        remove_btn = gr.Button("Remove Background", variant="primary")
        sig_output = gr.Image(label="Transparent PNG Preview", type="pil", image_mode="RGBA")
        remove_btn.click(fn=gradio_remove_bg, inputs=sig_input, outputs=sig_output)

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
