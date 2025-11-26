import io
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from src.models.unet import build_unet

app = FastAPI(title="Brain Tumor Segmentation API", version="1.0")

WEIGHTS_PATH = Path("./weights/best_model_unet.h5").expanduser().resolve()
IMAGE_SIZE = 256
INPUT_CHANNELS = 1
NUM_CLASSES = 4


@lru_cache(maxsize=1)
def load_model():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Model weights not found at {WEIGHTS_PATH}")
    model = build_unet(
        input_size=(IMAGE_SIZE, IMAGE_SIZE, INPUT_CHANNELS),
        num_classes=NUM_CLASSES,
        base_filters=32,
    )
    model.load_weights(WEIGHTS_PATH)
    return model


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("L")
    except Exception as exc:  # pillow-specific errors vary
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    array = np.asarray(image).astype("float32") / 255.0
    array = np.expand_dims(array, axis=(0, -1))  # shape: (1, H, W, 1)
    return array


def postprocess_mask(prediction: np.ndarray) -> Image.Image:
    """Convert softmax output to single-channel mask PNG."""
    # prediction shape: (1, H, W, num_classes)
    class_map = np.argmax(prediction, axis=-1)[0].astype("uint8")
    mask = Image.fromarray(class_map, mode="L")
    return mask


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    model = load_model()
    file_bytes = file.file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    input_tensor = preprocess_image(file_bytes)
    prediction = model.predict(input_tensor)
    mask_img = postprocess_mask(prediction)

    buffer = io.BytesIO()
    mask_img.save(buffer, format="PNG")
    buffer.seek(0)

    headers = {"Content-Disposition": f"inline; filename=\"mask_{file.filename or 'output'}.png\""}
    return StreamingResponse(buffer, media_type="image/png", headers=headers)


@app.post("/predict-json")
def predict_json(file: UploadFile = File(...)):
    """Alternative JSON response returning mask values."""
    model = load_model()
    file_bytes = file.file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    input_tensor = preprocess_image(file_bytes)
    prediction = model.predict(input_tensor)
    class_map = np.argmax(prediction, axis=-1)[0].astype(int).tolist()
    return JSONResponse({"mask": class_map})
