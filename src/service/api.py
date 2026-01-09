import base64
import io
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
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


def decode_vertex_instance(instance: Any) -> bytes:
    if isinstance(instance, str):
        payload = instance
    elif isinstance(instance, dict):
        if "b64" in instance:
            payload = instance["b64"]
        elif "image_bytes" in instance and isinstance(instance["image_bytes"], dict):
            payload = instance["image_bytes"].get("b64")
        else:
            raise HTTPException(status_code=400, detail="Unsupported instance format")
    else:
        raise HTTPException(status_code=400, detail="Unsupported instance type")

    if not payload:
        raise HTTPException(status_code=400, detail="Missing base64 payload")

    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {exc}")


def validate_prediction(prediction: np.ndarray):
    if prediction.ndim != 4 or prediction.shape[-1] != NUM_CLASSES:
        raise HTTPException(status_code=400, detail="Model output has unexpected shape")


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
    validate_prediction(prediction)
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
    validate_prediction(prediction)
    class_map = np.argmax(prediction, axis=-1)[0].astype(int).tolist()
    return JSONResponse({"mask": class_map})


@app.post("/vertex/predict")
def vertex_predict(payload: Dict[str, Any] = Body(...)):
    """Vertex AI-compatible prediction endpoint using base64-encoded image bytes."""
    model = load_model()
    instances = payload.get("instances")
    if not isinstance(instances, list) or not instances:
        raise HTTPException(status_code=400, detail="Payload must include non-empty 'instances' list")

    predictions = []
    for instance in instances:
        file_bytes = decode_vertex_instance(instance)
        input_tensor = preprocess_image(file_bytes)
        prediction = model.predict(input_tensor, verbose=0)
        validate_prediction(prediction)
        class_map = np.argmax(prediction, axis=-1)[0].astype(int).tolist()
        predictions.append({"mask": class_map})

    return JSONResponse({"predictions": predictions})
