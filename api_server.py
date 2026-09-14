import base64
import io
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image, ImageDraw
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.ops import batched_nms
from torchvision.transforms.functional import pil_to_tensor
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent
YOLO_MODEL_PATH = ROOT_DIR / "runs" / "detect" / "train-3" / "weights" / "best.pt"
SSD_MODEL_PATHS = (ROOT_DIR / "ssd_model.pth", ROOT_DIR / "runs" / "ssd" / "ssd_best.pth")
SSD_CLASSES = {1: "With Helmet", 2: "Without Helmet"}
SSD_CLASS_THRESHOLDS = {1: 0.25, 2: 0.15}

app = Flask(__name__)


def normalize_label(label):
    lowered = str(label).strip().lower()
    if "without" in lowered or "no helmet" in lowered:
        return "no_helmet"
    if "helmet" in lowered:
        return "helmet"
    return "unknown"


def load_ssd_model():
    checkpoint_path = next((path for path in SSD_MODEL_PATHS if path.exists()), None)
    if checkpoint_path is None:
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    def create_model(weights_backbone):
        model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=weights_backbone)
        head = model.head.classification_head
        in_channels = [layer[0][0].in_channels for layer in head.module_list]
        anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDLiteClassificationHead(in_channels, anchors, 3, torch.nn.BatchNorm2d)
        return model

    model = create_model(None)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model = create_model("DEFAULT")
        model.load_state_dict(state_dict)
    model.eval()
    return model


print("Loading detection models...")
yolo_model = YOLO(str(YOLO_MODEL_PATH)) if YOLO_MODEL_PATH.exists() else None
ssd_model = load_ssd_model()
print("Detection API ready")


def as_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def yolo_prediction(image):
    if yolo_model is None:
        return {"label": "unavailable", "status": "Model unavailable", "detail": "YOLO weights were not found.", "confidence": 0, "annotated_image": as_data_url(image), "detections": []}
    result = yolo_model(np.array(image), imgsz=640, conf=0.25)[0]
    names = result.names if hasattr(result, "names") else {}
    detections = []
    for box in result.boxes or []:
        class_id = int(box.cls[0])
        label = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
        detections.append({"label": label, "confidence": float(box.conf[0])})
    if not detections:
        return {"label": "no_object", "status": "No object detected", "detail": "No rider or helmet-related object was detected.", "confidence": 0, "annotated_image": as_data_url(image), "detections": []}
    top = max(detections, key=lambda item: item["confidence"])
    label_key = normalize_label(top["label"])
    status = "Helmet Detected" if label_key == "helmet" else "No Helmet Detected" if label_key == "no_helmet" else "Unknown Detection"
    detail = "The rider appears to be wearing a helmet." if label_key == "helmet" else "The rider appears to be without a helmet." if label_key == "no_helmet" else "The model detected an object but not a helmet class."
    annotated = Image.fromarray(result.plot())
    return {"label": label_key, "status": status, "detail": detail, "confidence": top["confidence"], "annotated_image": as_data_url(annotated), "detections": detections}


def ssd_prediction(image):
    if ssd_model is None:
        return {"label": "unavailable", "status": "Model unavailable", "detail": "SSD weights were not found.", "confidence": 0, "annotated_image": as_data_url(image), "detections": []}
    tensor = pil_to_tensor(image).float() / 255.0
    with torch.inference_mode():
        prediction = ssd_model([tensor])[0]
    width, height = image.size
    areas = (prediction["boxes"][:, 2] - prediction["boxes"][:, 0]) * (prediction["boxes"][:, 3] - prediction["boxes"][:, 1])
    valid = (areas <= width * height * 0.35) & (prediction["boxes"][:, 1] <= height * 0.65)
    keep = batched_nms(prediction["boxes"][valid], prediction["scores"][valid], prediction["labels"][valid], 0.30)
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    detections = []
    seen = set()
    for index in keep:
        label_id = int(prediction["labels"][valid][index])
        score = float(prediction["scores"][valid][index])
        if label_id in seen or score < SSD_CLASS_THRESHOLDS.get(label_id, 0.25):
            continue
        seen.add(label_id)
        label = SSD_CLASSES.get(label_id, str(label_id))
        box = prediction["boxes"][valid][index].tolist()
        coordinates = tuple(max(0, min(limit, int(value))) for value, limit in zip(box, (width, height, width, height)))
        if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
            continue
        color = "yellow" if label_id == 2 else "lime"
        draw.rectangle(coordinates, outline=color, width=3)
        draw.text((coordinates[0] + 4, coordinates[1] + 4), f"{label} {score:.2f}", fill=color)
        detections.append({"label": label, "confidence": score})
    detections = sorted(detections, key=lambda item: item["confidence"], reverse=True)[:2]
    if not detections:
        return {"label": "no_object", "status": "No object detected", "detail": "SSD did not detect an object.", "confidence": 0, "annotated_image": as_data_url(annotated), "detections": []}
    top = detections[0]
    label_key = normalize_label(top["label"])
    status = "Helmet Detected" if label_key == "helmet" else "No Helmet Detected"
    detail = "The rider appears to be wearing a helmet." if label_key == "helmet" else "The rider appears to be without a helmet."
    return {"label": label_key, "status": status, "detail": detail, "confidence": top["confidence"], "annotated_image": as_data_url(annotated), "detections": detections}


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.get("/health")
def health():
    return jsonify({"status": "ok", "yolo": yolo_model is not None, "ssd": ssd_model is not None})


@app.post("/predict")
@app.post("/api/predict")
def predict():
    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify({"error": "file is required"}), 400
    try:
        image = Image.open(uploaded.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "file must be a valid image"}), 400
    model = request.form.get("model", "yolo").lower()
    if model == "ssd":
        return jsonify(ssd_prediction(image))
    return jsonify(yolo_prediction(image))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
