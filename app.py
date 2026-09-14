import streamlit as st
from ultralytics import YOLO
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.ops import batched_nms
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
from PIL import ImageDraw
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
YOLO_MODEL_PATH = ROOT_DIR / "runs" / "detect" / "train-3" / "weights" / "best.pt"
SSD_MODEL_PATHS = (
    ROOT_DIR / "ssd_model.pth",
    ROOT_DIR / "runs" / "ssd" / "ssd_best.pth",
)
SSD_CLASSES = {1: "With Helmet", 2: "Without Helmet"}
SSD_CONFIDENCE_THRESHOLD = 0.25
SSD_CLASS_THRESHOLDS = {1: 0.25, 2: 0.15}
SSD_MAX_DETECTIONS = 2

TEXT = {
    "English": {
        "title": "Helmet Detection App",
        "caption": "Upload a bike image and the model will classify whether the rider is wearing a helmet or not.",
        "single": "Single Image",
        "folder": "Folder Upload",
        "camera": "Webcam",
        "language": "Language",
        "upload": "Choose an image",
        "folder_upload": "Upload multiple images",
        "camera_upload": "Take a picture",
        "detecting": "Detecting helmet status...",
        "helmet": "✅ Helmet Detected",
        "no_helmet": "⚠️ No Helmet Detected",
        "no_object": "❌ No object detected",
        "unknown": "⚠️ Unknown Detection",
        "detail_helmet": "The rider appears to be wearing a helmet.",
        "detail_no_helmet": "The rider appears to be without a helmet.",
        "detail_no_object": "No rider or helmet-related object was detected in the image.",
        "detail_unknown": "The model detected an object but not a helmet class.",
        "confidence": "Confidence",
        "result": "Detection Result",
        "uploaded": "Uploaded Image",
        "processing": "Processing image...",
        "status": "Status",
        "no_files": "No files selected.",
        "yolo_output": "YOLO output",
        "ssd_output": "SSD output",
        "ssd_missing": "SSD model not found. Run the SSD training cell in the notebook first.",
        "model_unavailable": "Model unavailable",
        "no_ssd_detection": "SSD did not detect an object."
    },
    "اردو": {
        "title": "ایپلیکیشن شناخت ہیلمیٹ",
        "language": "زبان",
        "upload": "تصویر منتخب کریں",
        "folder_upload": "کئی تصاویر اپ لوڈ کریں",
        "camera_upload": "تصویر لیں",
        "detecting": "ہیلمیٹ کی شناخت جاری ہے...",
        "helmet": "✅ ہیلمیٹ موجود ہے",
        "no_helmet": "⚠️ ہیلمیٹ نہیں ہے",
        "no_object": "❌ کوئی چیز نہیں ملی",
        "unknown": "⚠️ غیر معلوم شناخت",
        "detail_helmet": "سوار ہیلمیٹ پہنے ہوئے دکھائی دیتا ہے۔",
        "detail_no_helmet": "سوار کے سر پر ہیلمیٹ نہیں ہے۔",
        "detail_no_object": "تصویر میں سوار یا ہیلمیٹ سے متعلق کوئی چیز نہیں ملی۔",
        "detail_unknown": "ماڈل نے کوئی چیز تو پہچانی لیکن ہیلمیٹ کلاس نہیں ملی۔",
        "confidence": "اعتماد کا درجہ",
        "result": "شناختی نتیجہ",
        "uploaded": "اپ لوڈ شدہ تصویر",
        "processing": "تصویر پر کام جاری ہے...",
        "status": "حالت",
        "no_files": "کوئی فائل منتخب نہیں ہوئی۔",
        "yolo_output": "YOLO نتیجہ",
        "ssd_output": "SSD نتیجہ",
        "ssd_missing": "SSD ماڈل نہیں ملا۔ پہلے نوٹ بک میں SSD training cell چلائیں۔",
        "model_unavailable": "ماڈل دستیاب نہیں",
        "no_ssd_detection": "SSD نے کوئی چیز شناخت نہیں کی۔"
    }
}

def get_text(lang):
    return TEXT.get(lang, TEXT["English"])


@st.cache_resource
def load_yolo_model():
    if not YOLO_MODEL_PATH.exists():
        return None
    return YOLO(str(YOLO_MODEL_PATH))


@st.cache_resource
def load_ssd_model():
    checkpoint_path = next((path for path in SSD_MODEL_PATHS if path.exists()), None)
    if checkpoint_path is None:
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    def create_model(weights_backbone):
        model = ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=weights_backbone,
        )
        classification_head = model.head.classification_head
        in_channels = [layer[0][0].in_channels for layer in classification_head.module_list]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDLiteClassificationHead(
            in_channels,
            num_anchors,
            3,
            torch.nn.BatchNorm2d,
        )
        return model

    model = create_model(None)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model = create_model("DEFAULT")
        model.load_state_dict(state_dict)
    model.eval()
    return model


yolo_model = load_yolo_model()
ssd_model = load_ssd_model()

st.set_page_config(page_title="Helmet Detector", page_icon="🧢", layout="centered")

lang = st.selectbox("Language", ["English", "اردو"], index=0)
text = get_text(lang)

st.title(text["title"])
st.caption(text["caption"])


def normalize_label(label):
    label = str(label).strip()
    lowered = label.lower()
    if "without" in lowered or "no helmet" in lowered:
        return "no_helmet"
    if "helmet" in lowered:
        return "helmet"
    return "unknown"


def analyze_image(image):
    if yolo_model is None:
        yolo_output = {
            "label": "unavailable",
            "status": text["model_unavailable"],
            "detail": "YOLO weights are not available in this deployment.",
            "confidence": 0.0,
            "annotated": image,
        }
        return {"yolo": yolo_output, "ssd": analyze_with_ssd(image)}

    image_np = np.array(image)
    with st.spinner(text["detecting"]):
        yolo_results = yolo_model(image_np, imgsz=640, conf=0.25)

    result = yolo_results[0]
    names = result.names if hasattr(result, "names") else {}
    boxes = result.boxes

    detected_classes = []
    confidences = []
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            if isinstance(names, dict):
                label = names.get(cls_id, str(cls_id))
            elif isinstance(names, (list, tuple)):
                label = names[cls_id] if cls_id < len(names) else str(cls_id)
            else:
                label = str(cls_id)
            conf = float(box.conf[0])
            detected_classes.append(label)
            confidences.append(conf)

    if not detected_classes:
        yolo_output = {
            "label": "no_object",
            "status": text["no_object"],
            "detail": text["detail_no_object"],
            "confidence": 0.0,
            "annotated": image,
        }
    else:
        top_label, top_conf = max(zip(detected_classes, confidences), key=lambda x: x[1])
        top_label_key = normalize_label(top_label)
        if top_label_key == "helmet":
            status = text["helmet"]
            detail = text["detail_helmet"]
        elif top_label_key == "no_helmet":
            status = text["no_helmet"]
            detail = text["detail_no_helmet"]
        else:
            status = text["unknown"]
            detail = text["detail_unknown"]
        yolo_output = {
            "label": top_label_key,
            "status": status,
            "detail": detail,
            "confidence": float(top_conf),
            "annotated": result.plot(),
        }

    return {"yolo": yolo_output, "ssd": analyze_with_ssd(image)}


def analyze_with_ssd(image):
    if ssd_model is None:
        return {
            "label": "unavailable",
            "status": text["model_unavailable"],
            "detail": text["ssd_missing"],
            "confidence": 0.0,
            "annotated": image,
        }

    image_tensor = pil_to_tensor(image).float() / 255.0
    with torch.inference_mode():
        prediction = ssd_model([image_tensor])[0]

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    valid_detections = []
    image_width, image_height = image.size
    image_area = image_width * image_height
    boxes = prediction["boxes"]
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    valid_indices = box_areas <= image_area * 0.35
    box_tops = boxes[:, 1]
    valid_indices &= box_tops <= image_height * 0.65
    keep_indices = batched_nms(
        boxes[valid_indices],
        prediction["scores"][valid_indices],
        prediction["labels"][valid_indices],
        iou_threshold=0.30,
    )
    candidates = [
        (
            boxes[valid_indices][index],
            prediction["labels"][valid_indices][index],
            prediction["scores"][valid_indices][index],
        )
        for index in keep_indices
    ]
    best_by_class = {}
    for candidate in candidates:
        class_id = int(candidate[1])
        if class_id not in best_by_class:
            best_by_class[class_id] = candidate
    candidates = sorted(
        best_by_class.values(),
        key=lambda detection: float(detection[2]),
        reverse=True,
    )[:SSD_MAX_DETECTIONS]
    for box, label_id, confidence in candidates:
        confidence_value = float(confidence)
        label_id = int(label_id)
        threshold = SSD_CLASS_THRESHOLDS.get(label_id, SSD_CONFIDENCE_THRESHOLD)
        if confidence_value < threshold:
            continue
        label = SSD_CLASSES.get(label_id, str(label_id))
        x1, y1, x2, y2 = box.tolist()
        coordinates = (
            max(0, min(image_width, int(x1))),
            max(0, min(image_height, int(y1))),
            max(0, min(image_width, int(x2))),
            max(0, min(image_height, int(y2))),
        )
        if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
            continue
        color = "yellow" if label_id == 2 else "lime"
        draw.rectangle(coordinates, outline=color, width=3)
        draw.text(
            (coordinates[0] + 4, coordinates[1] + 4),
            f"{label} {confidence_value:.2f}",
            fill=color,
        )
        valid_detections.append((label, confidence_value))

    if not valid_detections:
        return {
            "label": "no_object",
            "status": text["no_object"],
            "detail": text["no_ssd_detection"],
            "confidence": 0.0,
            "annotated": annotated,
        }

    top_label, top_confidence = max(valid_detections, key=lambda detection: detection[1])
    label_key = normalize_label(top_label)
    if label_key == "helmet":
        status = text["helmet"]
        detail = text["detail_helmet"]
    elif label_key == "no_helmet":
        status = text["no_helmet"]
        detail = text["detail_no_helmet"]
    else:
        status = text["unknown"]
        detail = text["detail_unknown"]
    return {
        "label": label_key,
        "status": status,
        "detail": detail,
        "confidence": top_confidence,
        "annotated": annotated,
    }


def show_model_output(output, title):
    st.subheader(title)
    st.success(output["status"])
    st.write(output["detail"])
    st.metric(text["confidence"], f"{output['confidence'] * 100:.2f}%")
    st.image(output["annotated"], caption=text["result"], use_container_width=True)


mode = st.radio("Mode", [text["single"], text["folder"], text["camera"]], horizontal=True)

if mode == text["single"]:
    file = st.file_uploader(text["upload"], type=["png", "jpg", "jpeg"])
    if file is not None:
        image = Image.open(file).convert("RGB")
        st.image(image, caption=text["uploaded"], use_container_width=True)
        results = analyze_image(image)
        yolo_column, ssd_column = st.columns(2)
        with yolo_column:
            show_model_output(results["yolo"], text["yolo_output"])
        with ssd_column:
            show_model_output(results["ssd"], text["ssd_output"])

elif mode == text["folder"]:
    files = st.file_uploader(text["folder_upload"], type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if files:
        for uploaded in files:
            image = Image.open(uploaded).convert("RGB")
            st.subheader(uploaded.name)
            st.image(image, caption=text["uploaded"], use_container_width=True)
            results = analyze_image(image)
            yolo_column, ssd_column = st.columns(2)
            with yolo_column:
                show_model_output(results["yolo"], text["yolo_output"])
            with ssd_column:
                show_model_output(results["ssd"], text["ssd_output"])
            st.markdown("---")
    else:
        st.info(text["no_files"])

else:
    camera = st.camera_input(text["camera_upload"])
    if camera is not None:
        image = Image.open(camera).convert("RGB")
        st.image(image, caption=text["uploaded"], use_container_width=True)
        results = analyze_image(image)
        yolo_column, ssd_column = st.columns(2)
        with yolo_column:
            show_model_output(results["yolo"], text["yolo_output"])
        with ssd_column:
            show_model_output(results["ssd"], text["ssd_output"])
