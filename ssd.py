import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.transforms.functional import pil_to_tensor


ROOT = Path(__file__).resolve().parent
CLASS_TO_ID = {"With Helmet": 1, "Without Helmet": 2}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class HelmetDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        self.images = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        boxes, labels = [], []
        annotation_path = ROOT / "annotations" / f"{image_path.stem}.xml"

        if annotation_path.exists():
            root = ET.parse(annotation_path).getroot()
            for object_node in root.findall("object"):
                label = CLASS_TO_ID.get(object_node.findtext("name", default=""))
                box = object_node.find("bndbox")
                if label is None or box is None:
                    continue
                coordinates = [
                    float(box.findtext(name, default="0"))
                    for name in ("xmin", "ymin", "xmax", "ymax")
                ]
                if coordinates[2] > coordinates[0] and coordinates[3] > coordinates[1]:
                    boxes.append(coordinates)
                    labels.append(label)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index]),
        }
        return pil_to_tensor(image).float() / 255.0, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model():
    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone="DEFAULT",
    )
    classification_head = model.head.classification_head
    in_channels = [layer[0][0].in_channels for layer in classification_head.module_list]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels,
        num_anchors,
        len(CLASS_TO_ID) + 1,
        torch.nn.BatchNorm2d,
    )
    return model


def train(epochs, batch_size, learning_rate, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HelmetDataset(ROOT / "images" / "train")
    if not dataset:
        raise RuntimeError(f"No training images found in {dataset.image_dir}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )
    model = build_model().to(device)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.0005,
    )

    print(f"Training SSD on {len(dataset)} images using {device}.")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        average_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch + 1:02d}/{epochs} - loss: {average_loss:.4f}")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_to_id": CLASS_TO_ID,
                "epoch": epoch + 1,
                "loss": average_loss,
            },
            ROOT / "runs" / "ssd" / "ssd_latest.pth",
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_id": CLASS_TO_ID,
        },
        output_path,
    )
    print(f"SSD model saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the SSD helmet detector.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "ssd_model.pth",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        output_path=arguments.output,
    )