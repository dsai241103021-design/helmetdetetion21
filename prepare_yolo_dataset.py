from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parent
images_root = root / "images"
annotations_root = root / "annotations"

split_names = ["train", "val", "test"]
image_split_dirs = {name: root / "images" / name for name in split_names}
label_split_dirs = {name: root / "labels" / name for name in split_names}

for folder in image_split_dirs.values():
    folder.mkdir(parents=True, exist_ok=True)

for folder in label_split_dirs.values():
    folder.mkdir(parents=True, exist_ok=True)

image_files = sorted(images_root.glob("*.png"))
if not image_files:
    raise FileNotFoundError(f"No PNG images found in {images_root}.")

classes = {}
for image_file in image_files:
    ann_path = annotations_root / f"{image_file.stem}.xml"
    if not ann_path.exists():
        continue
    tree = ET.parse(ann_path)
    root_node = tree.getroot()
    for obj in root_node.findall("object"):
        class_name = obj.findtext("name", default="")
        if class_name and class_name not in classes:
            classes[class_name] = len(classes)

if not classes:
    raise ValueError("No valid annotations were found in the XML files.")

train_count = int(len(image_files) * 0.8)
val_count = int(len(image_files) * 0.1)
remaining = len(image_files) - train_count - val_count
# Make the split deterministic and exact.
if remaining < 0:
    remaining = 0

splits = {
    "train": image_files[:train_count],
    "val": image_files[train_count:train_count + val_count],
    "test": image_files[train_count + val_count:train_count + val_count + remaining],
}

for split_name, files in splits.items():
    image_dir = image_split_dirs[split_name]
    label_dir = label_split_dirs[split_name]
    for image_file in files:
        shutil.copy2(image_file, image_dir / image_file.name)

        ann_path = annotations_root / f"{image_file.stem}.xml"
        label_path = label_dir / f"{image_file.stem}.txt"
        if ann_path.exists():
            tree = ET.parse(ann_path)
            ann_root = tree.getroot()
            size = ann_root.find("size")
            if size is None:
                label_path.write_text("")
                continue

            width = float(size.findtext("width", default="1"))
            height = float(size.findtext("height", default="1"))
            if width <= 0 or height <= 0:
                label_path.write_text("")
                continue

            lines = []
            for obj in ann_root.findall("object"):
                name = obj.findtext("name", default="")
                if not name:
                    continue
                class_id = classes.get(name)
                if class_id is None:
                    class_id = len(classes)
                    classes[name] = class_id

                bnd = obj.find("bndbox")
                if bnd is None:
                    continue

                xmin = float(bnd.findtext("xmin", default="0"))
                ymin = float(bnd.findtext("ymin", default="0"))
                xmax = float(bnd.findtext("xmax", default="0"))
                ymax = float(bnd.findtext("ymax", default="0"))

                if xmax <= xmin or ymax <= ymin:
                    continue

                x_center = ((xmin + xmax) / 2.0) / width
                y_center = ((ymin + ymax) / 2.0) / height
                box_w = (xmax - xmin) / width
                box_h = (ymax - ymin) / height

                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        else:
            label_path.write_text("", encoding="utf-8")

# Write YOLO data.yaml
class_names = [name for name, _ in sorted(classes.items(), key=lambda item: item[1])]
train_path = "images/train"
val_path = "images/val"
test_path = "images/test"

data_yaml = (
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n\n"
    "nc: " + str(len(class_names)) + "\n"
    "names: [" + ", ".join(f'"{name}"' for name in class_names) + "]\n"
)
(root / "data.yaml").write_text(data_yaml, encoding="utf-8")

print(f"Prepared dataset with {len(image_files)} images.")
print(f"Classes: {class_names}")
print("Split counts:")
for split_name, files in splits.items():
    print(f"  {split_name}: {len(files)} images")
