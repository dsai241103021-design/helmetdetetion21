from pathlib import Path

root = Path(__file__).resolve().parent
images_root = root / 'images'
labels_root = root / 'labels'

for pattern in ('*.png', '*.jpg', '*.jpeg'):
    for file in images_root.glob(pattern):
        file.unlink()

for file in labels_root.glob('*.txt'):
    file.unlink()

print('Cleaned root image and label folders.')
print('images root entries:', sorted(p.name for p in images_root.iterdir()))
print('labels root entries:', sorted(p.name for p in labels_root.iterdir()))
