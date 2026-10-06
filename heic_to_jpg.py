"""Convert iPhone HEIC photos to JPG next to the originals, applying the rotation flag.

    python heic_to_jpg.py photos/
"""
import sys
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()
for p in sorted(Path(sys.argv[1]).iterdir()):
    if p.suffix.lower() in (".heic", ".heif"):
        ImageOps.exif_transpose(Image.open(p)).convert("RGB").save(p.with_suffix(".jpg"), quality=95)
        print(p.with_suffix(".jpg"))
