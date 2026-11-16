"""Contact sheet of ebscan PNGs (2 x 3 per sheet) for eyeballing. Usage: python -I cool/sheet.py OUT.png TIC ..."""
import sys
from pathlib import Path

from PIL import Image

D = Path("/home/antiraedus/Projects/oddball-worlds/outputs/planets_spike/cool/ebscan")
out, tics = sys.argv[1], sys.argv[2:]
ims = [Image.open(D / f"{t}.png") for t in tics if (D / f"{t}.png").exists()]
if ims:
    w, h = ims[0].size
    cols = 2
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, h * rows), "white")
    for i, im in enumerate(ims):
        sheet.paste(im.resize((w, h)), ((i % cols) * w, (i // cols) * h))
    sheet.save(out)
    print(out, len(ims))
