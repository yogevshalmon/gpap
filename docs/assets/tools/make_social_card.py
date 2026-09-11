"""Render the 1280x640 GitHub social-preview card the brand kit does not ship (INTEGRATION.md §4).

The mark is taken back to its native 48px grid (the 576px master is an exact 12x render, verified)
and re-scaled by an integer factor with nearest-neighbour, per the kit's rules 1 and 2.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]

DIE     = (0x2C, 0x12, 0x5A)
OUTLINE = (0x18, 0x0A, 0x2E)
GOLD    = (0xE4, 0xC4, 0x70)
CREAM   = (0xF5, 0xF0, 0xE8)
LAV     = (0xBA, 0x9A, 0xFF)
LAV_HI  = (0xDE, 0xCE, 0xFF)
LEAF    = (0x8A, 0xB0, 0x92)

W, H = 1280, 640
MONO  = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONOB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

card = Image.new("RGB", (W, H), DIE)
d = ImageDraw.Draw(card)

# a soft plum vignette so the flat ground does not read as a swatch
top = Image.new("RGB", (W, H), OUTLINE)
mask = Image.linear_gradient("L").resize((W, H)).rotate(0)
card = Image.composite(top, card, mask.point(lambda v: int(v * 0.55)))
d = ImageDraw.Draw(card)
d.rectangle([0, 0, W - 1, H - 1], outline=OUTLINE, width=8)

# --- the mark: 576 -> 48 (lossless) -> 384 (exactly 8x) ------------------------------------
mark = Image.open(ROOT / "docs/assets/icon_wake/gpap_icon_wake.png").convert("RGBA")
mark = mark.resize((48, 48), Image.NEAREST).resize((384, 384), Image.NEAREST)
card.paste(mark, (104, (H - 384) // 2), mark)

# --- wordmark ------------------------------------------------------------------------------
x = 560
d.text((x, 168), "GPAP", font=ImageFont.truetype(MONOB, 112), fill=CREAM)
d.text((x + 6, 298), "GPU-Parallel (partial) Assignment Propagation",
       font=ImageFont.truetype(MONO, 25), fill=LAV)
d.rectangle([x + 6, 348, x + 6 + 260, 353], fill=GOLD)

body = ImageFont.truetype(MONO, 23)
d.text((x + 6, 386), "Batched BCP over millions of partial", font=body, fill=LAV_HI)
d.text((x + 6, 416), "assignments, in one launch.", font=body, fill=LAV_HI)
d.text((x + 6, 458), "Guaranteed-valid CircuitSAT sampling.", font=body, fill=LEAF)

d.text((x + 6, 516), "github.com/yogevshalmon/gpap",
       font=ImageFont.truetype(MONO, 20), fill=(0x8A, 0x7C, 0xAD))

# a row of pins along the bottom edge, the mark's own motif
for i in range(14):
    px0 = 104 + i * 84
    d.rectangle([px0, H - 26, px0 + 34, H - 18], fill=(0x4A, 0x2C, 0x7A))

out = str(ROOT / "docs/assets/social/gpap_social_preview.png")
card.save(out)
print(out, card.size)
