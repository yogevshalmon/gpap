"""Render the README's header lockup: the GPAP wordmark, then the mark.

An `<h1>` was the wrong container for this. GitHub rules a line under every h1, so an icon big
enough to read sat astride that rule and the whole header looked broken. A lockup is one image
instead: the wordmark, the mark beside it, and a TRANSPARENT ground so it sits correctly on any
page in either theme.

The wordmark is not a font rendering — it is a BITMAP. DejaVu Sans Bold is set at a size whose
cap height is 16 pixels, thresholded to 1-bit (no antialiasing, so no grey fringe), then scaled
up by a whole number with nearest-neighbour. So the letters are built the way the mark is, at the
same pixel size on screen, and hand-drawing four glyphs badly is not required to get there.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]

CREAM = (0xF5, 0xF0, 0xE8)
DIE   = (0x2C, 0x12, 0x5A)

WORD = "GPAP"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
PT = 22                    # cap height lands on 16 pixels
SCALE = 6                  # ...which scales to 96, the mark's height
MARK = 144                 # the mark: 3 x the kit's 48px grid, so it reads as the letters' sibling
GAP = 30                   # between the wordmark and the mark
PAD = 16


def wordmark(colour):
    """The word as a 1-bit bitmap, scaled up whole and tinted."""
    probe = Image.new("L", (PT * len(WORD) * 2, PT * 3), 0)
    ImageDraw.Draw(probe).text((PT, PT), WORD, font=ImageFont.truetype(FONT, PT), fill=255)
    bits = probe.crop(probe.getbbox()).point(lambda v: 255 if v > 110 else 0, mode="1")

    art = Image.new("RGBA", bits.size, (0, 0, 0, 0))
    px, src = art.load(), bits.load()
    for y in range(bits.height):
        for x in range(bits.width):
            if src[x, y]:
                px[x, y] = (*colour, 255)
    return art.resize((bits.width * SCALE, bits.height * SCALE), Image.NEAREST)


def build(dark: bool):
    word = wordmark(CREAM if dark else DIE)

    # The mark: back to its own 48px grid, then up by a whole factor (96 = 2 x 48).
    mark = Image.open(ROOT / "docs/assets/icon_wake/gpap_icon_wake.png").convert("RGBA")
    mark = mark.resize((48, 48), Image.NEAREST).resize((MARK, MARK), Image.NEAREST)

    h = max(word.height, MARK) + 2 * PAD
    w = PAD + word.width + GAP + MARK + PAD
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(word, (PAD, (h - word.height) // 2), word)
    out.paste(mark, (PAD + word.width + GAP, (h - MARK) // 2), mark)

    path = ROOT / f"docs/assets/wordmark/gpap_wordmark_{'dark' if dark else 'light'}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    print(f"{path.name}  {w}x{h}  {path.stat().st_size // 1024} KB")


for dark in (True, False):
    build(dark)
