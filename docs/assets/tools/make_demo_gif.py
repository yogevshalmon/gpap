"""Render the README's terminal demo GIF from a verbatim `gpap sample` cast.

The text below is copied from cast_verified.txt, a real run on this box (RTX 4080 Laptop).
Nothing is invented: the only edit is the temp CNF path, which is a real mktemp name.
Palette is the brand kit's (INTEGRATION.md §7 rule 3).
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]

# --- brand palette ------------------------------------------------------------------------
BG       = (0x16, 0x10, 0x2B)
BAR      = (0x20, 0x16, 0x42)
BORDER   = (0x2C, 0x12, 0x5A)
GOLD     = (0xE4, 0xC4, 0x70)
CREAM    = (0xF5, 0xF0, 0xE8)
LAV_HI   = (0xDE, 0xCE, 0xFF)
LAV      = (0xBA, 0x9A, 0xFF)
VIOLET   = (0x8E, 0x5E, 0xFA)
MUTED    = (0x9C, 0x8F, 0xBA)
LEAF     = (0x8A, 0xB0, 0x92)
DIM      = (0x6B, 0x5D, 0x8A)

FS = 15
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONOB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
font = ImageFont.truetype(MONO, FS)
fontb = ImageFont.truetype(MONOB, FS)

CW = font.getlength("M")            # monospace advance
LH = 22
PAD_X, PAD_Y = 26, 16
BAR_H = 30

# The command shown is the command that produced the output shown. `gpap sample` prints the
# sound-check clause whether or not --verify-sound was passed, so it stays, dimmed.
CMD_LINES = ["gpap sample b10.bench --cycles 25 -n 50000 --target rare --out stim.out"]

# (segments, delay_ms) — segments are (text, colour, bold). Verbatim from cast_rare.txt; the two
# long lines are wrapped with a hanging indent, the way the terminal wrapped them.
OUT = [
    ([("b10.bench -> /tmp/tmpijhp30mx.cnf: ", MUTED, 0),
      ("PI=11 PO=6 DFF=17 gates=172", LAV, 0), (" | cycles=25 enc=tseitin", MUTED, 0)], 40),
    ([("  target=rare(freq~1.03e-02) -> ", MUTED, 0),
      ("vars=7851 clauses=19682 support=275", LAV, 0)], 420),
    ([("  encoded in 0.1s [tseitin]", MUTED, 0)], 300),
    ([("  portfolio [circuit]: random order, B=8; random order, B=1", MUTED, 0)], 900),
    ([("b10.bench: ", MUTED, 0), ("N=7851 M=19682 |sup|=275", LAV, 0),
      (" (269 active after simplify)", MUTED, 0)], 1600),
    ([("  ", MUTED, 0), ("50,464 models", LAV_HI, 1), (" in ", MUTED, 0),
      ("6.6s", LAV_HI, 1), (" e2e (4.1s sampling)", MUTED, 0)], 340),
    ([("  ", MUTED, 0), ("20,989 samples/s", VIOLET, 1), (" (steady-state), ", MUTED, 0),
      ("7,601/s", LAV, 1), (" end-to-end | success 2%", MUTED, 0)], 40),
    ([("         | sound-check not run (--verify-sound)", DIM, 0)], 600),
    ([("  mined 86 units / 0 binary / 0 ternary in 0.6s", MUTED, 0)], 700),
    ([("  wrote ", MUTED, 0), ("50,000 samples", LAV_HI, 1),
      (" (275 cols) -> stim.out [bits]", MUTED, 0)], 1000),
]


CMD_ROWS = [([("$ " if i == 0 else "> ", GOLD if i == 0 else DIM, 1)], line)
            for i, line in enumerate(CMD_LINES)]
n_cols = max([len(c) + 2 for c in CMD_LINES]
             + [sum(len(t) for t, _, _ in segs) for segs, _ in OUT])
W = int(PAD_X * 2 + n_cols * CW) + 2
H = PAD_Y * 2 + BAR_H + LH * (len(OUT) + len(CMD_ROWS) + 2) + 6


def blank():
    im = Image.new("RGB", (W, H), BORDER)
    d = ImageDraw.Draw(im)
    d.rectangle([1, 1, W - 2, H - 2], fill=BG)
    d.rectangle([1, 1, W - 2, BAR_H], fill=BAR)
    for i, c in enumerate((GOLD, VIOLET, LEAF)):
        x = 18 + i * 18
        d.ellipse([x, BAR_H // 2 - 4, x + 8, BAR_H // 2 + 4], fill=c)
    d.text((W // 2, BAR_H // 2), "gpap — sampling a rare reachable state of b10, 25 cycles",
           font=ImageFont.truetype(MONO, 12), fill=DIM, anchor="mm")
    return im, d


def draw_line(d, row, segs):
    x = PAD_X
    y = PAD_Y + BAR_H + row * LH
    for text, colour, bold in segs:
        d.text((x, y), text, font=fontb if bold else font, fill=colour)
        x += len(text) * CW


def cursor(d, row, col):
    x = PAD_X + col * CW
    y = PAD_Y + BAR_H + row * LH
    d.rectangle([x, y + 2, x + CW - 1, y + FS + 3], fill=LAV)


def draw_cmd(d, upto_row=None, upto_col=None):
    """The command as typed so far: whole rows above, a prefix on the row being typed."""
    for r, (pr, text) in enumerate(CMD_ROWS):
        if upto_row is not None and r > upto_row:
            break
        cut = len(text) if (upto_row is None or r < upto_row) else upto_col
        draw_line(d, r, pr + [(text[:cut], CREAM, 0)])


frames, delays = [], []

# 1. typing the command, two characters a frame, across the shell continuation
for r, (_, text) in enumerate(CMD_ROWS):
    for i in range(0, len(text) + 1, 2):
        im, d = blank()
        draw_cmd(d, r, i)
        cursor(d, r, 2 + i)
        frames.append(im)
        delays.append(300 if (r == 0 and i == 0) else 32)
    delays[-1] = 260
delays[-1] = 450

# 2. output lines, one frame each, held for their own delay
NC = len(CMD_ROWS)
shown = []
for segs, delay in OUT:
    shown.append(segs)
    im, d = blank()
    draw_cmd(d)
    for r, s in enumerate(shown):
        draw_line(d, r + NC, s)
    frames.append(im)
    delays.append(delay)

# 3. the prompt comes back, cursor blinking over the finished screen
for on, ms in ((1, 1400), (0, 520), (1, 520), (0, 520), (1, 1600)):
    im, d = blank()
    draw_cmd(d)
    for r, s in enumerate(shown):
        draw_line(d, r + NC, s)
    draw_line(d, len(shown) + NC + 1, [("$ ", GOLD, 1)])
    if on:
        cursor(d, len(shown) + NC + 1, 2)
    frames.append(im)
    delays.append(ms)

pal = frames[0].quantize(colors=64, method=Image.MEDIANCUT)
frames = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
out = str(ROOT / "docs/assets/demo/gpap_demo.gif")
frames[0].save(out, save_all=True, append_images=frames[1:], duration=delays,
               loop=0, optimize=True, disposal=1)
print(f"{out}  {W}x{H}  {len(frames)} frames  {sum(delays)/1000:.1f}s")
