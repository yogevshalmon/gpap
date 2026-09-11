"""Render the README's CPU-vs-GPU propagation animation, in the brand mark's own pixel language.

The cell sprite is lifted from the 48px grid of `icon/gpap_icon.png` (8x8 rounded cell on a 9px
pitch), and the eyes sit where `icon_wake` puts them: +2 and +5 across, +4 down. A cell asleep is
un-propagated; a cell awake has been reached by the implication wave. One row is one partial
assignment, so the CPU panel runs a single row and the GPU panel runs the whole card at once.

Scaling is nearest-neighbour at an integer factor, per the kit's rules 1 and 2.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]

DIE     = (0x2C, 0x12, 0x5A)
OUTLINE = (0x18, 0x0A, 0x2E)
GOLD    = (0xE4, 0xC4, 0x70)
CREAM   = (0xF5, 0xF0, 0xE8)
AWAKE   = (0xDE, 0xCE, 0xFF)
HI      = (0xF2, 0xEA, 0xFF)
ASLEEP  = (0x64, 0x3C, 0xBA)
EYE     = (0x1E, 0x0C, 0x3C)
LAV     = (0xBA, 0x9A, 0xFF)

CELL, PITCH, SCALE = 8, 9, 2
COLS, GPU_ROWS = 18, 12
MONO  = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONOB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def cell_mask():
    """The rounded-cell footprint, read straight off the mark rather than redrawn."""
    icon = Image.open(ROOT / "docs/assets/icon/gpap_icon.png").convert("RGBA")
    icon = icon.resize((48, 48), Image.NEAREST).load()
    return [[icon[11 + x, 11 + y][:3] not in (DIE, OUTLINE) and icon[11 + x, 11 + y][3] == 255
             for x in range(CELL)] for y in range(CELL)]


MASK = cell_mask()


def draw_cell(px, ox, oy, awake):
    body = AWAKE if awake else ASLEEP
    for y in range(CELL):
        for x in range(CELL):
            if MASK[y][x]:
                px[ox + x, oy + y] = body
    if awake:                                   # the mark's own top-left highlight
        for dx in (1, 2):
            px[ox + dx, oy + 1] = HI
    if awake:                                   # open eyes: one pixel each
        px[ox + 2, oy + 4] = EYE
        px[ox + 5, oy + 4] = EYE
    else:                                       # asleep: a closed lid, two pixels wide
        for dx in (1, 2, 5, 6):
            px[ox + dx, oy + 4] = EYE


PIN = 3          # pin length, in 48-grid pixels


def panel(rows, wave, jitter, ground):
    """One chip die: `rows` rows of COLS cells, lit up to each row's own wave front.

    Drawn on the 48px grid and scaled up whole, so the pins protrude past the die edge the way
    they do on the mark instead of landing on top of the first and last cell.
    """
    w, h = COLS * PITCH + 1, rows * PITCH + 1
    im = Image.new("RGB", (w, h + 2 * PIN), ground)
    d = ImageDraw.Draw(im)
    d.rectangle([0, PIN, w - 1, PIN + h - 1], fill=DIE)
    for c in range(1, COLS, 3):                  # gold pins, top and bottom, clear of the cells
        x = c * PITCH + 3
        d.rectangle([x, 0, x + 2, PIN], fill=GOLD)
        d.rectangle([x, PIN + h - 1, x + 2, h + 2 * PIN - 1], fill=GOLD)
    px = im.load()
    for r in range(rows):
        front = wave - jitter[r % len(jitter)]
        for c in range(COLS):
            draw_cell(px, c * PITCH + 1, PIN + r * PITCH + 1, c < front)
    return im.resize((im.width * SCALE, im.height * SCALE), Image.NEAREST)


def build(dark: bool):
    ground = (0x16, 0x10, 0x2B) if dark else CREAM
    ink    = CREAM if dark else (0x2C, 0x12, 0x5A)
    faint  = LAV if dark else (0x64, 0x3C, 0xBA)

    head = ImageFont.truetype(MONOB, 15)
    sub  = ImageFont.truetype(MONO, 13)

    pw = (COLS * PITCH + 1) * SCALE
    gh = (GPU_ROWS * PITCH + 1 + 2 * PIN) * SCALE
    gap, padx, top = 56, 26, 36
    W = padx * 2 + pw * 2 + gap
    H = top + gh + 20

    jitter = [0, 1, 0, 2, 1, 0, 1, 2, 0, 1, 0, 2]
    frames, delays = [], []
    steps = COLS + 5
    for step in range(steps):
        im = Image.new("RGB", (W, H), ground)
        d = ImageDraw.Draw(im)
        wave = min(step, COLS)

        cpu = panel(1, wave, [0], ground)
        gpu = panel(GPU_ROWS, wave, jitter, ground)
        im.paste(cpu, (padx, top + (gh - cpu.height) // 2))
        im.paste(gpu, (padx + pw + gap, top))

        d.text((padx, 9), "CPU", font=head, fill=ink)
        d.text((padx + 46, 11), "one partial assignment at a time", font=sub, fill=faint)
        d.text((padx + pw + gap, 9), "GPU", font=head, fill=ink)
        d.text((padx + pw + gap + 46, 11), "thousands, in the same launch",
               font=sub, fill=faint)

        frames.append(im)
        delays.append(70 if step < COLS else (900 if step == COLS else 260))

    pal = frames[-1].quantize(colors=32, method=Image.MEDIANCUT)
    frames = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
    out = ROOT / f"docs/assets/animation/gpap_propagate_{'dark' if dark else 'light'}.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=delays,
                   loop=0, optimize=True, disposal=1)
    print(f"{out.name}  {W}x{H}  {len(frames)} frames  {out.stat().st_size // 1024} KB")


for dark in (True, False):
    build(dark)
