"""Render the README's core-idea animation: a formula propagated, then a whole batch of them.

Three acts, told in the brand mark's own pixel language — the berry is the kit's 8x8 cell with
its L-shaped highlight and shadowed bottom edge, the bunch is staggered the way the kit's story
grape is, and the stem curls into a leaf between the top berries.

  1. THE FORMULA, drawn as the grape: every row of the bunch is a clause, every berry a literal.
     What plays is an actual BCP trace. A DECISION lights one literal (gold flash). Its negation
     KNOCKS OUT that literal wherever it occurs — the berry shrinks and darkens, `sat_count -= 1`.
     A clause down to one survivor FORCES it (white flash): unit propagation. Fixpoint, next
     decision. It ends with every clause satisfied by exactly one lit literal and most of the
     formula never touched — which is what "partial assignment" means.
  2. THE FOLD: under one assignment, a clause's whole state is one packed word. Each clause row
     collapses onto its lit literal, and the words line up as that assignment's row.
  3. THE BATCH: the engine's state is exactly that matrix — one row per partial assignment, one
     column per clause. The die grows a row at a time, and every row propagates in the same
     launch, each in its own order.

One scale throughout, 6x the 8px cell (a knocked-out literal sits at 5x; kit rules 1 and 2), at
the kit's own 50-140 ms frame cadence.
"""
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]

DIE     = (0x2C, 0x12, 0x5A)
OUTLINE = (0x18, 0x0A, 0x2E)
GOLD    = (0xE4, 0xC4, 0x70)
GOLD_HI, GOLD_LO = (0xF6, 0xE3, 0xA4), (0xB4, 0x94, 0x48)
CREAM   = (0xF5, 0xF0, 0xE8)
EYE     = (0x1E, 0x0C, 0x3C)
# The mark's cell ramp, with a near-white above it for highlights and the shadow shade below.
RAMP = [(0xFF, 0xFF, 0xFF), (0xF4, 0xEE, 0xFF), (0xDE, 0xCE, 0xFF), (0xBA, 0x9A, 0xFF),
        (0x8E, 0x5E, 0xFA), (0x64, 0x3C, 0xBA), (0x48, 0x26, 0x82)]
STEM_HI, STEM_LO = (0x62, 0x74, 0x34), (0x42, 0x4E, 0x22)
LEAF_HI, LEAF_LO = (0x8A, 0xB0, 0x92), (0x5E, 0x82, 0x66)

CELL = 8
CLAUSES = [4, 5, 4, 3, 2]       # the bunch, widest just under the stem, no unit clause (a unit
M = len(CLAUSES)                # is already lit): M clauses = M columns of the state
K = 5                           # partial assignments = rows of the state

U = 6                           # 6x: one cell pixel is six on screen
SIZE, PITCH, GROW = CELL * U, CELL * U + U, 46      # rows of the bunch nest, so a shorter pitch
OUT_SIZE = CELL * (U - 1)                            # a knocked-out literal, one step smaller

# Act 1's script — a real unit-propagation trace over clauses A..E = 0..4. Occurrence lists
# are what a decision touches: `out` is the negated literal falsified in another clause.
TRACE = [
    ("decide", 0, 0),     # A0 := T                          A satisfied
    ("out", 2, 3),        #   ~A0 occurs in C: C3 is out
    ("out", 4, 0),        #   ~A0 occurs in E: E0 is out     E has one survivor...
    ("force", 4, 1),      #   ...so E1 := T                  E satisfied
    ("out", 3, 0),        #   ~E1 occurs in D: D0 is out
    ("out", 1, 1),        #   ~E1 occurs in B: B1 is out     fixpoint
    ("decide", 1, 3),     # B3 := T                          B satisfied
    ("out", 3, 2),        #   ~B3 occurs in D: D2 is out     D has one survivor...
    ("force", 3, 1),      #   ...so D1 := T                  D satisfied
    ("out", 2, 1),        #   ~D1 occurs in C: C1 is out     fixpoint
    ("decide", 2, 0),     # C0 := T                          C satisfied — every clause is
]

MONO  = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONOB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

# The kit's stem and leaf, in cell pixels, relative to the top centre of the bunch.
STEM = [(-1, -6, STEM_HI), (0, -6, STEM_HI), (-1, -5, STEM_HI), (0, -5, STEM_HI),
        (-1, -4, STEM_HI), (0, -4, STEM_HI), (-1, -3, STEM_LO), (0, -3, STEM_LO),
        (-1, -2, STEM_LO), (0, -2, STEM_LO), (-1, -1, STEM_LO), (0, -1, STEM_LO),
        (1, -7, LEAF_HI), (2, -7, LEAF_HI), (1, -6, LEAF_HI), (2, -6, LEAF_HI), (3, -6, LEAF_LO),
        (2, -5, LEAF_HI), (3, -5, LEAF_LO)]


def berry(body, hi, lo, eyes, size=SIZE):
    """The kit's cell: rounded footprint, L highlight top-left, shadow along the bottom edge."""
    t = Image.new("RGBA", (CELL, CELL), (0, 0, 0, 0))
    p = t.load()
    for y in range(CELL):
        for x in range(CELL):
            if (y in (0, CELL - 1)) and (x in (0, CELL - 1)):
                continue
            p[x, y] = (*body, 255)
    for x, y in ((1, 1), (2, 1), (1, 2)):
        p[x, y] = (*hi, 255)
    for x in range(2, 6):
        p[x, 7] = (*lo, 255)
    p[6, 6] = (*lo, 255)
    if eyes == "open":
        p[2, 4] = p[5, 4] = (*EYE, 255)
    elif eyes == "closed":
        for x in (1, 2, 5, 6):
            p[x, 4] = (*EYE, 255)
    return t.resize((size, size), Image.NEAREST)


ASLEEP    = berry(RAMP[5], RAMP[4], RAMP[6], "closed")           # unassigned
AWAKE     = berry(RAMP[2], RAMP[1], RAMP[3], "open")             # true
FLASH     = berry(RAMP[1], RAMP[0], RAMP[2], "open")             # the frame it is forced
DECIDE    = berry(GOLD, GOLD_HI, GOLD_LO, "open")                # the frame it is decided
OUT_FLASH = berry(RAMP[6], RAMP[5], OUTLINE, "closed")           # the frame it is knocked out
OUT       = berry(RAMP[6], RAMP[5], OUTLINE, "closed", OUT_SIZE)  # falsified: smaller, darker


def halo_for(tile):
    """The berry's footprint in the outline colour, one cell pixel larger all round."""
    u = tile.width // CELL
    small = tile.resize((CELL, CELL), Image.NEAREST)
    h = Image.new("RGBA", (CELL + 2, CELL + 2), (0, 0, 0, 0))
    for y in range(CELL):
        for x in range(CELL):
            if small.getpixel((x, y))[3]:
                h.putpixel((x + 1, y + 1), (*OUTLINE, 255))
    return h.resize(((CELL + 2) * u, (CELL + 2) * u), Image.NEAREST)


HALOS = {t.width: halo_for(t) for t in (ASLEEP, OUT)}


def blit(canvas, x, y, tile, halo=False):
    """Paste a berry centred in its slot. On bare ground the halo is the separation the die
    would give it, without which a pale berry washes out on the cream theme."""
    x, y = round(x) + (SIZE - tile.width) // 2, round(y) + (SIZE - tile.height) // 2
    if halo:
        u = tile.width // CELL
        h = HALOS[tile.width]
        canvas.paste(h, (x - u, y - u), h)
    canvas.paste(tile, (x, y), tile)


def ease(t):
    return t * t * (3 - 2 * t)


def schedule(n, steps, rng, span_lo=0.5, span_hi=1.0):
    """When each cell of one row wakes: its own random order, its own span."""
    order = list(range(n))
    rng.shuffle(order)
    span = steps * rng.uniform(span_lo, span_hi)
    wake = [0] * n
    for k, c in enumerate(order):
        wake[c] = 1 + round(k * (span - 1) / max(1, n - 1))
    return wake


def build(dark: bool):
    ground = (0x16, 0x10, 0x2B) if dark else CREAM
    ink    = CREAM if dark else DIE
    faint  = RAMP[3] if dark else RAMP[5]
    head   = ImageFont.truetype(MONOB, 17)
    sub    = ImageFont.truetype(MONO, 15)

    gap, pin = U, 3 * U                            # the mark: one-pixel gap, three-pixel pins
    board = M * PITCH - gap                        # the die is square: M columns, K rows
    frame_out = 2 * gap + pin                      # gap + outline + pin, beyond the cells
    grape_h = (len(CLAUSES) - 1) * GROW + SIZE
    stem_h = 7 * U

    W = 580
    cap_h = 54
    art_h = max(board + 2 * frame_out, grape_h + stem_h)
    H = cap_h + art_h + 22

    gtop = cap_h + (art_h - grape_h - stem_h) / 2 + stem_h
    gpos, clause_of, first = [], [], []
    for k, width in enumerate(CLAUSES):
        first.append(len(gpos))
        row_w = width * PITCH - gap
        for j in range(width):
            gpos.append((W / 2 - row_w / 2 + j * PITCH, gtop + k * GROW))
            clause_of.append(k)

    sx0 = W / 2 - board / 2
    sy0 = cap_h + (art_h - board - 2 * frame_out) / 2 + frame_out

    frames, delays = [], []

    def frame(caption, lead=""):
        im = Image.new("RGB", (W, H), ground)
        d = ImageDraw.Draw(im)
        if lead:
            lw = d.textlength(lead + "  ", font=head)
            cw = d.textlength(caption, font=sub)
            x0 = (W - lw - cw) / 2
            d.text((x0, 17), lead, font=head, fill=ink)
            d.text((x0 + lw, 19), caption, font=sub, fill=faint)
        else:
            d.text((W / 2, 19), caption, font=sub, fill=faint, anchor="ma")
        return im, d

    def stem(d):
        for dx, dy, colour in STEM:
            x, y = W / 2 + dx * U, gtop + dy * U
            d.rectangle([x, y, x + U - 1, y + U - 1], fill=colour)

    def bracket(d, k, k_from=None, t=1.0):
        """Gold brackets either side of clause k — the clause propagation is visiting. Between
        clauses they SLIDE (position and width interpolated) rather than crossfade: a blended
        gold has no home in the GIF palette and would snap to a dark ghost."""
        def geom(k):
            row_w = CLAUSES[k] * PITCH - gap
            return W / 2 - row_w / 2, gtop + k * GROW, row_w
        x0, y0, row_w = geom(k)
        if k_from is not None and t < 1:
            xa, ya, wa = geom(k_from)
            x0, y0, row_w = xa + (x0 - xa) * t, ya + (y0 - ya) * t, wa + (row_w - wa) * t
        xl, xr = x0 - 14, x0 + row_w + 8
        for x, tick in ((xl, xl + 13), (xr, xr - 8)):
            d.rectangle([x, y0 - 4, x + 5, y0 + SIZE + 3], fill=GOLD)
            d.rectangle([min(x, tick), y0 - 4, max(x + 5, tick), y0 - 1], fill=GOLD)
            d.rectangle([min(x, tick), y0 + SIZE, max(x + 5, tick), y0 + SIZE + 3], fill=GOLD)

    def die(d, rows):
        """The chip behind the state: outline, die, and a pin per cell on every side — the mark
        itself, grown to K x M."""
        h = rows * PITCH - gap
        d.rectangle([sx0 - 2 * gap, sy0 - 2 * gap, sx0 + board + 2 * gap - 1,
                     sy0 + h + 2 * gap - 1], fill=OUTLINE)
        d.rectangle([sx0 - gap, sy0 - gap, sx0 + board + gap - 1, sy0 + h + gap - 1], fill=DIE)
        for c in range(M):
            x = sx0 + c * PITCH + SIZE // 2 - U
            d.rectangle([x, sy0 - 2 * gap - pin, x + 2 * U - 1, sy0 - 2 * gap - 1], fill=GOLD)
            d.rectangle([x, sy0 + h + 2 * gap, x + 2 * U - 1, sy0 + h + 2 * gap + pin - 1],
                        fill=GOLD)
        for r in range(rows):
            y = sy0 + r * PITCH + SIZE // 2 - U
            d.rectangle([sx0 - 2 * gap - pin, y, sx0 - 2 * gap - 1, y + 2 * U - 1], fill=GOLD)
            d.rectangle([sx0 + board + 2 * gap, y, sx0 + board + 2 * gap + pin - 1, y + 2 * U - 1],
                        fill=GOLD)

    # --- act 1: the formula, a decision and its implications, one clause at a time -----------
    lit, out = set(), set()

    def grape(im, d, flash=None, k=None, k_from=None, t=1.0):
        stem(d)
        if k is not None:
            bracket(d, k, k_from, t)
        for i, (x, y) in enumerate(gpos):
            if flash and flash[0] == i:
                tile = flash[1]
            else:
                tile = AWAKE if i in lit else (OUT if i in out else ASLEEP)
            blit(im, x, y, tile, halo=True)

    cap1 = ("a decision, then its implications, one clause at a time", "CPU")
    im, d = frame(*cap1)
    grape(im, d)
    frames.append(im)
    delays.append(900)

    k_prev = None
    for kind, k, j in TRACE:
        i = first[k] + j
        if k != k_prev:                          # the bracket eases over to the clause touched
            for f in range(1, 5):
                im, d = frame(*cap1)
                grape(im, d, k=k, k_from=k_prev, t=ease(f / 4))
                frames.append(im)
                delays.append(60)
            k_prev = k
        pop, hold = {"decide": (DECIDE, 110), "force": (FLASH, 70), "out": (OUT_FLASH, 70)}[kind]
        im, d = frame(*cap1)
        grape(im, d, flash=(i, pop), k=k)
        frames.append(im)
        delays.append(hold)
        (out if kind == "out" else lit).add(i)
        im, d = frame(*cap1)
        grape(im, d, k=k)
        frames.append(im)
        delays.append(200 if kind == "decide" else 140)
    delays[-1] = 900
    for k in range(M):                           # every clause satisfied by exactly one literal
        assert sum(1 for i in range(first[k], first[k] + CLAUSES[k]) if i in lit) == 1

    # --- act 2: each clause folds into one word; the words are that assignment's row -----------
    # Two motions, not one: first every clause's berries slide together IN PLACE and merge onto
    # the literal that satisfies it (the clause's packed word), then the five words travel to
    # the row. Doing it in one diagonal sweep smeared eighteen overlapping sprites into a streak.
    cap2a = ("...under one assignment, a clause's whole state is one word",)
    cap2b = ("...and the assignment is one row of words",)
    # The bunch's rows nest closer than a cell is tall, so the merged column opens out to the
    # die's own pitch as it forms, or the five words would overlap.
    col_top = gtop + (grape_h - (M * PITCH - gap)) / 2
    centre = [(W / 2 - SIZE / 2, col_top + k * PITCH) for k in range(M)]
    draw_order = [i for i in range(len(gpos)) if i not in lit] + sorted(lit)   # lit on top

    n_a = 10
    for f in range(1, n_a + 1):
        t = ease(f / n_a)
        im, d = frame(*cap2a)
        stem(d)
        for i in draw_order:
            x, y = gpos[i]
            cx, cy = centre[clause_of[i]]
            tile = AWAKE if i in lit else (OUT if i in out else ASLEEP)
            if f == n_a:
                tile = FLASH
            blit(im, x + (cx - x) * t, y + (cy - y) * t, tile, halo=True)
        frames.append(im)
        delays.append(50)
    delays[-1] = 500

    n_b = 12
    for f in range(1, n_b + 1):
        t = ease(f / n_b)
        im, d = frame(*cap2b)
        if f == n_b:
            die(d, 1)
        for k, (cx, cy) in enumerate(centre):
            tx, ty = sx0 + k * PITCH, sy0
            blit(im, cx + (tx - cx) * t, cy + (ty - cy) * t, AWAKE, halo=f < n_b)
        frames.append(im)
        delays.append(50)
    delays[-1] = 700

    # --- act 3: the state matrix, one row per assignment ---------------------------------------
    cap3 = ("one row per partial assignment, one column per clause", "GPU")
    cap4 = ("every row propagating in the same launch", "GPU")

    def state(im, d, rows, tile_of):
        die(d, rows)
        for r in range(rows):
            for c in range(M):
                blit(im, sx0 + c * PITCH, sy0 + r * PITCH, tile_of(r, c))

    for n in range(2, K + 1):                    # the die grows a row at a time
        im, d = frame(*cap3)
        state(im, d, n, lambda r, c: AWAKE if r == 0 else ASLEEP)
        frames.append(im)
        delays.append(110)
    delays[-1] = 800

    rng = random.Random(11)
    steps = 12
    wake = [None] + [schedule(M, steps, rng) for _ in range(K - 1)]
    for step in range(steps + 2):
        im, d = frame(*cap4)

        def tile_of(r, c, step=step):
            if r == 0:
                return AWAKE
            w = wake[r][c]
            return FLASH if step == w else (AWAKE if step > w else ASLEEP)

        state(im, d, K, tile_of)
        frames.append(im)
        delays.append(90)
    delays[-1] = 2200

    # The palette must be built from frames that between them carry every colour used: the last
    # frame is all purple, so sampling it alone maps the stem, leaf and gold flash onto the
    # nearest plum.
    swatch = Image.new("RGB", (W, H * 3), ground)
    swatch.paste(frames[2], (0, 0))                       # act 1: the grape, greens, the bracket
    swatch.paste(frames[len(frames) // 3], (0, H))        # act 1 late: lit, out, the gold flash
    swatch.paste(frames[-1], (0, 2 * H))                  # act 3: the die and the batch
    pal = swatch.quantize(colors=64, method=Image.MEDIANCUT)
    frames = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
    out_path = ROOT / f"docs/assets/animation/gpap_propagate_{'dark' if dark else 'light'}.gif"
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=delays,
                   loop=0, optimize=True, disposal=1)
    print(f"{out_path.name}  {W}x{H}  {len(frames)} frames  "
          f"{sum(delays) / 1000:.1f}s  {out_path.stat().st_size // 1024} KB")


for dark in (True, False):
    build(dark)
