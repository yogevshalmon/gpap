"""Render the README's core-idea animation: a formula propagated, then a whole batch of them.

Told in the brand mark's own pixel language — the berry is the kit's 8x8 cell with its L-shaped
highlight and shadowed bottom edge, the bunch is staggered the way the kit's story grape is, and
the stem and leaf greens come off the mark.

THE BUNCH IS THE POINT, and it is the only place the LITERAL-LEVEL truth is visible: one row per
clause, one berry per literal. Reduce a clause to a single berry and you can no longer see a
literal being falsified, a clause narrowing, or a clause down to ONE survivor forcing it — which
is the whole of BCP. Everything else in the picture exists to say what happens to that bunch.

TWO OBJECTS ON SCREEN AT A TIME, in two materials:

  THE ASSIGNMENT, on the LEFT, is a VINE — a green stem with a node per variable, there from the
  first frame and never moving. A bare node is a variable nothing has touched, a GOLD leaf one
  DECIDED at random, a GREEN leaf one propagation FORCED. Nothing purple ever hangs on it.
  THE FORMULA, on the RIGHT, is the BUNCH — purple berries, one row per clause, one berry per
  literal. It HANGS FROM THE VINE, off the lower of the vine's two levels, its own six-pixel stem
  topping out just under that run with its leaf poking out at the junction. From act 2 on, that same space holds THE CLAUSE STATES on a die instead, which is what
  the bunch turns into; the label above it says which it currently is.

  1. THE FORMULA. A real BCP trace, and the vine moves WITH it: a DECISION lights a berry gold and
     grows a gold leaf in the same frame. Its negation KNOCKS OUT that literal wherever it occurs
     (the berry shrinks and darkens, `sat_count -= 1`). A clause down to one survivor FORCES it —
     white flash, green leaf. It halts at a FIXPOINT with one clause still open and most of the
     formula never touched, which is what "partial assignment" means.
  2. THE FOLD. Under one assignment a clause's whole state is ONE PACKED WORD, so each clause row
     collapses onto one berry, the five stack into an even column, and that column unfolds into
     the row beside the vine — symmetric about the middle word, which barely moves.
  3. THE BATCH. A row is a vine and its clause states. Rows multiply DOWNWARD from the one just
     watched — which never moves — and all propagate in the same launch, each to its own fixpoint,
     so no vine ever fills and no row ever lights up completely.

One scale throughout, 5x the 8px cell (a knocked-out literal sits at 4x; kit rules 1 and 2).
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
# The bunch: widest just under the stem, down to a binary clause at the tip. This is a PICTURE of
# the idea, not a diagram of the state layout — it trades exactness for something a reader takes
# in at a glance, so do not "correct" it into the engine's real shape.
CLAUSES = [4, 5, 4, 3, 2]
M = len(CLAUSES)                # M clauses = M cells of a row
K = 5                           # partial assignments = rows
NV = 6                          # variables = nodes on a vine
ROW0 = 0                        # act 1's row is the TOP one, level with the bunch's first
                                # clause; the die then grows DOWNWARD past the bunch

U = 5                                       # 5x: one cell pixel is five on screen
SIZE, GAP, PIN = CELL * U, 2 * U, 3 * U
PITCH = SIZE + GAP                          # cell + the die's own one-pixel gap
GROW = 38                                   # the bunch's rows nest, so a shorter row pitch
OUT_SIZE = CELL * (U - 1)                   # a knocked-out literal, one step smaller

# Act 1's script — a real unit-propagation trace over clauses A..E = 0..4. Occurrence lists are
# what a decision touches: `out` is the negated literal falsified in another clause.
TRACE = [
    ("decide", 0, 0),     # A0 := T                          A satisfied
    ("out", 2, 3),        #   ~A0 occurs in C: C3 is out
    ("out", 3, 0),        #   ~A0 occurs in D: D0 is out     no clause is unit yet: FIXPOINT
    ("decide", 1, 3),     # B3 := T                          B satisfied
    ("out", 3, 2),        #   ~B3 occurs in D: D2 is out     D is down to one survivor...
    ("force", 3, 1),      #   ...so D1 := T                  D satisfied
    ("out", 4, 0),        #   ~D1 occurs in E: E0 is out     ...which leaves E with one...
    ("force", 4, 1),      #   ...so E1 := T                  E satisfied
    ("out", 2, 1),        #   ~E1 occurs in C: C1 is out     FIXPOINT — nothing else is forced
]
# ...and it stops there. Four clauses are satisfied, C still has C0 and C2 live with no reason to
# pick between them, and nine literals were never touched. That is what PARTIAL means, and it is
# why no vine ever fills and no row ever lights up completely. The middle of the trace is one
# implication CHAIN — B3 falsifies D2, which forces D1, which falsifies E0, which forces E1,
# which falsifies C1 — because a chain is what propagation actually is.

# Which node on the vine each set variable lands on, in the order act 1 sets them: two decided,
# two forced, and nodes left bare between them.
VINE_SLOTS = [0, 2, 3, 5]

READ = 1500         # a caption gets its own still beat, long enough to read before anything moves
READ_FIRST = 1900   # ...and the opening line, which defines the whole picture, a little longer

MONO  = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONOB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

# The kit's stem and leaf, in cell pixels, relative to the top centre of the bunch.
BUNCH_STEM = [(-1, -6, STEM_HI), (0, -6, STEM_HI), (-1, -5, STEM_HI), (0, -5, STEM_HI),
              (-1, -4, STEM_HI), (0, -4, STEM_HI), (-1, -3, STEM_LO), (0, -3, STEM_LO),
              (-1, -2, STEM_LO), (0, -2, STEM_LO), (-1, -1, STEM_LO), (0, -1, STEM_LO),
              (1, -7, LEAF_HI), (2, -7, LEAF_HI), (1, -6, LEAF_HI), (2, -6, LEAF_HI),
              (3, -6, LEAF_LO), (2, -5, LEAF_HI), (3, -5, LEAF_LO)]


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


ASLEEP    = berry(RAMP[5], RAMP[4], RAMP[6], "closed")            # unassigned / untouched
AWAKE     = berry(RAMP[2], RAMP[1], RAMP[3], "open")              # set true / clause satisfied
FLASH     = berry(RAMP[1], RAMP[0], RAMP[2], "open")              # the frame propagation lands
DECIDED   = berry(GOLD, GOLD_HI, GOLD_LO, "open")                 # chosen at random
DEC_FLASH = berry(GOLD_HI, (0xFF, 0xF6, 0xD8), GOLD, "open")      # the frame it is chosen
OUT_FLASH = berry(RAMP[6], RAMP[5], OUTLINE, "closed")            # the frame it is knocked out
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
    """Paste a berry centred in its slot. On bare ground the halo is the separation the die would
    give it, without which a pale berry washes out on the cream theme."""
    x, y = round(x) + (SIZE - tile.width) // 2, round(y) + (SIZE - tile.height) // 2
    if halo:
        u = tile.width // CELL
        h = HALOS[tile.width]
        canvas.paste(h, (x - u, y - u), h)
    canvas.paste(tile, (x, y), tile)


def ease(t):
    return t * t * (3 - 2 * t)


# A LEAF is DRAWN, not tiled: small, with a stalk at the top so the attachment end is never in
# doubt, and short enough to leave clear air above the next vine.
LEAF_BLADE = ["#####",
              "#####",
              ".###.",
              "..#.."]
LEAF_BRIGHT = (0xB4, 0xD2, 0xB8)
GREEN_LEAF, GREEN_POP = (LEAF_HI, LEAF_LO), (LEAF_BRIGHT, LEAF_HI)
GOLD_LEAF, GOLD_POP = (GOLD, GOLD_LO), (GOLD_HI, GOLD)


def row_trace(rng, steps):
    """Another row's propagation: a decision, implications behind it, clause states reacting a
    step later. Never all variables and never all clauses — a row at its fixpoint is partial."""
    order = list(range(NV))
    rng.shuffle(order)
    chosen = order[:rng.randint(2, NV - 2)]
    vstep = {c: 1 + round(i * (steps - 4) / max(1, len(chosen) - 1)) for i, c in enumerate(chosen)}
    vgold = {c: i == 0 for i, c in enumerate(chosen)}     # the decision; the rest are forced
    cls = list(range(M))
    rng.shuffle(cls)
    cstep, ckind, i = {}, {}, 0
    for c in chosen:
        for _ in range(rng.randint(0, 2)):
            if i < M - 1:                                # never every clause: a row stays partial
                k = cls[i]
                i += 1
                cstep[k] = vstep[c] + 1
                ckind[k] = "sat" if rng.random() < 0.6 else "knock"
    return vstep, vgold, cstep, ckind


def build(dark: bool):
    ground = (0x16, 0x10, 0x2B) if dark else CREAM
    ink    = CREAM if dark else DIE
    faint  = RAMP[3] if dark else RAMP[5]
    head   = ImageFont.truetype(MONOB, 16)
    sub    = ImageFont.truetype(MONO, 14)
    tag    = ImageFont.truetype(MONO, 13)

    # --- layout: the vine on the left; the bunch, and later the die, on the right --------------
    board_w, board_h = M * PITCH - GAP, K * PITCH - GAP
    vine_w = (NV - 1) * PITCH + SIZE
    frame_out = 2 * GAP + PIN                   # die edge + outline + pin
    pad, tail, cap_h, label_h = 20, 7 * U, 52, 26

    vx0 = pad                                   # the vines' leftmost node
    dx0 = vx0 + vine_w + tail + frame_out       # the die's first cell column
    W = dx0 + board_w + frame_out + pad
    stem_x1 = dx0 - 2 * GAP - PIN               # a vine ends at the die's pin for its row

    grape_h = (len(CLAUSES) - 1) * GROW + SIZE
    stem_h = 7 * U
    art_top = cap_h + label_h
    art_h = max(board_h + 2 * frame_out, grape_h + stem_h)
    H = art_top + art_h + 20
    sy0 = art_top + frame_out                   # the die's first cell row
    y0 = sy0 + ROW0 * PITCH                     # act 1's row: the vine sits here throughout

    # The bunch's widest clause is exactly as wide as the die, so it sits square over the columns.
    gcx = dx0 + board_w / 2
    gtop = sy0 + SIZE // 2 + 7 * U              # the bunch hangs off the vine's LOWER run: its
    #                                             own 6-pixel stem tops out just under that run
    gpos, clause_of, first = [], [], []
    for k, width in enumerate(CLAUSES):
        first.append(len(gpos))
        row_w = width * PITCH - GAP
        for j in range(width):
            gpos.append((gcx - row_w / 2 + j * PITCH, gtop + k * GROW))
            clause_of.append(k)

    frames, delays = [], []

    def frame(caption, lead="", right="the formula"):
        im = Image.new("RGB", (W, H), ground)
        d = ImageDraw.Draw(im)
        if lead:
            lw = d.textlength(lead + "  ", font=head)
            cw = d.textlength(caption, font=sub)
            x0 = (W - lw - cw) / 2
            d.text((x0, 16), lead, font=head, fill=ink)
            d.text((x0 + lw, 18), caption, font=sub, fill=faint)
        else:
            d.text((W / 2, 18), caption, font=sub, fill=faint, anchor="ma")
        # The left object never changes what it is; the right one does, so its label says which.
        d.text((vx0 + vine_w / 2, cap_h + 2), "the assignment", font=tag, fill=faint, anchor="ma")
        d.text((gcx, cap_h + 2), right, font=tag, fill=faint, anchor="ma")
        return im, d

    def bunch_stem(d):
        for dx, dy, colour in BUNCH_STEM:
            x, y = gcx + dx * U, gtop + dy * U
            d.rectangle([x, y, x + U - 1, y + U - 1], fill=colour)

    def bracket(d, k, k_from=None, t=1.0):
        """Gold brackets either side of clause k — the clause propagation is visiting. Between
        clauses they SLIDE rather than crossfade: a blended gold has no home in the GIF palette
        and would snap to a dark ghost."""
        def geom(k):
            row_w = CLAUSES[k] * PITCH - GAP
            return gcx - row_w / 2, gtop + k * GROW, row_w
        x0, y0_, row_w = geom(k)
        if k_from is not None and t < 1:
            xa, ya, wa = geom(k_from)
            x0, y0_, row_w = xa + (x0 - xa) * t, ya + (y0_ - ya) * t, wa + (row_w - wa) * t
        xl, xr = x0 - 13, x0 + row_w + 8
        for x, tick in ((xl, xl + 12), (xr, xr - 7)):
            d.rectangle([x, y0_ - 4, x + 4, y0_ + SIZE + 3], fill=GOLD)
            d.rectangle([min(x, tick), y0_ - 4, max(x + 4, tick), y0_ - 1], fill=GOLD)
            d.rectangle([min(x, tick), y0_ + SIZE, max(x + 4, tick), y0_ + SIZE + 3], fill=GOLD)

    def die(d, r0, rows):
        """The chip behind the clause states: outline, die, and a pin per cell on every side —
        the mark itself. The left pins are where the vines land."""
        top, h = sy0 + r0 * PITCH, rows * PITCH - GAP
        d.rectangle([dx0 - 2 * GAP, top - 2 * GAP, dx0 + board_w + 2 * GAP - 1,
                     top + h + 2 * GAP - 1], fill=OUTLINE)
        d.rectangle([dx0 - GAP, top - GAP, dx0 + board_w + GAP - 1, top + h + GAP - 1], fill=DIE)
        for c in range(M):
            x = dx0 + c * PITCH + SIZE // 2 - U
            d.rectangle([x, top - 2 * GAP - PIN, x + 2 * U - 1, top - 2 * GAP - 1], fill=GOLD)
            d.rectangle([x, top + h + 2 * GAP, x + 2 * U - 1, top + h + 2 * GAP + PIN - 1],
                        fill=GOLD)
        for r in range(r0, r0 + rows):
            y = sy0 + r * PITCH + SIZE // 2 - U
            d.rectangle([stem_x1, y, dx0 - 2 * GAP - 1, y + 2 * U - 1], fill=GOLD)
            d.rectangle([dx0 + board_w + 2 * GAP, y, dx0 + board_w + 2 * GAP + PIN - 1,
                         y + 2 * U - 1], fill=GOLD)

    def vine(d, y, tile_of, connect="pin"):
        """One assignment: a green stem with a node per variable, which runs level over those
        nodes and then STEPS DOWN once the assignment ends —

            ----|____

        — and carries on at the lower level to whatever it feeds. ONE shape, two lengths: it stops
        at the die's pin for this row, or carries on to the bunch's own stem, which hangs from it.
        Because only the length differs, the swap at the fold is all but invisible; running level
        the whole way instead put the vine along the top of the bunch rather than into it."""
        sy, py = y - 2 * U, y + SIZE // 2 - U
        bend = stem_x1 - 3 * U                       # the step down, just past the last node
        x_end = stem_x1 if connect == "pin" else gcx + U - 1
        d.rectangle([vx0 + 2 * U, sy, bend + 2 * U - 1, sy + U - 1], fill=STEM_HI)
        d.rectangle([vx0 + 2 * U, sy + U, bend + 2 * U - 1, sy + 2 * U - 1], fill=STEM_LO)
        d.rectangle([bend, sy, bend + U - 1, py + 2 * U - 1], fill=STEM_HI)
        d.rectangle([bend + U, sy, bend + 2 * U - 1, py + 2 * U - 1], fill=STEM_LO)
        d.rectangle([bend, py, x_end, py + U - 1], fill=STEM_HI)
        d.rectangle([bend, py + U, x_end, py + 2 * U - 1], fill=STEM_LO)
        for j in range(NV):
            x, colours = vx0 + j * PITCH, tile_of(j)
            d.rectangle([x + 3 * U, y, x + 4 * U - 1, y + 2 * U - 1], fill=STEM_LO)   # the stalk
            if colours is None:
                continue                                   # a bare node: nothing has grown here
            body, edge = colours
            for r, blade in enumerate(LEAF_BLADE):
                for c, bit in enumerate(blade):
                    if bit == "#":
                        px, py2 = x + (c + 1) * U, y + (r + 2) * U
                        dark_px = r == len(LEAF_BLADE) - 1 or c == len(blade) - 1
                        d.rectangle([px, py2, px + U - 1, py2 + U - 1],
                                    fill=edge if dark_px else body)

    # --- act 1: the formula, and the vine that is causing it to change -------------------------
    lit, out, gold, lit_order = set(), set(), set(), []
    vine_state = {}

    def vine0(d, pop=None, connect="bunch"):
        vine(d, y0, lambda j: None if j not in vine_state else (
            (GOLD_POP if vine_state[j] else GREEN_POP) if j == pop else
            (GOLD_LEAF if vine_state[j] else GREEN_LEAF)), connect=connect)

    def grape(im, d, flash=None, k=None, k_from=None, t=1.0):
        bunch_stem(d)
        if k is not None:
            bracket(d, k, k_from, t)
        for i, (x, y) in enumerate(gpos):
            if flash and flash[0] == i:
                tile = flash[1]
            else:
                tile = (DECIDED if i in gold else AWAKE) if i in lit else (
                    OUT if i in out else ASLEEP)
            blit(im, x, y, tile, halo=True)

    def act1(caption, lead, flash=None, k=None, k_from=None, t=1.0, pop=None):
        im, d = frame(caption, lead)
        vine0(d, pop)
        grape(im, d, flash=flash, k=k, k_from=k_from, t=t)
        return im

    cap1a = ("each row of the bunch is a clause, each cell a literal", "CPU")
    cap1 = ("a variable is decided, then the implications it forces", "CPU")
    for cap, hold in ((cap1a, READ_FIRST), (cap1, READ)):   # name the parts, then show them move
        frames.append(act1(*cap))
        delays.append(hold)

    idx_gold, k_prev = None, None
    for kind, k, j in TRACE:
        i = first[k] + j
        if k != k_prev:                          # the bracket eases over to the clause touched
            for f in range(1, 7):
                frames.append(act1(*cap1, k=k, k_from=k_prev, t=ease(f / 6)))
                delays.append(45)
            k_prev = k
        # A decision or an implication SETS A VARIABLE, so the berry and the leaf are one event
        # and land in the same frame. What follows on later frames is what that set literal
        # falsifies elsewhere — the causal order is already in the trace.
        pop = None
        if kind == "out":
            tile, hold = OUT_FLASH, 130
        else:
            pop = VINE_SLOTS[len(vine_state)]
            vine_state[pop] = kind == "decide"
            tile, hold = (DEC_FLASH, 330) if kind == "decide" else (FLASH, 260)
        frames.append(act1(*cap1, flash=(i, tile), k=k, pop=pop))
        if kind == "decide" and idx_gold is None:
            idx_gold = len(frames) - 1
        delays.append(hold)
        if kind == "out":
            out.add(i)
        else:
            lit.add(i)
            lit_order.append(i)
            if kind == "decide":
                gold.add(i)
        frames.append(act1(*cap1, k=k))          # settle
        delays.append(290 if kind == "decide" else 200)
    delays[-1] = 1500

    # The fixpoint: a satisfied clause has exactly one lit literal, the one open clause has none
    # and at least two literals still live, and the vine still has bare nodes. Nothing here may
    # quietly end up total.
    sat_clause = {clause_of[i] for i in lit}
    for k in range(M):
        cells = range(first[k], first[k] + CLAUSES[k])
        assert sum(1 for i in cells if i in lit) == (1 if k in sat_clause else 0)
        if k not in sat_clause:
            assert sum(1 for i in cells if i not in lit and i not in out) >= 2
    assert len(sat_clause) == M - 1
    assert len(lit_order) == len(vine_state) == len(VINE_SLOTS) < NV

    def word(k):
        """The packed word clause k folds into: satisfied, or still open."""
        return AWAKE if k in sat_clause else ASLEEP

    # --- act 2: each clause folds into one word; the words are that assignment's row -----------
    # Two motions, not one: first every clause's berries slide together IN PLACE and merge onto
    # the literal that satisfies it (the clause's packed word), then the five words travel to the
    # row beside the vine. Doing it in one diagonal sweep smeared eighteen overlapping sprites
    # into a streak.
    # These two lines ARE GPAP, so they carry its name the way act 1 carries CPU and act 3 GPU:
    # the three leads read CPU -> GPAP -> GPU, which is the whole argument in three words.
    cap2a = ("the idea: under one assignment, a clause is one cell of state", "GPAP")
    cap2b = ("so one partial assignment is one row of clause states", "GPAP")
    # The bunch's rows nest closer than a cell is tall, so the merged column OPENS OUT to the
    # die's own pitch as it forms, or the five words would overlap. That even column is what
    # makes the unfold read: it is one shape spreading into another, symmetric about the middle
    # word, which barely moves. Folding each clause straight into its own column instead gives a
    # diagonal staircase at the bunch's tighter pitch, and a stair is not a shape.
    col_top = gtop + (grape_h - (M * PITCH - GAP)) / 2
    centre = [(gcx - SIZE / 2, col_top + k * PITCH) for k in range(M)]
    draw_order = [i for i in range(len(gpos)) if i not in lit] + sorted(lit)   # lit on top

    n_a = 14
    for f in range(0, n_a + 1):           # f = 0 is the caption's beat: new text, no motion yet
        t = ease(f / n_a)
        im, d = frame(*cap2a)
        vine0(d)
        bunch_stem(d)
        for i in draw_order:
            x, y = gpos[i]
            cx, cy = centre[clause_of[i]]
            tile = (DECIDED if i in gold else AWAKE) if i in lit else (
                OUT if i in out else ASLEEP)
            if f == n_a:
                tile = FLASH if clause_of[i] in sat_clause else ASLEEP
            blit(im, x + (cx - x) * t, y + (cy - y) * t, tile, halo=True)
        frames.append(im)
        delays.append(READ if f == 0 else 40)
    delays[-1] = 600

    n_b = 18
    for f in range(0, n_b + 1):
        t = ease(f / n_b)
        im, d = frame(*cap2b, right="the clause states")
        if f == n_b:             # the chip arrives WITH the words, not ahead of them: drawing it
            die(d, ROW0, 1)      # up front puts a row of pins on screen long before it means
        vine0(d, connect="pin")  # anything, and the eye reads that as the animation stalling
        for k, (cx, cy) in enumerate(centre):
            tx, ty = dx0 + k * PITCH, y0
            blit(im, cx + (tx - cx) * t, cy + (ty - cy) * t, word(k), halo=f < n_b)
        frames.append(im)
        delays.append(READ if f == 0 else 40)
    delays[-1] = 900

    # --- act 3: the batch — a row is a vine and its clause states ------------------------------
    cap3 = ("thousands of rows: the same formula, different decisions", "GPU")
    cap4 = ("every row propagating in the same launch", "GPU")

    rng = random.Random(11)
    steps = 12
    traces = [row_trace(rng, steps) for _ in range(K)]
    # Row ROW0 is the assignment act 1 just built, at step -1 so it starts settled — and it is
    # then LEFT ALONE. Rows in a batch hit their fixpoints at different times, so the one we
    # watched simply holds while the rest catch up; giving it one more decision out of nowhere
    # was motion without a reason, and read as a glitch.
    traces[ROW0] = ({n: -1 for n in vine_state}, dict(vine_state),
                    {k: -1 for k in sat_clause}, {k: "sat" for k in sat_clause})

    # No row may end with every clause satisfied and no vine with every variable set: a batch of
    # PARTIAL assignments is the whole point, and a full row would quietly contradict it.
    for vstep, _, cstep, ckind in traces:
        assert sum(1 for k in cstep if ckind[k] == "sat") < M
        assert len(vstep) < NV

    def band(im, d, r, step):
        vstep, vgold, cstep, ckind = traces[r]
        y = sy0 + r * PITCH

        def node(j):
            s = vstep.get(j)
            if s is None or step < s:
                return None
            if step == s:
                return GOLD_POP if vgold.get(j) else GREEN_POP
            return GOLD_LEAF if vgold.get(j) else GREEN_LEAF

        vine(d, y, node)
        for c in range(M):
            s = cstep.get(c)
            if s is None or step < s:
                tile = ASLEEP
            elif ckind[c] == "sat":
                tile = FLASH if step == s else AWAKE
            else:
                tile = OUT_FLASH if step == s else OUT
            blit(im, dx0 + c * PITCH, y, tile)

    def batch(caption, lead, r0, rows, step):
        im, d = frame(caption, lead, right="the clause states")
        die(d, r0, rows)
        for r in range(r0, r0 + rows):
            band(im, d, r, step)
        return im

    # The die grows DOWNWARD from the row just watched, ONE ROW A FRAME, so the row the reader is
    # holding on to never moves and the empty frame below act 1 fills steadily. Growing it in a
    # few big chunks instead is the same build in three jumps, and it reads as stiff.
    for n in range(1, K + 1):                # n = 1 is the caption's beat, then a row a frame
        frames.append(batch(*cap3, ROW0, n, 0))
        delays.append(READ if n == 1 else 190)
    delays[-1] = 900

    idx_mid = None
    for step in range(steps + 2):
        frames.append(batch(*cap4, 0, K, step))
        if step == steps // 2:
            idx_mid = len(frames) - 1
        delays.append(READ if step == 0 else 150)
    delays[-1] = 2400

    # THE PALETTE IS PINNED, NOT SAMPLED. Median cut splits by pixel population, and the colours
    # that matter most here are the rarest: a leaf's bright green exists on the single frame it
    # grows on, so any swatch-based palette quantized it to the nearest plum. Every brand colour
    # gets an exact entry, and median cut is left only the job it is good at — the antialiased
    # greys inside the caption text.
    brand = list(dict.fromkeys(
        [ground, ink, faint, DIE, OUTLINE, CREAM, EYE, GOLD, GOLD_HI, GOLD_LO,
         STEM_HI, STEM_LO, LEAF_HI, LEAF_LO, LEAF_BRIGHT, (0xFF, 0xF6, 0xD8)] + RAMP))
    text = Image.new("RGB", (W, H * 3), ground)
    text.paste(frames[0], (0, 0))
    text.paste(frames[idx_gold], (0, H))
    text.paste(frames[idx_mid], (0, 2 * H))
    spare = 64 - len(brand)
    entries = [c for rgb in brand for c in rgb]
    entries += text.quantize(colors=spare, method=Image.MEDIANCUT).getpalette()[:3 * spare]
    pal = Image.new("P", (1, 1))
    pal.putpalette(entries + [0] * (768 - len(entries)))
    frames = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
    out_path = ROOT / f"docs/assets/animation/gpap_propagate_{'dark' if dark else 'light'}.gif"
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=delays,
                   loop=0, optimize=True, disposal=1)
    print(f"{out_path.name}  {W}x{H}  {len(frames)} frames  "
          f"{sum(delays) / 1000:.1f}s  {out_path.stat().st_size // 1024} KB")


for dark in (True, False):
    build(dark)
