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
  topping out just under that run. The plant's one leaf sits ON TOP of that same lower run, just
  before the bunch hangs off it — on the vine, not beside it, which is what makes it read. From act 2 on, that same space holds THE CLAUSE STATES on a die instead, which is what
  the bunch turns into; the label above it says which it currently is.

  1. THE FORMULA. A real CNF and its real BCP trace, and the vine moves WITH it: a DECISION
     lights a berry gold and grows a gold leaf in the same frame. Its negation KNOCKS OUT that
     literal wherever it occurs (the berry shrinks and darkens, `sat_count -= 1`). A clause down
     to one survivor FORCES it — white flash, green leaf. It halts at a FIXPOINT with one clause
     still open and two variables never touched, which is what "partial assignment" means.
  2. THE FOLD. Under one assignment a clause's whole state is ONE PACKED WORD, so each clause
     collapses onto one berry, the five stack into an even column, and that column unfolds into
     the row beside the vine — symmetric about the middle word, which barely moves.
  3. THE BATCH. A row is a vine and its clause states. Rows multiply DOWNWARD from the one just
     watched — which never moves — and all propagate in the same launch, each to its own fixpoint,
     so no vine ever fills and no row ever lights up completely.

THE VINE IS LABELLED; THE BUNCH IS NOT. A variable's name sits above its node, and one line under
the vine says what its three kinds of node mean (decided, forced, untouched), for as long as the
vine is the only one on screen. The berries keep their faces: the caption names the KIND of event
— a decision, a forced literal, the fixpoint — and the bunch shows it happen. Two versions were
tried and thrown away: one that named every berry and counted every clause, which buried the
picture under text, and one that named a berry only while it acted, which still put text on the
bunch, where the eye is.

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
# Act 1's formula: a real CNF in DIMACS literals, one row of the bunch per clause, one berry per
# literal. The bunch is widest just under the stem, down to a binary clause at the tip. This is a
# PICTURE of the idea, not a diagram of the state layout — it trades exactness for something a
# reader takes in at a glance, so do not "correct" it into the engine's real shape.
FORMULA = [
    [1, -2, 3, 5],
    [-1, -2, -3, -4, 5],
    [-1, 2, -5, 6],
    [-1, 3, 4],
    [-4, -6],
]
DECISIONS = [1, -3]             # x1 := 1, then x3 := 0: one decision of each polarity
CLAUSES = [len(c) for c in FORMULA]
M = len(CLAUSES)                # M clauses = M cells of a row
K = 5                           # partial assignments = rows
NV = 6                          # variables = nodes on a vine: x_v is node v-1
assert CLAUSES == [4, 5, 4, 3, 2] and max(abs(lit) for c in FORMULA for lit in c) == NV
ROW0 = 0                        # act 1's row is the TOP one, level with the bunch's first
                                # clause; the die then grows DOWNWARD past the bunch

U = 5                                       # 5x: one cell pixel is five on screen
SIZE, GAP, PIN = CELL * U, 2 * U, 3 * U
PITCH = SIZE + GAP                          # cell + the die's own one-pixel gap
GROW = 38                                   # the bunch's rows nest, so a shorter row pitch
OUT_SIZE = CELL * (U - 1)                   # a knocked-out literal, one step smaller


def propagate(formula, decisions):
    """Act 1's script, DERIVED rather than typed: unit propagation over `formula` from
    `decisions`, one event per thing the picture shows. ("decide" | "force", lit, k, j) sets a
    literal, lighting clause k's slot j (for a force, k is the clause that was down to one);
    ("out", lit, k, j) falsifies one; ("fixpoint",) is propagation running dry. The trail is FIFO
    and a literal is assigned when it is pushed, which is when the engine's bitmap `atomic_or`
    lands — so a force follows the knock-out that caused it on the very next event."""
    value, events = {}, []

    def where(lit):
        return [(k, c.index(lit)) for k, c in enumerate(formula) if lit in c]

    def assign(kind, lit, k=None):
        # Every literal the script sets occurs exactly once, so setting it lights one berry and
        # the berry and the leaf stay one event.
        assert len(where(lit)) == 1 and abs(lit) not in value
        (at, j), = where(lit)
        assert k in (None, at)
        k = at
        value[abs(lit)] = lit > 0
        events.append((kind, lit, k, j))

    for dec in decisions:
        assign("decide", dec)
        trail = [dec]
        while trail:
            lit = trail.pop(0)
            for k, j in where(-lit):
                events.append(("out", -lit, k, j))
                if any(value.get(abs(x)) == (x > 0) for x in formula[k]):
                    continue                              # already satisfied: nothing to force
                live = [x for x in formula[k] if abs(x) not in value]
                assert live, "the script must never conflict"
                if len(live) == 1:
                    assign("force", live[0], k)
                    trail.append(live[0])
        events.append(("fixpoint",))
    return value, events


# x1 := 1 knocks ~x1 out of three clauses and forces nothing: a FIXPOINT after one decision. x3 :=
# 0 then starts one implication CHAIN — it leaves (~x1 v x3 v x4) with x4 alone, x4 := 1 leaves
# (~x4 v ~x6) with ~x6 alone, x6 := 0 knocks x6 out of the third clause — because a chain is what
# propagation actually is. And it stops there: (~x1 v x2 v ~x5 v x6) still has x2 and ~x5 live
# with no reason to pick between them, and x2 and x5 were never touched. That is what PARTIAL
# means, and it is why no vine ever fills and no row ever lights up completely.
VALUE, TRACE = propagate(FORMULA, DECISIONS)

READ = 1500         # a new caption's still beat, before anything moves
# ...and NO CAPTION IS REPLACED BEFORE IT CAN BE READ: subtitle pace, 14 characters a second, with a
# floor. The reader is also watching the picture, so this is a minimum, not a target.
READ_CPS, READ_MIN = 14, 2500


def read_ms(caption):
    return max(READ_MIN, round(len(caption) * 1000 / READ_CPS))


def var(v):
    return f"x{v}"


MONO= "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONOB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

# The bunch's stem, in cell pixels relative to the top centre of the bunch: it hangs from the
# vine's lower run and carries no leaf of its own.
BUNCH_STEM = [(-1, -6, STEM_HI), (0, -6, STEM_HI), (-1, -5, STEM_HI), (0, -5, STEM_HI),
              (-1, -4, STEM_HI), (0, -4, STEM_HI), (-1, -3, STEM_LO), (0, -3, STEM_LO),
              (-1, -2, STEM_LO), (0, -2, STEM_LO), (-1, -1, STEM_LO), (0, -1, STEM_LO)]

# The mark's leaf, on the VINE and sitting ON TOP of it, which is what makes it read as a leaf
# rather than a fleck. It goes on the LOWER run — the one that carries on to the bunch — just
# before the bunch hangs off it. In cell pixels relative to (the bunch's stem, the lower run's
# top row), so its bottom row lands one pixel above the run and its right edge one before the
# stem. Hung off the END of that run it floated in the air; up on the upper run it belonged to
# the wrong half of the vine.
VINE_LEAF = [(-4, -3, LEAF_HI), (-3, -3, LEAF_HI),
             (-4, -2, LEAF_HI), (-3, -2, LEAF_HI), (-2, -2, LEAF_LO),
             (-3, -1, LEAF_HI), (-2, -1, LEAF_LO)]


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
# On the dark ground RAMP[6] all but disappears, so the dark theme's knocked-out berry is a step
# lighter: still the darkest berry on screen, no longer a hole in it.
OUT_DARK = (0x56, 0x32, 0x9C)
OUT_FLASH_D = berry(OUT_DARK, RAMP[5], RAMP[6], "closed")
OUT_D       = berry(OUT_DARK, RAMP[5], RAMP[6], "closed", OUT_SIZE)


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


def leaf(d, x, y, colours, u=U):
    """The blade with its top-left at (x, y): the body colour, shadowed along the bottom row and
    the right column."""
    body, edge = colours
    for r, blade in enumerate(LEAF_BLADE):
        for c, bit in enumerate(blade):
            if bit == "#":
                px, py = x + c * u, y + r * u
                shade = r == len(LEAF_BLADE) - 1 or c == len(blade) - 1
                d.rectangle([px, py, px + u - 1, py + u - 1], fill=edge if shade else body)


def row_trace(rng, steps):
    """Another row's propagation: a decision, implications behind it, clause states reacting a
    step later. Never all variables and never all clauses — a row at its fixpoint is partial.

    THE SPAN IS PER ROW. Two rows that set the same number of variables must not set them on the
    same frames, or the batch pulses in unison: a common spacing formula makes every row tick
    together, which is monotonous however random the contents are. A row's own span, plus knocks
    scattered on their own moments rather than pinned behind a variable, is what makes the batch
    bubble — several rows mid-flight at any instant, each on its own clock.
    """
    order = list(range(NV))
    rng.shuffle(order)
    chosen = order[:rng.randint(2, NV - 2)]
    span = steps * rng.uniform(0.55, 1.0)
    start = rng.randint(1, 3)                             # ...and its own moment to start
    vstep = {c: start + round(i * (span - start) / max(1, len(chosen) - 1))
             for i, c in enumerate(chosen)}
    vgold = {c: i == 0 for i, c in enumerate(chosen)}     # the decision; the rest are forced

    cls = list(range(M))
    rng.shuffle(cls)
    cstep, ckind = {}, {}
    for i, c in enumerate(chosen):                        # a clause that variable satisfies...
        if rng.random() < 0.55:
            k = cls[i]
            cstep[k] = min(steps, vstep[c] + rng.randint(1, 2))
            ckind[k] = "sat"
    for k in cls:                                         # ...and clauses merely knocked down,
        if k not in cstep and rng.random() < 0.40:        # on their own moments, filling the gaps
            cstep[k] = rng.randint(1, max(1, round(span)))
            ckind[k] = "knock"
    return vstep, vgold, cstep, ckind


def build(dark: bool):
    ground = (0x16, 0x10, 0x2B) if dark else CREAM
    ink    = CREAM if dark else DIE
    faint  = RAMP[3] if dark else RAMP[5]
    head   = ImageFont.truetype(MONOB, 16)
    sub    = ImageFont.truetype(MONO, 14)
    tag    = ImageFont.truetype(MONO, 13)
    out_tile, out_flash = (OUT_D, OUT_FLASH_D) if dark else (OUT, OUT_FLASH)

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

    def frame(caption, lead="", right="the formula", left="the assignment"):
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
        d.text((vx0 + vine_w / 2, cap_h + 2), left, font=tag, fill=faint, anchor="ma")
        d.text((gcx, cap_h + 2), right, font=tag, fill=faint, anchor="ma")
        legend(d)
        return im, d

    def bunch_stem(d):
        """What is drawn only while the bunch is on screen: the stem it hangs by, and the vine's
        one leaf, up on the vine's own line above the step down."""
        for dx, dy, colour in BUNCH_STEM:
            x, y = gcx + dx * U, gtop + dy * U
            d.rectangle([x, y, x + U - 1, y + U - 1], fill=colour)
        for dx, dy, colour in VINE_LEAF:
            x, y = (gcx - U) + dx * U, (y0 + SIZE // 2 - U) + dy * U
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

    def vine(d, y, tile_of, connect="pin", named=False):
        """One assignment: a green stem with a node per variable, which runs level over those
        nodes and then STEPS DOWN once the assignment ends —

            ----|____

        — and carries on at the lower level to whatever it feeds. ONE shape, two lengths: it stops
        at the die's pin for this row, or carries on to the bunch's own stem, which hangs from it.
        Because only the length differs, the swap at the fold is all but invisible; running level
        the whole way instead put the vine along the top of the bunch rather than into it.

        `named` puts each variable's name above its node. Act 1's vine carries them throughout,
        and in act 3 it is the top row, so its names head the columns for every vine below."""
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
            if named:
                d.text((x + 3.5 * U, sy - 4), var(j + 1), font=tag, fill=faint, anchor="ms")
            if colours is None:
                continue                                   # a bare node: nothing has grown here
            leaf(d, x + U, y + 2 * U, colours)

    def legend(d):
        """One line in the bottom-left corner saying what a vine's nodes mean, at the key's own
        smaller scale. The corner is below the last row's leaves and left of the die, so the key
        is on every frame of all three acts."""
        u, x, y = 4, vx0, H - 8 * 4 - 10
        for mark, text in ((GOLD_LEAF, "decided"), (GREEN_LEAF, "forced"), (None, "untouched")):
            d.rectangle([x, y, x + 7 * u - 1, y + u - 1], fill=STEM_HI)
            d.rectangle([x, y + u, x + 7 * u - 1, y + 2 * u - 1], fill=STEM_LO)
            d.rectangle([x + 3 * u, y + 2 * u, x + 4 * u - 1, y + 4 * u - 1], fill=STEM_LO)
            if mark is not None:
                leaf(d, x + u, y + 4 * u, mark, u)
            d.text((x + 7 * u + 6, y + 4 * u), text, font=tag, fill=faint, anchor="lm")
            x += 7 * u + 6 + d.textlength(text, font=tag) + 12

    shown = [None, 0]                           # the caption on screen, and how long it has been

    def push(im, ms, caption):
        """Every frame goes through here, so no caption is replaced before `read_ms` says it can
        be read. A caption whose frames run short is held on its LAST frame, which is a settled
        one, so the wait lands on stillness rather than mid-motion."""
        if caption != shown[0]:
            if shown[0] is not None and shown[1] < read_ms(shown[0][0]):
                delays[-1] += read_ms(shown[0][0]) - shown[1]
            shown[:] = [caption, 0]
        frames.append(im)
        delays.append(ms)
        shown[1] += ms

    # --- act 1: the formula, and the vine that is causing it to change -------------------------
    lit, out, gold, lit_order = set(), set(), set(), []
    vine_state = {}                             # node -> decided? (else forced)

    def vine0(d, pop=None, connect="bunch"):
        vine(d, y0, lambda j: None if j not in vine_state else (
            (GOLD_POP if vine_state[j] else GREEN_POP) if j == pop else
            (GOLD_LEAF if vine_state[j] else GREEN_LEAF)), connect=connect, named=True)

    def state(i):
        return (DECIDED if i in gold else AWAKE) if i in lit else (out_tile if i in out else ASLEEP)

    def grape(im, d, flash=None, k=None, k_from=None, t=1.0):
        bunch_stem(d)
        if k is not None:
            bracket(d, k, k_from, t)
        for i, (x, y) in enumerate(gpos):
            tile = flash[1] if flash and flash[0] == i else state(i)
            blit(im, x, y, tile, halo=True)

    def act1(caption, lead, flash=None, k=None, k_from=None, t=1.0, pop=None):
        im, d = frame(caption, lead)
        vine0(d, pop)
        grape(im, d, flash=flash, k=k, k_from=k_from, t=t)
        return im

    # FOUR CAPTIONS, each naming a KIND of event rather than a variable: the picture shows which
    # one. The caption changes only when the kind does, so the second decision and the second
    # force reuse theirs, and the knock-outs and the fixpoint between the decisions get none.
    # "tier" and "berry", not "row" and "cell": from act 2 on a ROW is an assignment and a CELL is
    # a clause, and a word that meant something else a moment earlier would fight the fold.
    cap1 = ("each tier is a clause, each berry a literal", "CPU")
    says = {"decide": ("decide a variable at random: its negation turns false", "CPU"),
            "force": ("a clause left with one literal forces it", "CPU")}
    free = [var(v) for v in range(1, NV + 1) if v not in VALUE]
    cap_fix = (f"fixpoint: {' and '.join(free)} untouched, a partial assignment", "CPU")
    push(act1(*cap1), read_ms(cap1[0]), cap1)

    idx_gold, k_prev, cap = None, None, cap1
    for event in TRACE:
        if event[0] == "fixpoint":              # propagation ran dry: the next decision picks up
            continue                            # where the bracket already is
        kind, x, k, j = event
        i = first[k] + j
        if kind != "out":
            # The caption turns over the moment its event starts to move, with no still beat of
            # its own: a beat before each decision, with the bracket switched off for it, stalled
            # the trace and read as a hiccup. push() still holds every caption long enough to
            # read. A force is caused by the knock-out just before it, in the same clause.
            assert kind == "decide" or k_prev == k
            cap = says[kind]
        if k != k_prev:                          # the bracket eases over to the clause touched
            for f in range(1, 7):
                push(act1(*cap, k=k, k_from=k_prev, t=ease(f / 6)), 45, cap)
            k_prev = k
        # A decision or an implication SETS A VARIABLE, so the berry and the leaf are one event
        # and land in the same frame. What follows on later frames is what that set literal
        # falsifies elsewhere — the causal order is already in the trace.
        pop = None
        if kind == "out":
            tile, hold = out_flash, 130
        else:
            pop = abs(x) - 1
            vine_state[pop] = kind == "decide"
            tile, hold = (DEC_FLASH, 330) if kind == "decide" else (FLASH, 260)
        push(act1(*cap, flash=(i, tile), k=k, pop=pop), hold, cap)
        if kind == "decide" and idx_gold is None:
            idx_gold = len(frames) - 1
        if kind == "out":
            out.add(i)
        else:
            lit.add(i)
            lit_order.append(i)
            if kind == "decide":
                gold.add(i)
        push(act1(*cap, k=k), 290 if kind == "decide" else 200, cap)      # settle
    push(act1(*cap_fix), read_ms(cap_fix[0]), cap_fix)
    idx_fix = len(frames) - 1

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
    assert len(lit_order) == len(vine_state) < NV
    assert set(vine_state) == {v - 1 for v in VALUE}             # the vine shows what BCP set

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
            tile = state(i)
            if f == n_a:
                tile = FLASH if clause_of[i] in sat_clause else ASLEEP
            blit(im, x + (cx - x) * t, y + (cy - y) * t, tile, halo=True)
        push(im, READ if f == 0 else 600 if f == n_a else 40, cap2a)

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
        push(im, READ if f == 0 else 900 if f == n_b else 40, cap2b)

    # --- act 3: the batch — a row is a vine and its clause states ------------------------------
    # ONE caption for the whole act: the batch growing and the batch propagating are one claim,
    # and a second line arriving mid-act was one more thing to read while the picture moved.
    cap3 = ("thousands of rows, each with its own decisions, in one launch", "GPU")

    rng = random.Random(11)
    # A row here fires on BOTH objects — leaves on the vine and cells on the die — so it carries
    # roughly twice the events origin's die-only row did. A step is now two frames (pop, then
    # rest), which is 260 ms of air per step, so the count comes back down to origin's.
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

    def band(im, d, r, step, rest=False):
        """`rest` draws the same step with nothing flashing: whatever landed on this step is
        already settled. Act 1 gives every event a pop frame and then a settle frame, and that
        beat is most of why it reads; act 3 without it packs new pops and old settles into the
        same frame, so there is no rhythm for anything to pop out of."""
        vstep, vgold, cstep, ckind = traces[r]
        y = sy0 + r * PITCH

        def node(j):
            s = vstep.get(j)
            if s is None or step < s:
                return None
            if step == s and not rest:
                return GOLD_POP if vgold.get(j) else GREEN_POP
            return GOLD_LEAF if vgold.get(j) else GREEN_LEAF

        vine(d, y, node, named=r == ROW0)
        for c in range(M):
            s = cstep.get(c)
            landing = step == s and not rest
            if s is None or step < s:
                tile = ASLEEP
            elif ckind[c] == "sat":
                tile = FLASH if landing else AWAKE
            else:
                tile = out_flash if landing else out_tile
            blit(im, dx0 + c * PITCH, y, tile)

    def batch(caption, lead, r0, rows, step, rest=False):
        im, d = frame(caption, lead, right="the clause states", left="the assignments")
        die(d, r0, rows)
        for r in range(r0, r0 + rows):
            band(im, d, r, step, rest)
        return im

    # The die grows DOWNWARD from the row just watched, ONE ROW A FRAME, so the row the reader is
    # holding on to never moves and the empty frame below act 1 fills steadily. Growing it in a
    # few big chunks instead is the same build in three jumps, and it reads as stiff.
    for n in range(1, K + 1):                # n = 1 is the caption's beat, then a row a frame
        push(batch(*cap3, ROW0, n, 0), READ if n == 1 else 900 if n == K else 190, cap3)

    idx_mid = None
    for step in range(steps + 2):
        push(batch(*cap3, 0, K, step), 110, cap3)                 # what lands on this step...
        if step == steps // 2:
            idx_mid = len(frames) - 1
        last = step == steps + 1
        push(batch(*cap3, 0, K, step, rest=True), 2400 if last else 150, cap3)   # ...then calm

    # THE PALETTE IS PINNED, NOT SAMPLED. Median cut splits by pixel population, and the colours
    # that matter most here are the rarest: a leaf's bright green exists on the single frame it
    # grows on, so any swatch-based palette quantized it to the nearest plum. Every brand colour
    # gets an exact entry, and median cut is left only the job it is good at — the antialiased
    # greys inside the caption text.
    brand = list(dict.fromkeys(
        [ground, ink, faint, DIE, OUTLINE, CREAM, EYE, GOLD, GOLD_HI, GOLD_LO,
         STEM_HI, STEM_LO, LEAF_HI, LEAF_LO, LEAF_BRIGHT, (0xFF, 0xF6, 0xD8)] + RAMP
        + ([OUT_DARK] if dark else [])))
    samples = (0, idx_gold, idx_fix, idx_mid)
    text = Image.new("RGB", (W, H * len(samples)), ground)
    for n, idx in enumerate(samples):
        text.paste(frames[idx], (0, n * H))
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
