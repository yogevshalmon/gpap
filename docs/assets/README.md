# GPAP brand assets

> **Reader:** maintainers of the project's logo and images — what each asset is, where it is used, and how to regenerate it.

| folder | what it is | where it is used |
|---|---|---|
| `icon/` | the baseline mark, master SVG + PNGs (576 = 12× the 48px grid) | docs, badges, PyPI, anywhere inline |
| `icon_wake/` | the eyes variant, hero sizes only | the header lockup, social preview, slides |
| `wordmark/` | the generated header lockup: GPAP, then the mark, on a transparent ground | the README header |
| `animation/` | idle / wave / story loops from the kit, plus the generated `propagate` pair | README header and sign-off (`idle_wake`), the core-idea section (`propagate`). `wave` and `story` are unused today |
| `favicon/` | hand-drawn 16 / 24 / 32 px marks + `favicon.ico` | a docs site, PyPI project links |
| `avatar/` | 1024px tiles, inset so a circle crop keeps the pin ring | GitHub avatar, PyPI |
| `social/` | 1280×640 card | GitHub → Settings → Social preview (**upload manually**; a repo file is not picked up) |
| `demo/` | the terminal recording in the README | README only |
| `tools/` | the generators for `demo/`, `social/` and the `propagate` loops | run them, do not hand-edit their output |

## Rules that keep the mark intact

1. **Nearest-neighbour only.** Any smooth resample blurs the pixel grid and the mark dies. In CSS,
   `image-rendering: pixelated` on every `<img>` shown below native size.
2. **Integer multiples of 48 only**: 96, 144, 192, 240, 288, 384, 576. The 576px masters are an
   exact 12× render of a 48px grid, so go back through 48 rather than resampling 576 directly.
3. **Never recolour ad hoc.** die `#2C125A` · outline `#180A2E` · cells `#DECEFF → #BA9AFF →
   #8E5EFA → #643CBA` · pins `#E4C470` · stem `#627434` · leaf `#8AB092` · cream ground `#F5F0E8`.
4. **`icon/` is the default; `icon_wake/` is the exception.** The eyes stop being charming when
   they are everywhere.
5. **Every animation ends on the static icon frame.** Keep that when re-exporting; it is why the
   loops do not destabilise the identity.
6. **The favicons are drawn at their own pixel grid**, not scaled from the 48px master — at 16px
   there is no stem and no cell highlight, because both turn to noise. Re-export from the `.svg`
   next to each PNG; never rescale the PNG.

## Rendering surfaces

**README header** — `<picture>` with the dark/light `idle_wake` loop and the static
`icon_wake_256.png` as the `<img>` fallback for renderers that ignore `<picture>`. GitHub strips
`<video>` from Markdown, so a README animation is a GIF or nothing; GIFs cannot honour
`prefers-reduced-motion`, so a docs site should use `<video autoplay muted loop playsinline poster>`
instead.

**PyPI** drops `<picture>`, `<source>`, `<video>` and anything theme-conditional, and renders on
white. Point `<img src>` at an **absolute** raw-GitHub URL pinned to a tag, use the transparent
`icon/gpap_icon.png`, and prefer static over animated.

**Alt text is not optional.** `alt="GPAP"` gives a screen reader nothing, and the mark is the first
thing on the page.

## Regenerating the generated pieces

The social card, the terminal demo and the `propagate` loops are generated, not drawn:

```bash
python docs/assets/tools/make_wordmark.py       # the header lockup, dark + light
python docs/assets/tools/make_social_card.py    # 1280x640 card, from the 48px grid up
python docs/assets/tools/make_demo_gif.py       # the terminal recording
python docs/assets/tools/make_propagation_gif.py  # the core-idea story, dark + light
```

The demo's text is a verbatim capture of a real `gpap sample` run, pasted into the generator. If it
is re-recorded, the command shown must be the command that produced the output shown, flag for flag:
several of those lines change wording with the flags, so a command edited after the fact stops
matching its own output. Capture it from a scratch directory (so the 50,000-row `--out` file lands
there rather than in the checkout) with `GPAP_HOME=<that directory>/.gpap` — the `.gpap` suffix is
load-bearing, since the scratch CNF is `$GPAP_HOME/tmp/...` and the transcript has to show the
`.gpap/tmp/...` a default run from a checkout prints. The line ORDER is the run's own — formula, then what mining decided, then the result — so a
re-record must not reshuffle it into a tidier story than the run told.

`make_propagation_gif.py` tells the core idea in three acts. **The bunch is the point**: one row per
clause, one berry per literal, and it is the only place the literal-level truth is visible — a
literal falsified, a clause narrowing, a clause down to ONE survivor forcing it, which is the whole
of BCP. A version that reduced each clause to a single grape was tried and thrown away for exactly
that reason. Beside it, from the first frame and never moving, is **the assignment as a vine** on
the left: a node per variable, a gold leaf where it was decided at random, a green leaf where
propagation forced it, a bare node where nothing touched it. A decision lights a berry gold and
grows its leaf *in the same frame*, because setting a variable is one event; what follows on later
frames is what that literal falsifies elsewhere, which is the causal order already in the trace.
**The vine is labelled; the bunch is not.** Each variable's name sits above its node (in act 3 the
top row's names head the columns for every vine below), and one line in the bottom-left corner,
on every frame, says what the three kinds of node mean.
The berries keep their faces: each caption names a kind of event (a decision, a forced literal, the
fixpoint) and the bunch shows it happen. Two versions were tried and thrown away: one that named
every berry and counted every clause buried the picture under text, and one that named a berry only
while it acted still put text on the bunch, where the eye is.
Act 2 folds each clause row onto one cell — the engine's actual compression, one packed word per
clause. The five stack into a **centred column opened out to the die's own pitch** (the bunch's
rows nest tighter than a cell is tall, so without that they would overlap), and that column then
unfolds into the row beside the vine, symmetric about the middle word, which barely moves. Folding
each clause straight into its own column and lifting instead gives a diagonal staircase at the
bunch's pitch, and a stair is not a shape: the even column is what makes the unfold read. Act 3 grows the die downwards from that row, so the
row the reader is holding on to never moves — and it is then LEFT ALONE, because rows in a batch
reach their fixpoints at different times and giving it one more decision out of nowhere was motion
with no reason behind it. The vine's right end has two forms: while the bunch is on screen it runs
STRAIGHT ON into the bunch's own stem (one plant, joined under the bunch's leaf), and it bends down
into the die's pin only once the bunch has become that row. The two swap on the single frame the
bunch leaves and the die arrives, so it reads as the right-hand object changing identity rather
than as a glitch. The vine has ONE shape and two lengths: it runs
level over its own nodes, STEPS DOWN once the assignment ends (`----|____`), and carries on at the
lower level either to the die's pin for that row or on to the bunch, which HANGS from it by its own
six-pixel stem. The plant's one leaf sits on TOP of that lower run, just before the bunch hangs
off it. On the vine and on top of it is what makes it read: hung off the end of the run it floated,
and drawn on the bunch's own stem it was a fleck beside the berries. Because only the length differs, the swap at the fold
is all but invisible. Running level the whole way instead laid the vine along the top of the bunch,
and running it level with the first clause row drove a straight line into the side of the berries.

Two invariants are asserted at build time rather than trusted: no row ever ends with every clause
satisfied and no vine with every variable set, because propagation stops at a fixpoint and that is
what makes the assignment partial. No caption is replaced before it can be read (`read_ms`: subtitle
pace, with a floor). Every size drawn is a whole multiple of the 8px cell, and **the GIF palette is
pinned, not sampled**: median cut splits by pixel population, so the colours that matter most here
are the rarest (a leaf's bright green exists only on the frame it grows on) and every swatch-based
palette quantized them to plum.
