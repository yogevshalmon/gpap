# GPAP brand assets

| folder | what it is | where it is used |
|---|---|---|
| `icon/` | the baseline mark, master SVG + PNGs (576 = 12× the 48px grid) | docs, badges, PyPI, anywhere inline |
| `icon_wake/` | the eyes variant, hero sizes only | README header, social preview, slides |
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
python docs/assets/tools/make_social_card.py    # 1280x640 card, from the 48px grid up
python docs/assets/tools/make_demo_gif.py       # the terminal recording
python docs/assets/tools/make_propagation_gif.py  # the core-idea story, dark + light
```

The demo's text is a verbatim capture of a real `gpap sample` run, pasted into the generator. If it
is re-recorded, the command shown must be the command that produced the output shown, flag for flag:
several of those lines change wording with the flags, so a command edited after the fact stops
matching its own output.

`make_propagation_gif.py` tells the core idea in three acts — a formula drawn as the mark's grape
with one clause per row, propagated a clause at a time; each clause collapsing to one packed word
and the words lining up as that assignment's row; then the die, one row per assignment and one
column per clause, every row propagating at once. It reads the cell sprite off
`icon/gpap_icon.png` rather than redrawing it, so the animation stays in the mark's pixel language
if the mark is ever re-exported, and every size it draws is an integer multiple of the 8px cell.
