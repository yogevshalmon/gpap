# The selection API — how you get answers out of the engine

*(Companion to [README.md](README.md). If you are using `gpap::Sampler`, you never need this — it
does all of the below for you. Read this if you are driving `gpap::Engine` yourself, or if you
want to know why the readout looks the way it does.)*

---

## 1. The problem this solves

A chunk just finished. You have **K rows** — often 100,000+ — and each one walked its own random
assignment attempt. You now need to answer:

> **Which rows finished successfully, and what values did they land on?**

The obvious implementation is to copy a K-long yes/no array to the host, loop over it, then copy
the values back. That is a PCIe round-trip **per chunk**, and at ~100 chunks/second it becomes the
bottleneck — the GPU ends up waiting on the host to tell it what it already knows.

The selection API exists so the host never sees per-row data. Everything is computed on the
device; what crosses the bus is a **count**, and later only the **bits you asked for**.

---

## 2. A worked example

Take **N = 5 variables**, **K = 6 rows**, and an independent support of **{x1, x2}** — those are
the variables actually being sampled; x3–x5 are internal machinery.

After a chunk the rows look like this:

| row | x1 | x2 | x3 | x4 | x5 | status |
|-----|----|----|----|----|----|--------|
| 0 | T | F | T | F | T | active |
| 1 | — | — | — | — | — | **conflicted** |
| 2 | T | T | F | *free* | T | active |
| 3 | F | T | T | T | F | active |
| 4 | — | — | — | — | — | **conflicted** |
| 5 | T | F | T | T | *free* | active |

**"Which rows are done?" has two different correct answers**, depending on what you are asking:

- **Fully assigned** — a complete model of the whole formula: rows **{0, 3}**
- **Support complete** — every *sampled* variable decided: rows **{0, 2, 3, 5}**

The sampler wants both at once: the first is what it emits, the second tells it how often the
support got fixed without conflict. Remember this — it is why *slots* exist.

---

## 3. The assignment bitmap (the layout everything below reads)

Each variable `v` owns **two adjacent bits**:

- bit `2v`   = "v is TRUE"
- bit `2v+1` = "v is FALSE"
- both 0     = **v is still free**

Those bits live in a `(K, ceil(2N/32))` array of `int32`. For N = 5 that is 10 bits in one word:

```
bit:    0    1    2    3    4    5    6    7    8    9  | 10..31
       +x1  -x1  +x2  -x2  +x3  -x3  +x4  -x4  +x5  -x5 | padding
row 2:  1    0    1    0    0    1    0    0    1    0  |  0 0 0 ...
       └ x1=T ┘  └ x2=T ┘  └ x3=F ┘  └x4 FREE┘ └ x5=T ┘
```

Because `2v` and `2v+1` are adjacent and 32 is even, **a variable's two bits are always in the
same word** — one load answers free / assigned / contradicted.

---

## 4. `pair_mask` — "which variables am I asking about?"

```cpp
std::int32_t mask_all = engine.pair_mask();          // empty = every variable
std::int32_t mask_sup = engine.pair_mask({0, 1});    // 0-BASED variable indices: x1, x2
```

A pair mask is a word marking the variables to consider, by setting the **even** bit of each:

```
mask_all  (x1..x5) : bits 0, 2, 4, 6, 8 set
mask_sup  (x1, x2) : bits 0, 2          set
```

The completion test is then three bit operations over the row's words:

```
w | (w >> 1)   ->  even bit 2v is 1 iff variable v is ASSIGNED (either polarity)
~( ... )       ->  even bit 2v is 1 iff variable v is FREE
     & mask    ->  keep only the variables I asked about
     != 0      ->  "is any of them still free?"
```

Run it on row 2 from the table:

| | x1 | x2 | x3 | x4 | x5 | verdict |
|---|----|----|----|----|----|---------|
| assigned? `w \| w>>1` | 1 | 1 | 1 | **0** | 1 | |
| free? `~(...)`        | 0 | 0 | 0 | **1** | 0 | |
| `& mask_sup`          | 0 | 0 | – | – | – | `0` → **support complete** |
| `& mask_all`          | 0 | 0 | 0 | **1** | 0 | `≠0` → **not a full model** |

Exactly the two answers from §2, in one pass over the row — no per-variable loop.

> ### Why a mask is required, not optional
> Bits 10–31 are **padding** — variables that do not exist. They are always 0, so
> `~(w | w>>1)` marks every one of them as "free". Without a mask excluding them, **no row would
> ever be reported complete.** The mask is what makes the test *correct*, not merely configurable.

> ### ⚠️ Lifetime
> A mask lives until the engine is destroyed and **cannot be freed**. Create one per *variable
> set* — you will have one or two — and keep the id. Creating one inside the chunk loop leaks
> `ceil(2N/32)` int32 of device memory every chunk.

---

## 5. `select_complete` — run the test, leave the answer on the GPU

```cpp
std::int32_t n = engine.select_complete(mask_all, /*slot=*/0);   // -> 2
```

One kernel applies the §4 test to **all K rows in parallel** and writes a K-long yes/no array into
**slot 0**, on the device. A second tiny kernel sums it.

**The only thing returned to the host is the number `2`.** The K-long array never crosses the bus.

---

## 6. Slots — numbered shelves on the GPU

A **slot holds one row-selection**. There are `GPAP_SLOTS` (= 3) of them. That is the whole
concept: it lets you say "the rows on shelf 0" later, without ever having shipped that list to the
host and back.

```cpp
engine.select_complete(mask_all, /*slot=*/0);   // shelf 0 = fully assigned   -> {0, 3}
engine.select_complete(mask_sup, /*slot=*/1);   // shelf 1 = support complete -> {0, 2, 3, 5}
```

- **Slots 0 and 1 are yours.** Two exist because of the two-answers situation in §2 — the sampler
  needs both alive simultaneously.
- **Slot 2 is scratch**, for one-off reads.

---

## 7. `selection_rows` — "now give me the actual row numbers"

```cpp
std::vector<std::int32_t> rows = engine.selection_rows(/*slot=*/0, n);   // -> {0, 3}
```

Only now is anything compacted: the device mask is fetched once, turned into the ascending list
`{0, 3}`, and cached on the shelf. Most callers never need this — they want the count, then the
values.

> ### ⚠️ The capacity argument is a real bound, not a formality
> It writes `min(actual selection, capacity)`. Before that bound existed, the output size was only
> *implied* by a count you had been handed earlier — so a caller whose count was one chunk stale
> wrote past the end of its own buffer. Pass the capacity the buffer really has.

---

## 8. `selection_set` — the reverse direction

```cpp
engine.selection_set(/*slot=*/2, {0, 1, 2, 3, 4, 5});
```

"Here is a row list **I** chose; put it on shelf 2." This lets the readout serve *arbitrary* rows
rather than only successful ones. Two real uses:

- `AssumptionChecker` wants **all K rows**, conflicted ones included — no completion test involved.
- The sampler hits its keep-cap mid-chunk and wants only the **first 40 of 100** models found.

---

## 9. Reading the values

```cpp
engine.check_sound(/*slot=*/0);                     // -> 0
engine.var_bits(/*cols=*/{0, 1}, n, /*slot=*/0);    // -> x1,x2 of rows 0 and 3
engine.assignment_rows(n, /*slot=*/0);              // -> full (n x 2N) literal assignment
```

**`check_sound`** is the independent safety net. At a *fully assigned* row, "no clause has zero
surviving literals" is equivalent to "every clause has a true literal" — i.e. the row really
satisfies the formula. It runs on the device and returns one number, which must be `0`.

**`var_bits`** gathers only the columns you name, for only the rows on that shelf, straight out of
the bitmap. For this example that is `2 rows × 2 variables = 4 bytes`, rather than materialising
anything K-sized.

**`assignment_rows`** gives the full literal-level assignment when you genuinely need it. It is
sliced internally so the transient stays bounded regardless of N.

---

## 10. The whole sequence

```
                                                            bytes to host
engine.pair_mask({0,1})       -> mask_sup   (ONCE, ever)           0
engine.pair_mask()            -> mask_all   (ONCE, ever)           0
--- per chunk ------------------------------------------------------------
select_complete(mask_all, 0)  -> 2          shelf 0 = {0,3}        4
select_complete(mask_sup, 1)  -> 4          shelf 1 = {0,2,3,5}    4
check_sound(0)                -> 0                                 4
var_bits({0,1}, 2, slot=0)    -> the answers                       4
```

**Sixteen bytes of host traffic to fully harvest a chunk.**

That is the point of the whole design, and it explains each choice: `select_complete` returns a
count rather than a mask, selections live in numbered slots rather than being passed around, and
`var_bits` takes a column list rather than handing you the entire assignment.

---

## 11. Reference

| C++ ([gpap.hpp](include/gpap/gpap.hpp)) | C ABI ([gpap.h](include/gpap/gpap.h)) |
|---|---|
| `pair_mask(vars = {})` | `gpap_pair_mask_create` |
| `select_complete(mask_id, slot = 0)` | `gpap_select_complete` |
| `selection_rows(slot, count)` | `gpap_selection_rows` |
| `selection_set(slot, rows)` | `gpap_selection_set` |
| `check_sound(slot = 0, max_rows = 4096)` | `gpap_check_sound` |
| `var_bits(cols, count, slot = 0)` | `gpap_var_bits` |
| `assignment_rows(count, slot = 0)` | `gpap_assignment_rows` |
| `flags()` / `num_propagated()` | `gpap_flags` / `gpap_num_propagated` |

Variable and column indices on this API are **0-based** (tier 2 speaks the ABI's indices
throughout). `gpap::Sampler` converts to and from DIMACS numbering for you.

Implementation: [src/readout.cpp](src/readout.cpp); the kernels it launches are `complete_mask`,
`sound_rows`, `var_bits` and `unpack_rows` in [kernels/helpers.py](kernels/helpers.py).
