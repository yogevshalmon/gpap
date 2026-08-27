"""Compact sample-output writers.

A sample is a support projection: one bit per output column (for circuits, one primary-input wire
per cycle). The matrix is a numpy uint8 `(n, S)` array of 0/1 produced by
`GPUSampler.sample_bits` (GPU-side projection — only the bits cross to host RAM, so a large output
is cheap). Three formats:

  bits    — one 0/1 line per sample, `c`-comment header naming each column (default; compact + human
            readable + trivial to parse).
  npy     — `numpy.save` of the (n, S) uint8 array (zero-overhead round-trip: `numpy.load`).
  dimacs  — one line per sample as signed vars ending in 0 (CMSGen/UniGen convention; tool-compatible
            but the largest format).
"""
import numpy as np

#: Peak host bytes one `write_dimacs` block may materialise. Blocking by bytes rather than by a
#: fixed row count is what keeps the peak independent of the support width.
_BLOCK_BYTES = 32 << 20


def col_labels(cols_1based, wire_map=None):
    """Human labels for each output column: the wire name (`input@cycle`) when a `wire_map`
    {var: label} is available (from a .bench conversion), else the CNF variable number."""
    if wire_map:
        return [wire_map.get(v, f"v{v}") for v in cols_1based]
    return [f"v{v}" for v in cols_1based]


def write_bits(path, matrix, cols_1based, *, wire_map=None, meta=None):
    """Text 0/1: a `c`-comment header (meta + column labels) then one S-char 0/1 line per sample.

    The body is written as one bulk byte buffer (0/1 -> '0'/'1' via a uint8 lookup, plus newlines)
    — no per-cell Python formatting, so writing 100k rows is a single memcpy-speed write."""
    labels = col_labels(cols_1based, wire_map)
    n, S = matrix.shape
    header = [f"c gpap samples: {n} rows x {S} cols (support projection; one 0/1 per column)"]
    header += [f"c {k}: {v}" for k, v in (meta or {}).items()]
    for i in range(0, S, 20):                              # column legend, chunked
        header.append("c cols " + " ".join(labels[i:i + 20]))
    lut = np.array([ord("0"), ord("1")], dtype=np.uint8)
    body = np.concatenate([lut[matrix], np.full((n, 1), ord("\n"), dtype=np.uint8)], axis=1)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        f.write(body.tobytes())


def write_npy(path, matrix):
    """numpy .npy round-trip (`numpy.load(path)` -> (n, S) uint8)."""
    np.save(path, matrix)


def write_dimacs(path, matrix, cols_1based):
    """Signed-literal lines (CMSGen convention): `+v` where the column bit is 1, `-v` where 0, `0`
    terminated. `cols_1based` gives each column's variable number.

    Each column has only TWO possible tokens (`+v` and `-v`), so both are rendered ONCE and the
    rows become a gather over that table — no per-row Python formatting. The obvious per-row loop
    is many times slower than the `bits` writer, which made the one tool-compatible format the
    one nobody could afford to ask for.

    The fields are space-padded to a common width so the gather has a fixed stride. That costs
    file size, and the cost is set by the WIDEST column id, so a support mixing short and long
    variable numbers pays more than a uniform one (the padding is whitespace, which every DIMACS
    reader treats as a separator). Worth it here — the format is already several times the `bits`
    one, and the alternative was a writer too slow to use. Numbers: docs/findings/CLI_DELIVERY_FINDINGS.md."""
    cols = np.asarray(cols_1based, dtype=np.int64)
    n, S = matrix.shape
    # tok[0, j] = "-vj", tok[1, j] = "vj", space-padded to a common width so the render is a
    # fixed-stride byte gather.
    w = max((len(f"-{v}") for v in cols.tolist()), default=1) + 1
    blob = "".join(f"{sign}{v}".ljust(w) for sign in ("-", "") for v in cols.tolist())
    table = np.frombuffer(blob.encode(), dtype=np.uint8).reshape(2, S, w)
    tail = np.frombuffer(b"0\n", dtype=np.uint8)
    # Block by BYTES, not rows: a fixed row count makes the peak scale with S, so a wide-support
    # dump would allocate gigabytes where the row-at-a-time loop it replaced used O(S).
    rows = max(1, _BLOCK_BYTES // max(1, S * w + tail.size))
    cidx = np.arange(S)
    with open(path, "wb") as f:
        for lo in range(0, n, rows):
            # `!= 0` and not the raw values: this is an INDEX into a length-2 table now, where the
            # writer it replaced did `row.astype(bool)`. Any nonzero has always meant TRUE here,
            # and a public writer must not start raising IndexError on a caller's 0/255 mask.
            blk = matrix[lo:lo + rows] != 0
            body = table[blk.astype(np.uint8), cidx]              # (rows, S, w) uint8
            out = np.empty((body.shape[0], S * w + tail.size), dtype=np.uint8)
            out[:, :S * w] = body.reshape(body.shape[0], S * w)
            out[:, S * w:] = tail
            f.write(memoryview(out))


def write_samples(path, matrix, cols_1based, *, fmt="bits", wire_map=None, meta=None):
    """Dispatch to the requested format. `fmt`: 'bits' (default) | 'npy' | 'dimacs'."""
    if fmt == "npy":
        write_npy(path, matrix)
    elif fmt == "dimacs":
        write_dimacs(path, matrix, cols_1based)
    else:
        write_bits(path, matrix, cols_1based, wire_map=wire_map, meta=meta)
