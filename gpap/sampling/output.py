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
    terminated. `cols_1based` gives each column's variable number."""
    cols = np.asarray(cols_1based, dtype=np.int64)
    with open(path, "wb") as f:
        for row in matrix:
            lits = np.where(row.astype(bool), cols, -cols)
            f.write((" ".join(map(str, lits.tolist())) + " 0\n").encode())


def write_samples(path, matrix, cols_1based, *, fmt="bits", wire_map=None, meta=None):
    """Dispatch to the requested format. `fmt`: 'bits' (default) | 'npy' | 'dimacs'."""
    if fmt == "npy":
        write_npy(path, matrix)
    elif fmt == "dimacs":
        write_dimacs(path, matrix, cols_1based)
    else:
        write_bits(path, matrix, cols_1based, wire_map=wire_map, meta=meta)
