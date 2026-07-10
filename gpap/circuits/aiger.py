"""Reader for the AIGER circuit format (http://fmv.jku.at/aiger) — binary `aig` and ASCII `aag`,
combinational or sequential (latches, with AIGER 1.9 reset values).

Two consumers:
  * `gpap.circuits.aig2cnf` — NATIVE .aig/.aag input support: parse, simulate, unroll, Tseitin
    (no external tool needed).
  * `gpap.circuits.abc_backend` — reads the framed AIG that berkeley-abc's `write_aiger -s`
    produces (its variable numbering is fully specified, which is why ABC output is routed
    through AIGER rather than `write_cnf`, whose CNF-var assignment exposes no input map).

Literal encoding: variable `v` has literals `2v` (positive) and `2v+1` (negated); literal 0 is
constant FALSE, 1 constant TRUE. Variables `1..I` are inputs, `I+1..I+L` latches, then ANDs.
Latch reset values: 0 (default and the AIGER-1.0 meaning), 1, or `x` (uninitialized — the reset
field equals the latch's own literal). Uninitialized latches are REJECTED here: GPAP's circuit
semantics pin the T=0 state (reset-state-0 convention), and silently treating `x` as free would
enlarge the sampled solution space.
"""
from dataclasses import dataclass, field


@dataclass
class Aig:
    max_var: int                                   # M — highest variable index
    n_inputs: int                                  # I
    n_latches: int                                 # L
    n_outputs: int                                 # O
    inputs: list[int] = field(default_factory=list)            # input LITERALS, in order
    latches: list[tuple[int, int, int]] = field(default_factory=list)  # (cur, next, reset∈{0,1})
    outputs: list[int] = field(default_factory=list)           # output LITERALS, in order
    ands: list[tuple[int, int, int]] = field(default_factory=list)     # (lhs, rhs0, rhs1) literals
    input_names: dict[int, str] = field(default_factory=dict)  # input index  -> symbol name
    output_names: dict[int, str] = field(default_factory=dict)  # output index -> symbol name

    def and_defs(self) -> dict[int, tuple[int, int]]:
        """{lhs_var: (rhs0_lit, rhs1_lit)} — order-independent view (ASCII files may list AND
        definitions in any order; the binary format is topologically sorted by construction)."""
        return {lhs >> 1: (r0, r1) for lhs, r0, r1 in self.ands}


def _parse_symbols(lines) -> tuple[dict, dict]:
    """`i<idx> name` / `o<idx> name` symbol lines (latch symbols `l<idx>` are skipped),
    terminated by end-of-input or a comment section (`c` line)."""
    in_names, out_names = {}, {}
    for raw in lines:
        if not raw or raw == "c":
            break
        tag, _, name = raw.partition(" ")
        if tag[:1] == "i" and tag[1:].isdigit():
            in_names[int(tag[1:])] = name
        elif tag[:1] == "o" and tag[1:].isdigit():
            out_names[int(tag[1:])] = name
        elif tag[:1] == "l" and tag[1:].isdigit():
            pass
        elif tag[:1] not in ("i", "o", "l"):
            break
    return in_names, out_names


def _check_header(path: str, hdr: list) -> tuple[int, int, int, int, int]:
    if len(hdr) < 6:
        raise ValueError(f"{path}: short AIGER header {hdr!r}")
    m, i, l, o, a = (int(x) for x in hdr[1:6])  # noqa: E741
    if any(int(x) for x in hdr[6:]):
        raise ValueError(f"{path}: AIGER 1.9 B/C/J/F sections are not supported")
    return m, i, l, o, a


def _latch_reset(cur: int, fields: list[int], path: str) -> int:
    """Decode the optional reset field. Default (absent) = 0 per AIGER."""
    if not fields:
        return 0
    r = fields[0]
    if r in (0, 1):
        return r
    if r == cur:
        raise ValueError(f"{path}: latch {cur >> 1} is UNINITIALIZED (reset=x) — GPAP pins the "
                         "T=0 state; re-encode with a defined reset (0/1)")
    raise ValueError(f"{path}: bad latch reset {r} for latch literal {cur}")


def _read_aag(path: str, text: str) -> Aig:
    lines = text.splitlines()
    hdr = lines[0].split()
    M, I, L, O, A = _check_header(path, hdr)
    pos = 1
    inputs = [int(lines[pos + k].split()[0]) for k in range(I)]
    pos += I
    latches = []
    for k in range(L):
        f = [int(x) for x in lines[pos + k].split()]
        if len(f) < 2:
            raise ValueError(f"{path}: bad aag latch line {lines[pos + k]!r}")
        latches.append((f[0], f[1], _latch_reset(f[0], f[2:], path)))
    pos += L
    outputs = [int(lines[pos + k].split()[0]) for k in range(O)]
    pos += O
    ands = []
    for k in range(A):
        lhs, r0, r1 = (int(x) for x in lines[pos + k].split()[:3])
        ands.append((lhs, r0, r1))
    pos += A
    in_names, out_names = _parse_symbols(lines[pos:])
    return Aig(M, I, L, O, inputs, latches, outputs, ands, in_names, out_names)


def _read_aig(path: str, data: bytes) -> Aig:
    nl = data.index(b"\n")
    M, I, L, O, A = _check_header(path, data[:nl].split())
    pos = nl + 1
    inputs = [2 * (k + 1) for k in range(I)]       # binary format: implicit, consecutive
    latches = []
    for k in range(L):                             # latch lines: "next [reset]" (ASCII)
        nl = data.index(b"\n", pos)
        f = [int(x) for x in data[pos:nl].split()]
        cur = 2 * (I + 1 + k)
        latches.append((cur, f[0], _latch_reset(cur, f[1:], path)))
        pos = nl + 1
    outputs = []
    for _ in range(O):
        nl = data.index(b"\n", pos)
        outputs.append(int(data[pos:nl]))
        pos = nl + 1

    def _uint(p: int) -> tuple[int, int]:
        """LEB128-style variable-length unsigned int (the AIGER delta encoding)."""
        x, shift = 0, 0
        while True:
            ch = data[p]
            p += 1
            x |= (ch & 0x7F) << shift
            if not (ch & 0x80):
                return x, p
            shift += 7

    ands = []
    for k in range(A):
        lhs = 2 * (I + L + 1 + k)                  # AND lhs literals are consecutive and even
        d0, pos = _uint(pos)
        d1, pos = _uint(pos)
        rhs0 = lhs - d0
        rhs1 = rhs0 - d1
        ands.append((lhs, rhs0, rhs1))
    in_names, out_names = _parse_symbols(
        data[pos:].decode("utf-8", "replace").splitlines())
    return Aig(M, I, L, O, inputs, latches, outputs, ands, in_names, out_names)


def read_aiger(path: str) -> Aig:
    """Parse an AIGER file, binary (`aig`) or ASCII (`aag`), combinational or sequential."""
    with open(path, "rb") as f:
        data = f.read()
    magic = data[:4].split()[0] if data[:4].strip() else b""
    if data.startswith(b"aag"):
        return _read_aag(path, data.decode("utf-8", "replace"))
    if data.startswith(b"aig"):
        return _read_aig(path, data)
    raise ValueError(f"{path}: not an AIGER file (magic={magic!r})")


def split_frame_name(name: str) -> tuple[str, int] | None:
    """Split ABC `frames` signal names `<orig>_<frame>` -> (orig, frame). Returns None when the
    trailing field is not a frame number. Original names may themselves contain (or end with)
    underscores — e.g. `NL_REG_3__00` -> (`NL_REG_3_`, 0) — so split on the LAST underscore only."""
    base, sep, tail = name.rpartition("_")
    if not sep or not tail.isdigit():
        return None
    return base, int(tail)
