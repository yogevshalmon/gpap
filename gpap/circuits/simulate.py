"""Bit-parallel circuit simulation: the REJECTION-SAMPLING baseline + end-to-end sample checking.

For a circuit instance the trivial rival to any CNF sampler is simulate-and-filter: draw random
input sequences, run the circuit itself (64 patterns per machine word per gate, no CNF) and keep
the ones whose last-cycle outputs hit the pinned target. It is perfectly uniform over the
projected solution space and its rate is p_accept x simulation throughput, so the whole value of
BCP sampling on a circuit reduces to how far p_accept is below the sampler's completion rate
(amplification = success_rate / p_accept). `sim_baseline` measures that at the SAME
cycles/seed/pin/target the CNF conversion used; `rare_target` is the adversarial target knob (the
RAREST last-cycle pattern observed in a large batch — observed, so reachable, so SAT — against
bench.random_target's size-biased single draw); `simulate_check` is encoder-independent
soundness, running every EMITTED sample through the circuit to verify the pinned outputs, which
validates the whole unroll+Tseitin/ABC encoding and not just CNF clause satisfaction.

Everything here is numpy uint64 bit-parallel on the CPU, deliberately: it is the baseline an
adversarial reviewer gets for free, and it is already memory-bound.
"""
import time

import numpy as np

_W = 64
_FULL = np.uint64(0xFFFFFFFFFFFFFFFF)


def _popcount(arr: np.ndarray) -> int:
    if hasattr(np, "bitwise_count"):
        return int(np.bitwise_count(arr).sum())
    return int(np.unpackbits(arr.view(np.uint8)).sum())


def _topo_gates(circ):
    inputs, _outputs, dffs, gates = circ
    leaves = set(inputs) | set(dffs)
    order, done = [], set(leaves)
    for root in gates:
        stack = [root]
        while stack:
            sig = stack[-1]
            if sig in done:
                stack.pop()
                continue
            need = [i for i in gates[sig][1] if i not in done]
            if need:
                stack.extend(need)
                continue
            done.add(sig)
            order.append(sig)
            stack.pop()
    return order


def _eval_gate_words(op, vals):
    if op in ("AND", "NAND"):
        r = vals[0].copy()
        for v in vals[1:]:
            r &= v
        return r ^ _FULL if op == "NAND" else r
    if op in ("OR", "NOR"):
        r = vals[0].copy()
        for v in vals[1:]:
            r |= v
        return r ^ _FULL if op == "NOR" else r
    if op == "NOT":
        return vals[0] ^ _FULL
    if op in ("BUFF", "BUF"):
        return vals[0]
    if op in ("XOR", "XNOR"):
        r = vals[0].copy()
        for v in vals[1:]:
            r ^= v
        return r ^ _FULL if op == "XNOR" else r
    raise ValueError(f"unknown gate {op}")


def _run_bench(circ, order, T, n_words, rng=None, given=None):
    """Simulate n_words*64 patterns for T cycles (reset 0); inputs from `given` ({(input, t):
    uint64 array}) or fresh random words. Returns last-cycle {output: words}."""
    inputs, outputs, dffs, gates = circ
    zeros = np.zeros(n_words, dtype=np.uint64)
    state = {d: zeros for d in dffs}
    vals = {}
    for t in range(T):
        vals = dict(state)
        for inp in inputs:
            vals[inp] = (given[(inp, t)] if given is not None
                         else rng.integers(0, _FULL, n_words, dtype=np.uint64, endpoint=True))
        for sig in order:
            op, ins = gates[sig]
            vals[sig] = _eval_gate_words(op, [vals[i] for i in ins])
        state = {d: vals[din] for d, din in dffs.items()}
    return {o: vals[o] for o in outputs}


def _topo_ands(aig):
    and_def = aig.and_defs()
    done = {0} | {lit >> 1 for lit in aig.inputs} | {cur >> 1 for cur, _n, _r in aig.latches}
    order = []
    for lhs, _r0, _r1 in aig.ands:
        stack = [lhs >> 1]
        while stack:
            v = stack[-1]
            if v in done:
                stack.pop()
                continue
            need = [u >> 1 for u in and_def[v] if (u >> 1) not in done]
            if need:
                stack.extend(need)
                continue
            done.add(v)
            order.append(v)
            stack.pop()
    return order


def _run_aig(aig, order, T, n_words, rng=None, given=None):
    """AIG twin of _run_bench: `given` keys are (input index k, t); returns last-cycle words."""
    and_def = aig.and_defs()
    zeros = np.zeros(n_words, dtype=np.uint64)
    state = {cur >> 1: (np.full(n_words, _FULL) if r else zeros)
             for cur, _nxt, r in aig.latches}
    latch_next = [(cur >> 1, nxt) for cur, nxt, _r in aig.latches]

    def lit_val(vals, lit):
        w = vals[lit >> 1]
        return w ^ _FULL if (lit & 1) else w

    vals = {}
    for t in range(T):
        vals = {0: zeros}
        vals.update(state)
        for k, lit in enumerate(aig.inputs):
            vals[lit >> 1] = (given[(k, t)] if given is not None
                              else rng.integers(0, _FULL, n_words, dtype=np.uint64, endpoint=True))
        for v in order:
            r0, r1 = and_def[v]
            vals[v] = lit_val(vals, r0) & lit_val(vals, r1)
        state = {cur: lit_val(vals, nxt) for cur, nxt in latch_next}
    return [lit_val(vals, o) for o in aig.outputs]


def _load(path):
    if path.endswith(".bench"):
        from gpap.circuits.bench import parse_bench
        return "bench", parse_bench(path)
    if path.endswith((".aig", ".aag")):
        from gpap.circuits.aiger import read_aiger
        return "aig", read_aiger(path)
    raise ValueError(f"{path}: not a circuit input (.bench/.aig/.aag)")


def _pin_and_target(kind, circ, T, seed, pin_outputs, target_mode, target=None):
    """(pinned output keys, {key: bool} target over them, effective T), resolved exactly as the CNF
    converters do; bench keys are output names, aig keys are output indices."""
    if kind == "bench":
        outputs = circ[1]
        if target is None:
            from gpap.circuits.bench import random_target
            target = (rare_target(kind, circ, T, seed, pin_outputs)[0]
                      if target_mode == "rare" else random_target(circ, T, seed))
        pin = outputs if pin_outputs is None else outputs[:pin_outputs]
        return pin, {o: target[o] for o in pin}, T
    if circ.n_latches == 0:
        T = 1
    if target is None:
        from gpap.circuits.aig2cnf import random_target_aig
        target = (rare_target(kind, circ, T, seed, pin_outputs)[0]
                  if target_mode == "rare" else random_target_aig(circ, T, seed))
    n_out = circ.n_outputs
    pin = list(range(n_out) if pin_outputs is None else range(min(pin_outputs, n_out)))
    return pin, {j: target[j] for j in pin}, T


def _accept_words(last_out, pin_target):
    acc = None
    for key, want in pin_target.items():
        w = last_out[key] if want else (last_out[key] ^ _FULL)
        acc = w if acc is None else (acc & w)
    return acc


def rare_target(kind, circ, T, seed, pin_outputs=None, n_words=4096):
    """The adversarial target: simulate n_words*64 random sequences, count the observed last-cycle
    patterns RESTRICTED TO THE PINNED OUTPUTS, and return the full pattern of a representative of
    the RAREST one — observed, so reachable, so the pinned CNF stays SAT, but p_accept drops to
    ~<= 1/(n_words*64). Deterministic in `seed`; returns (target, est_freq), formatted as the
    converters expect ({output: bool} for .bench, a [bool] list for AIGER)."""
    if kind == "bench":
        outputs = circ[1]
        order = _topo_gates(circ)
        last = _run_bench(circ, order, T, n_words, rng=np.random.default_rng(seed))
        keys = outputs
    else:
        if circ.n_latches == 0:
            T = 1
        order = _topo_ands(circ)
        last = _run_aig(circ, order, T, n_words, rng=np.random.default_rng(seed))
        keys = list(range(len(last)))
        last = dict(enumerate(last))
    pin = keys if pin_outputs is None else keys[:pin_outputs]
    n = n_words * _W
    # bits[i, j] = pattern i's value of pinned output j (little-endian: bit b of word w is
    # pattern w*64+b)
    cols = [np.unpackbits(last[k].view(np.uint8), bitorder="little")[:n] for k in pin]
    mat = np.stack(cols, axis=1) if cols else np.zeros((n, 0), dtype=np.uint8)
    packed = np.packbits(mat, axis=1)
    uniq, inv, counts = np.unique(packed, axis=0, return_inverse=True, return_counts=True)
    rare_i = int(counts.argmin())
    rep = int(np.nonzero(inv == rare_i)[0][0])
    full = {k: bool(np.unpackbits(last[k].view(np.uint8), bitorder="little")[rep])
            for k in keys}
    est = counts[rare_i] / n
    if kind == "bench":
        return full, float(est)
    return [full[j] for j in keys], float(est)


def sim_baseline(path, cycles, seed, pin_outputs=None, target_mode="random", target=None,
                 n_min=1 << 20, budget_s=2.0, hard_cap_s=10.0, sim_seed=0):
    """Measure the rejection-sampling baseline at the SAME conversion parameters `gpap
    sample/bench` used, batching until both n_min patterns and budget_s seconds are reached, never
    past hard_cap_s. Returns p_accept (0.0 with `p_accept_lt`, a 95% upper bound, when no hit was
    seen), hits, n_sim, patterns_per_s, accepted_per_s, T, n_pinned, target_mode."""
    kind, circ = _load(path)
    pin, pin_target, T = _pin_and_target(kind, circ, cycles, seed, pin_outputs, target_mode,
                                         target)
    order = _topo_gates(circ) if kind == "bench" else _topo_ands(circ)
    rng = np.random.default_rng(sim_seed)
    n_words_batch = 2048
    hits = n = 0
    t0 = time.perf_counter()
    while True:
        if kind == "bench":
            last = _run_bench(circ, order, T, n_words_batch, rng=rng)
        else:
            last = dict(enumerate(_run_aig(circ, order, T, n_words_batch, rng=rng)))
        acc = _accept_words(last, pin_target)
        if acc is None:
            acc = np.full(n_words_batch, _FULL)
        hits += _popcount(acc)
        n += n_words_batch * _W
        el = time.perf_counter() - t0
        if (n >= n_min and el >= budget_s) or el >= hard_cap_s:
            break
    wall = time.perf_counter() - t0
    rate = n / wall if wall else 0.0
    out = {"p_accept": hits / n, "hits": hits, "n_sim": n, "patterns_per_s": rate,
           "accepted_per_s": hits / n * rate, "T": T, "n_pinned": len(pin),
           "target_mode": target_mode}
    if hits == 0:
        out["p_accept_lt"] = 3.0 / n
        out["accepted_per_s"] = 0.0
    return out


def simulate_check(path, cycles, seed, bits, col_labels, pin_outputs=None,
                   target_mode="random", target=None):
    """Encoder-independent soundness: simulate every emitted sample (rows of `bits`, columns
    labelled `input@cycle`) through the ORIGINAL circuit and verify the pinned last-cycle outputs
    reproduce the conversion target. Returns the number of BAD rows, which must be 0."""
    kind, circ = _load(path)
    pin, pin_target, T = _pin_and_target(kind, circ, cycles, seed, pin_outputs, target_mode,
                                         target)
    bits = np.asarray(bits, dtype=np.uint8)
    n, S = bits.shape
    names = (circ[0] if kind == "bench"
             else [circ.input_names.get(k, f"i{k}") for k in range(circ.n_inputs)])
    key_of = {nm: nm for nm in names} if kind == "bench" else {nm: k
                                                               for k, nm in enumerate(names)}
    n_words = (n + _W - 1) // _W

    def pack(col):
        padded = np.zeros(n_words * _W, dtype=np.uint8)
        padded[:n] = col
        return np.packbits(padded, bitorder="little").view(np.uint64)

    given = {}
    for j, lbl in enumerate(col_labels):
        nm, _, t = lbl.rpartition("@")
        if nm not in key_of:
            raise ValueError(f"sample column {lbl!r} is not an input of {path}")
        given[(key_of[nm], int(t))] = pack(bits[:, j])
    for nm in names:
        for t in range(T):
            given.setdefault((key_of[nm], t), np.zeros(n_words, dtype=np.uint64))
    order = _topo_gates(circ) if kind == "bench" else _topo_ands(circ)
    if kind == "bench":
        last = _run_bench(circ, order, T, n_words, given=given)
    else:
        last = dict(enumerate(_run_aig(circ, order, T, n_words, given=given)))
    acc = _accept_words(last, pin_target)
    if acc is None:
        return 0
    ok = np.unpackbits(acc.view(np.uint8), bitorder="little")[:n]
    return int(n - ok.sum())
