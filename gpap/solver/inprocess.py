"""GPU-accelerated inprocessing primitives: discover entailed short clauses in parallel.

`GPUInprocessor` wraps a `CnfFormula` + `BCPEngine` and exposes batched discovery of
clauses that are *entailed* by the formula (RUP) but non-trivial, suitable for injection
into a CDCL solver during search:

  probe_failed_literals()    — L=1. Assume each of the 2N literals alone; any whose BCP
                               propagates to a conflict is a *failed literal*, so its
                               negation is a forced unit clause.
  discover_binaries(vars)    — all (±a, ±b) over a variable subset; a joint assumption
                               that propagates to a conflict yields a binary (¬a ∨ ¬b).
  discover_ternaries(vars)   — all (±a, ±b, ±c) over a subset → ternary clause.

Every returned clause is the negation of a conflicting assumption row, restricted to
`engine.real_conflict` (BCP-derived, round_count > 1) — so trivial round-1 conflicts
(where the assumption directly violates an existing clause) are excluded and the clause
is never subsumed by a single existing clause.

The formula + engine are built **once**; each call only does `reset_state(K)` + apply +
propagate, so the immutable-formula cost is amortized. K is auto-sized to the GPU once at
construction (`compute_k_safe`, uncapped by default).

Variable subsets are **0-based** variable indices (0..N-1), matching the internal 2N
literal-index scheme (`lit_index = 2*var0 + polarity`). Callers working in DIMACS (1-based)
must subtract 1.
"""
import time

import torch

from gpap.clsdata.cnf_formula import CnfFormula
from gpap.clsdata.occ_list import ext_lit_to_lit_index
from gpap.solver.bcp_engine import BCPEngine
from gpap.solver.clause_learner import extract_conflict_clauses_tensor
from gpap.utils.gpu_budget import compute_k_safe, gpu_mem_available


# ---------------------------------------------------------------------------
# Subset assumption generators (0-based variable indices)
# ---------------------------------------------------------------------------

def _as_sorted_unique_vars(var_subset, device: torch.device) -> torch.Tensor:
    """Coerce a list/tensor of 0-based variable indices to a sorted, deduped 1D int64 tensor."""
    t = torch.as_tensor(var_subset, dtype=torch.long, device=device).flatten()
    if t.numel() == 0:
        return t
    return torch.unique(t)  # sorted + deduped


def generate_subset_pairs_chunks_gpu(var_subset, K: int, device: torch.device):
    """Yield (≤K, 2) int64 lit-index tensors for all (±a, ±b) pairs over a variable subset.

    `var_subset`: 0-based variable indices (subset of range(N)). Pairs are formed over
    distinct variables a < b from the (sorted, deduped) subset, each with all 4 polarity
    combinations. Total rows = s*(s-1)*2 where s = |subset|. Mirrors
    `generate_all_pairs_chunks_gpu` but indexes the subset instead of arange(N).
    """
    sub = _as_sorted_unique_vars(var_subset, device)
    if sub.numel() < 2:
        return
    pol = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long, device=device)  # (4,2)
    combos = torch.combinations(sub, r=2)            # (C(s,2), 2), a < b
    base = 2 * combos                                # positive lit indices
    rows = (base.unsqueeze(1) + pol.unsqueeze(0)).reshape(-1, 2)  # (C(s,2)*4, 2)
    for c in range(0, rows.size(0), K):
        yield rows[c:c + K]


def generate_anchor_pairs_chunks_gpu(anchor_vars, num_vars: int, K: int, device: torch.device):
    """Yield (≤K, 2) lit-index tensors for pairs (anchor × every other variable), all polarities.

    `anchor_vars`: 0-based "active" variables. Each anchor is paired with every other variable in
    0..num_vars-1 (both polarities) — so each binary touches an active variable but the *other*
    endpoint ranges over the whole formula, avoiding the subsumption trap of within-active-set
    (clique) pairs. Total ≈ |anchors|·(num_vars−1)·4 rows — a subset of all C(N,2)·4 pairs.
    Duplicate pairs (when two anchors pair with each other) are removed downstream by clause dedup.
    """
    anchors = _as_sorted_unique_vars(anchor_vars, device)
    if anchors.numel() == 0 or num_vars < 2:
        return
    pol = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long, device=device)
    allv = torch.arange(num_vars, dtype=torch.long, device=device)
    for a in anchors.tolist():
        others = allv[allv != a]                                   # (num_vars-1,)
        base = torch.empty((others.numel(), 2), dtype=torch.long, device=device)
        base[:, 0] = 2 * a
        base[:, 1] = 2 * others
        rows = (base.unsqueeze(1) + pol.unsqueeze(0)).reshape(-1, 2)
        for c in range(0, rows.size(0), K):
            yield rows[c:c + K]


def generate_subset_triplets_chunks_gpu(var_subset, K: int, device: torch.device):
    """Yield (≤K, 3) int64 lit-index tensors for all (±a, ±b, ±c) triples over a variable subset.

    `var_subset`: 0-based variable indices. Triples over distinct a < b < c with all 8
    polarity combos. Total rows = C(s,3)*8. Intended for SMALL subsets (e.g. ≤ ~32) — the
    C(s,3) combination tensor is materialized in one shot.
    """
    sub = _as_sorted_unique_vars(var_subset, device)
    if sub.numel() < 3:
        return
    pol = torch.tensor(
        [[p1, p2, p3] for p1 in (0, 1) for p2 in (0, 1) for p3 in (0, 1)],
        dtype=torch.long, device=device,
    )  # (8, 3)
    combos = torch.combinations(sub, r=3)            # (C(s,3), 3), a < b < c
    base = 2 * combos
    rows = (base.unsqueeze(1) + pol.unsqueeze(0)).reshape(-1, 3)  # (C(s,3)*8, 3)
    for c in range(0, rows.size(0), K):
        yield rows[c:c + K]


# ---------------------------------------------------------------------------
# GPU inprocessor
# ---------------------------------------------------------------------------

class GPUInprocessor:
    """Batched GPU discovery of entailed short clauses for solver inprocessing.

    Solver-agnostic: it knows nothing about CaDiCaL. Build it once over the original
    formula, then call the discovery methods repeatedly (e.g. between solver segments).
    `_emitted` deduplicates across all calls for this instance's lifetime so a clause is
    returned at most once.
    """

    def __init__(
        self,
        formula: CnfFormula,
        device: torch.device,
        *,
        k_cap: int | None = None,
        max_rounds: int | None = None,
        soft_stop_frac: float | None = None,
        min_rounds: int = 0,
    ):
        self.formula = formula
        self.device = device
        self.N = formula.num_vars
        self.M = formula.num_clauses
        self.engine = BCPEngine(formula)
        self._max_cls = int(formula.clause_sizes.max().item()) if self.M else 1
        self._total_lits = int(formula.clause_sizes.sum().item()) if self.M else 0
        # K auto-sized to the GPU once; mem_get_info already excludes the loaded formula.
        self.K = compute_k_safe(
            self.M, self.N, gpu_mem_available(device), cap=k_cap,
            max_clause_size=self._max_cls, total_lits=self._total_lits or None,
            deduct_formula=False,
        )
        self._mr = max_rounds
        self._ssf = soft_stop_frac
        self._minr = min_rounds
        self._emitted: set[tuple[int, ...]] = set()

    @classmethod
    def from_cnf_file(cls, path: str, device: torch.device, **kw) -> "GPUInprocessor":
        formula = CnfFormula(device)
        formula.parse_cnf_file(path)
        return cls(formula, device, **kw)

    @classmethod
    def from_clauses(cls, num_vars: int, clauses, device: torch.device, **kw) -> "GPUInprocessor":
        formula = CnfFormula(device)
        formula.init_formula(num_vars, clauses)
        return cls(formula, device, **kw)

    # ----------------------------------------------------------------- internals

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _run_chunk(self, assum_lit_idx: torch.Tensor) -> list[list[int]]:
        """reset → apply → propagate → extract for one (K_chunk, L) lit-index tensor."""
        K_chunk = assum_lit_idx.size(0)
        if K_chunk == 0:
            return []
        eng = self.engine
        eng.reset_state(K_chunk)
        eng.apply_assumptions_bulk(assum_lit_idx)
        eng.propagate(max_rounds=self._mr, soft_stop_frac=self._ssf, min_rounds=self._minr)
        return extract_conflict_clauses_tensor(assum_lit_idx, eng.real_conflict)

    def _collect(self, chunk_gen) -> tuple[list[list[int]], int]:
        """Run every chunk; return (all conflict clauses before dedup, total rows propagated)."""
        raw: list[list[int]] = []
        n_rows = 0
        for chunk in chunk_gen:
            n_rows += int(chunk.size(0))
            raw.extend(self._run_chunk(chunk))
        return raw, n_rows

    def _emit_new(self, clauses: list[list[int]]) -> list[list[int]]:
        """Dedup against everything emitted so far (order-insensitive); record + return new ones."""
        out: list[list[int]] = []
        for c in clauses:
            key = tuple(sorted(c))
            if key not in self._emitted:
                self._emitted.add(key)
                out.append(c)
        return out

    def reset_emitted(self):
        """Forget previously emitted clauses (allow re-discovery)."""
        self._emitted.clear()

    # ----------------------------------------------------------------- discovery

    def probe_failed_literals(self) -> tuple[list[list[int]], dict]:
        """L=1 failed-literal elimination → forced unit clauses (each = [¬lit])."""
        self._sync()
        t0 = time.perf_counter()
        rows = torch.arange(2 * self.N, dtype=torch.long, device=self.device).unsqueeze(1)  # (2N,1)
        raw, n_rows = self._collect(rows[c:c + self.K] for c in range(0, rows.size(0), self.K))
        new = self._emit_new(raw)
        self._sync()
        stats = {"mode": "probe", "n_rows": n_rows, "n_conflicts": len(raw),
                 "n_new": len(new), "K": self.K, "ms": (time.perf_counter() - t0) * 1e3}
        return new, stats

    def discover_binaries(self, var_subset) -> tuple[list[list[int]], dict]:
        """All (±a, ±b) over a 0-based variable subset → binary clauses (¬a ∨ ¬b)."""
        self._sync()
        t0 = time.perf_counter()
        gen = generate_subset_pairs_chunks_gpu(var_subset, self.K, self.device)
        raw, n_rows = self._collect(gen)
        new = self._emit_new(raw)
        self._sync()
        stats = {"mode": "binaries", "n_rows": n_rows, "n_conflicts": len(raw),
                 "n_new": len(new), "K": self.K, "ms": (time.perf_counter() - t0) * 1e3}
        return new, stats

    def discover_binaries_anchored(self, anchor_vars) -> tuple[list[list[int]], dict]:
        """Binaries between active anchor variables and every other variable (¬anchor ∨ ¬other).

        Each pair touches an anchor (e.g. a VSIDS-hot variable) but reaches across the whole
        formula, so the resulting binaries are relevant to the search yet not subsumed the way
        within-anchor-set (clique) pairs tend to be. `anchor_vars` are 0-based.
        """
        self._sync()
        t0 = time.perf_counter()
        gen = generate_anchor_pairs_chunks_gpu(anchor_vars, self.N, self.K, self.device)
        raw, n_rows = self._collect(gen)
        new = self._emit_new(raw)
        self._sync()
        stats = {"mode": "binaries_anchored", "n_rows": n_rows, "n_conflicts": len(raw),
                 "n_new": len(new), "K": self.K, "ms": (time.perf_counter() - t0) * 1e3}
        return new, stats

    def discover_ternaries(self, var_subset) -> tuple[list[list[int]], dict]:
        """All (±a, ±b, ±c) over a 0-based variable subset → ternary clauses."""
        self._sync()
        t0 = time.perf_counter()
        gen = generate_subset_triplets_chunks_gpu(var_subset, self.K, self.device)
        raw, n_rows = self._collect(gen)
        new = self._emit_new(raw)
        self._sync()
        stats = {"mode": "ternaries", "n_rows": n_rows, "n_conflicts": len(raw),
                 "n_new": len(new), "K": self.K, "ms": (time.perf_counter() - t0) * 1e3}
        return new, stats

    # ----------------------------------------------------------------- vivification

    def _bcp_conflict_mask(self, rows_cpu: torch.Tensor, max_rounds: int | None) -> torch.Tensor:
        """BCP every assumption row; return a (R,) CPU bool conflict mask.

        `rows_cpu`: (R, L) int64, -1 padded, on CPU. Processed in chunks of self.K. A row
        conflicts iff assuming its literals propagates to a conflict (any conflict — trivial
        or BCP-derived — proves the row's literals are jointly unsatisfiable with the formula).
        """
        R = rows_cpu.size(0)
        out = torch.zeros(R, dtype=torch.bool)
        if R == 0:
            return out
        eng = self.engine
        for c0 in range(0, R, self.K):
            chunk = rows_cpu[c0:c0 + self.K].to(self.device)
            eng.reset_state(chunk.size(0))
            eng.apply_assumptions_bulk(chunk)
            eng.propagate(max_rounds=max_rounds, soft_stop_frac=None, min_rounds=0)
            out[c0:c0 + chunk.size(0)] = eng.conflict.to("cpu")
        self._sync()
        return out

    def vivify_clauses(
        self, clauses: list[list[int]], *, max_len: int = 20, max_rounds: int | None = None,
    ) -> tuple[list[list[int]], dict]:
        """Shorten each clause to its shortest entailed prefix (equivalence-preserving).

        For a clause C = (l1 ∨ … ∨ lk) with 2 ≤ k ≤ max_len, assume the negated prefix
        (¬l1, …, ¬lp) and BCP; the smallest p whose propagation conflicts proves the formula
        entails (l1 ∨ … ∨ lp), which subsumes C — so C may be replaced by that shorter clause
        without changing the model set. Two passes: pass 1 tests the full (k−1)-prefix (one row
        per clause) to find which clauses shorten at all; pass 2 refines only those to the
        minimal p. `max_rounds` caps BCP depth — a lower cap only under-vivifies, never unsound.

        The clauses MUST be the same ones this inprocessor was built from (the engine propagates
        over that formula). Returns (new_clauses, stats); new_clauses[i] replaces clauses[i] in
        order (unchanged where no prefix conflicts).
        """
        t0 = time.perf_counter()
        new = [list(c) for c in clauses]
        to_viv = [i for i, c in enumerate(clauses) if 2 <= len(c) <= max_len]
        stats = {"mode": "vivify", "n_candidates": len(to_viv), "n_vivified": 0,
                 "lits_removed": 0, "K": self.K}
        if not to_viv:
            stats["ms"] = (time.perf_counter() - t0) * 1e3
            return new, stats

        # neg lit-index matrix over the vivification candidates (local rows), -1 padded.
        maxk = max(len(clauses[i]) for i in to_viv)
        maxlen = maxk - 1
        neg = torch.full((len(to_viv), maxk), -1, dtype=torch.long)
        klen = torch.empty(len(to_viv), dtype=torch.long)
        for r, i in enumerate(to_viv):
            c = clauses[i]
            klen[r] = len(c)
            neg[r, :len(c)] = torch.tensor([ext_lit_to_lit_index(l) ^ 1 for l in c], dtype=torch.long)
        tv = torch.tensor(to_viv, dtype=torch.long)

        def rows_for(local_idx: torch.Tensor, p_len: torch.Tensor) -> torch.Tensor:
            """Build (R, maxlen) negated-prefix rows: row r = neg[local_idx[r], :p_len[r]]."""
            R = local_idx.numel()
            block = torch.full((R, maxlen), -1, dtype=torch.long)
            for p in torch.unique(p_len).tolist():
                m = p_len == p
                block[m, :p] = neg[local_idx[m], :p]
            return block

        # --- Pass 1: full (k−1)-prefix per candidate (monotonicity: if it does not conflict,
        #     no shorter prefix conflicts either, so the clause is left untouched). ---
        local = torch.arange(len(to_viv))
        conf1 = self._bcp_conflict_mask(rows_for(local, klen - 1), max_rounds)
        viv_local = local[conf1]
        if viv_local.numel() == 0:
            stats["ms"] = (time.perf_counter() - t0) * 1e3
            return new, stats
        best_p = {int(tv[l]): int(klen[l]) - 1 for l in viv_local.tolist()}  # default = k−1

        # --- Pass 2: refine vivifiable clauses to the minimal prefix (p = 1 … k−2). ---
        rc, rp = [], []
        for l in viv_local.tolist():
            k = int(klen[l])
            for p in range(1, k - 1):
                rc.append(l); rp.append(p)
        if rc:
            rc_t = torch.tensor(rc, dtype=torch.long)
            rp_t = torch.tensor(rp, dtype=torch.long)
            conf2 = self._bcp_conflict_mask(rows_for(rc_t, rp_t), max_rounds)
            for j in torch.nonzero(conf2, as_tuple=False).view(-1).tolist():
                i, p = int(tv[rc_t[j]]), int(rp_t[j])
                if p < best_p[i]:
                    best_p[i] = p

        removed = 0
        for i, p in best_p.items():
            removed += len(clauses[i]) - p
            new[i] = list(clauses[i][:p])
        stats.update(n_vivified=len(best_p), lits_removed=removed,
                     ms=(time.perf_counter() - t0) * 1e3)
        return new, stats

    def run(
        self,
        *,
        probe: bool = True,
        pairs_vars=None,
        triples_vars=None,
        max_clauses: int | None = None,
    ) -> tuple[list[list[int]], dict]:
        """Config-driven discovery round: probe + binaries + ternaries, capped + shortest-first.

        Returns (clauses, stats) where `clauses` are new (deduped vs this instance's history),
        sorted shortest-first and truncated to `max_clauses` (units kept first — highest value).
        """
        out: list[list[int]] = []
        stats: dict = {}
        if probe:
            c, s = self.probe_failed_literals()
            out += c
            stats["probe"] = s
        if pairs_vars is not None and len(pairs_vars) >= 2:
            c, s = self.discover_binaries(pairs_vars)
            out += c
            stats["binaries"] = s
        if triples_vars is not None and len(triples_vars) >= 3:
            c, s = self.discover_ternaries(triples_vars)
            out += c
            stats["ternaries"] = s
        out.sort(key=len)  # shortest-first: units, then binaries, then ternaries
        if max_clauses is not None and len(out) > max_clauses:
            out = out[:max_clauses]
        stats["n_total_new"] = len(out)
        return out, stats
