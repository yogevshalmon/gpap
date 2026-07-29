"""GPU-parallel SAT solution sampler via exact BCP-completion.

`GPUSampler` draws satisfying assignments by running K independent randomized
DPLL-completions in parallel on the GPU. Each "trajectory" repeatedly:

  1. picks the next B free (unassigned) variables per sample,
  2. assigns each a *random per-sample polarity* (a decision),
  3. runs BCP to fixpoint,

all inside ONE fused kernel launch per chunk (engine.run_trajectory — the engine is the C++
core `libgpap`, which requires CUDA; its kernels are Triton, compiled ahead of time).

A trajectory that reaches a conflict is a dead-end (discarded); a trajectory whose
variables all become assigned with no conflict is a **guaranteed-valid model** — every
clause keeps a non-falsified literal, so the assignment satisfies the formula exactly
(no rounding / repair, unlike gradient-relaxation GPU samplers).

This is the "uniform-like, speed-over-uniformity" regime (cf. QuickSampler / CMSGen):
samples are biased by the propagation order, but produced by the millions and every one
is valid. The sampler is a *throughput* primitive — its fit depends on how shallow the
decision depth is (heavily-propagating structured formulas → few decisions → few rounds).

Design knobs (the throughput vs. success-rate tradeoff):
  decisions_per_round : assign B free vars at once. B=1 = textbook DPLL (most rounds,
                        highest success rate); B>1 = a random restriction (fewer rounds,
                        more dead-ends).
  random_var          : pick a random free var per sample (diverse) vs. the first free
                        one (cheap, low-index-biased).

K is auto-sized to the GPU (capped by default to stay off the WSL2 host-RAM spill cliff).
"""
import os
# Use CUDA expandable-segments allocation BEFORE torch initializes its CUDA allocator. The sampler's
# hot loop churns many VARIABLE-sized temporaries per round (propagate scratch sized by the frontier;
# the (K,N) decision masks), which fragments the default caching allocator badly: measured peak
# RESERVED = 6.5 GB while live tensors were only 2.5 GB (62% wasted to fragmentation), and that
# fragmentation is what "grows mid-run" and forces a small K. expandable_segments grows one virtual
# segment instead of many fixed blocks → reserved tracks allocated (6.5→2.8 GB here, ~2.3× more K
# for the same VRAM). setdefault so an explicit user PYTORCH_CUDA_ALLOC_CONF still wins. Must be set
# before the first CUDA allocation; importing this module before any CUDA use guarantees that.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time

import numpy as np
import torch

from gpap.engine.formula import CnfFormula
from gpap.engine.bcp_engine import BCPEngine
from gpap.engine.calibrate import compute_k_safe, gpu_mem_available


class GPUSampler:
    """Batched GPU sampler: K parallel randomized BCP-completions per chunk."""

    def __init__(
        self,
        formula: CnfFormula,
        device: torch.device,
        *,
        k_cap: int | None = 16384,
        seed: int | None = None,
    ):
        self.formula = formula
        self.device = device
        self.N = formula.num_vars
        self.M = formula.num_clauses
        # The engine core (the ONLY execution path — raises off-CUDA / without libgpap;
        # field split 16|16 or 8|24 picked per formula, see gpap.engine).
        self.engine = BCPEngine(formula)
        # Shape comes from the engine core (free) — never a reduction over a (M,) tensor.
        self._max_cls = formula.max_clause_len or 1
        self._total_lits = formula.total_lits
        # RNG + state must exist before K calibration (the probe chunk in _calibrate_k uses _gen).
        self._gen = torch.Generator(device=device)
        if seed is not None:
            self._gen.manual_seed(seed)
        import random as _random
        self._pyrng = _random.Random(seed)
        self.support: list[int] | None = None   # 0-based independent-support vars (from `c ind`)
        self._recon: dict | None = None          # set when built with simplify=True (see _from_simplified)
        # K auto-sized to the GPU (see _calibrate_k): K = min(analytic, cap) — the analytic
        # model matches the packed engine's real footprint (retuned 2026-07-13, kill-switch
        # validated; the PyTorch-era 70M+40N floor that used to bind is gone with that path).
        free_bytes = gpu_mem_available(device)
        analytic = compute_k_safe(
            self.M, self.N, free_bytes, cap=k_cap,
            max_clause_size=self._max_cls, total_lits=self._total_lits or None,
            deduct_formula=False,
        )
        self.K = self._calibrate_k(free_bytes, k_cap, analytic)
        self._K_init = self.K
        # FUSED-config K boost: a fused-trajectory chunk never builds the (K,N) per-round decision
        # masks — its true footprint is just the engine state + the trail + a few (K,N) bool readout
        # temps: ≈ (sat_bytes+4)·M + 10·N per batch. That is far below the v1 heuristic, so fused
        # chunks can run a much larger K (throughput ∝ K). Applied per-chunk in sample() ONLY for
        # fused configs and only while K is at its construction value (a lowered K — e.g. the
        # coverage run — is honoured). Hard cap 262,144; validated under the nvidia-smi kill-switch.
        if self.M > 0:
            sat_b = 1 if self._max_cls <= 127 else (2 if self._max_cls <= 32767 else 4)
            # Hard ceiling on the fused-K boost. 262,144 was kill-switch-validated on the 12 GB 4080;
            # on a big-VRAM GPU (A100 80 GB) the caller's VRAM-proportional k_cap is larger, so let it
            # raise the ceiling — the memory-safe `budget // fused_pb` below still bounds the actual K.
            # No-op on the 4080 (its k_cap < 262,144); unlocks A100 fused headroom for the K-scaling story.
            fused_cap = max(262_144, k_cap or 0)
            # Per-batch fused footprint: ONE (K,M) int32 packed clause word + ~10N of assignment,
            # trail and readout temps (no per-round decision masks — everything is in-kernel).
            # The assignment itself is a BITMAP now (N/4 bytes, was 2N) — the 10N budget is kept
            # UNCHANGED as conservative headroom; shrinking it = a K change = kill-switch re-run.
            fused_pb = 4 * self.M + 300 + 10 * self.N
            # eng_hold: kept UNCONDITIONALLY as headroom although no base-K sat/lsum exists any
            # more — it was ACCIDENTALLY covering the fixed 4096×M soundness-check temporary, and
            # skipping it spilled b05-c50 under the nvidia-smi kill-switch. If the ~9% K is ever
            # wanted back, it must go with an explicit 4096·M reservation + kill-switch re-run.
            eng_hold = ((sat_b + 4) * self.M + 2 * self.N) * self.K
            budget = max(0, int(0.6 * (free_bytes - eng_hold)))
            self.K_fused = max(self.K, min(budget // fused_pb, fused_cap))
        else:
            self.K_fused = self.K
        # Kernel-variant auto-dispatch on the CALIBRATED K (a pure launch flag, gate-validated
        # bit-identical): below memory saturation each row's serial trail leaves requests-in-
        # flight on the table, and the warp-wide WAVE propagate multiplies them. Crossover
        # RETUNED 2026-07-14 after the assignment-BITMAP change removed wave's WxW dedup tax:
        # wave now wins well past saturation (b03 K=49k = 17.7x resident: wave 1.41x, was
        # 0.9x pre-bitmap) and scalar only holds the huge-K/tiny-M extreme (b06 K=129k = 46x
        # resident: scalar 1.13x, interleaved A/Bs) — boundary set mid-gap at ~30x resident
        # warps (83k here, device-derived so it transfers to bigger GPUs). GPAP_KERNEL forces.
        if os.environ.get("GPAP_KERNEL") is None and device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            resident = props.multi_processor_count * (props.max_threads_per_multi_processor // 32)
            self.engine.set_mode("wave" if self.K_fused < 30 * resident else "scalar")

    # Byte cap for one (rows, M) int32 soundness-check slice (see sample()); bounds the
    # transient regardless of M so the K budget needs no silent headroom for it.
    _SOUND_SLICE_BYTES = 256e6

    def _calibrate_k(self, free_bytes: int, k_cap: int | None, analytic_k: int) -> int:
        """Size K to the spill-safe VRAM budget: K = min(analytic, cap).

        RETUNED 2026-07-13 (kill-switch validated): the old extra floor 4×0.7·free/(70M+40N)
        modeled the DELETED PyTorch path's scatter temporaries and bound K everywhere at ~1/3
        of what the packed engine actually fits (measured real engine bytes/row = 4M+2N; the
        roofline showed exactly those instances MLP-starved). The analytic model
        (compute_k_safe ≈ (sat_b+4)M + 2N + 8N-class temps) matches the packed engine's true
        footprint incl. the (K,N) readout masks, so it IS the floor now. The 4096×M soundness
        transient that the old headroom accidentally covered is gone (sat_rows is read in
        bounded slices — see _SOUND_SLICE_BYTES).

        A runtime memory probe was tried and abandoned (2026-07-01): with expandable_segments
        the peak-reserved read at construction is systematically ~1.6× below production and the
        peak varies with the RNG draw → probes under-measure → spill. Do not resurrect it.
        CPU / M==0: analytic only (no CUDA budget to exploit)."""
        cap = k_cap if k_cap else analytic_k
        return max(1, min(analytic_k, cap))

    @classmethod
    def from_cnf_file(cls, path: str, device: torch.device, *, simplify: bool = False,
                      els: bool = False, **kw) -> "GPUSampler":
        """Build a sampler from a DIMACS file. `simplify=True` first applies unit-propagation
        simplification (shrinks M -> larger K -> more throughput) and samples the compacted
        formula, reconstructing full original models at collection time (see simplify_cnf).
        `els=True` additionally does equivalent-literal substitution (simplify_cnf_els): removes
        more variables/clauses -> faster propagate + bigger K (implies simplify)."""
        support = parse_independent_support(path)   # 0-based, or None
        if not (simplify or els):
            formula = CnfFormula(device)
            formula.parse_cnf_file(path)
            smp = cls(formula, device, **kw)
            smp.support = support
            return smp
        from pysat.formula import CNF
        cnf = CNF(from_file=path)
        return cls._from_simplified(cnf.nv, cnf.clauses, support, device, els=els, **kw)

    @classmethod
    def from_clauses(cls, num_vars: int, clauses, device: torch.device, *,
                     simplify: bool = False, els: bool = False,
                     support: list | None = None, **kw) -> "GPUSampler":
        if simplify or els:
            return cls._from_simplified(num_vars, clauses, support, device, els=els, **kw)
        formula = CnfFormula(device)
        formula.init_formula(num_vars, clauses)
        smp = cls(formula, device, **kw)
        smp.support = support
        return smp

    @classmethod
    def _from_simplified(cls, nv: int, clauses: list, support: list | None,
                         device: torch.device, *, els: bool = False, **kw) -> "GPUSampler":
        # `clauses` is treated read-only here and by simplify_cnf/simplify_cnf_els (they copy
        # before mutating) — no defensive deep copy: it costs ~1s per call on 1M-clause unrolls.
        if els:
            res = simplify_cnf_els(nv, clauses, support)
            if res is None:
                return cls._unsat_sampler(device, **kw)
            nv2, cl2, sup2, recon = res
            formula = CnfFormula(device)
            formula.init_formula(nv2, cl2)
            smp = cls(formula, device, **kw)
            smp.support = sup2
            smp._recon = {"unsat": False, "els": True, **recon}
            return smp
        res = simplify_cnf(nv, clauses, support)
        if res is None:                                  # UNSAT after unit propagation
            return cls._unsat_sampler(device, **kw)
        nv2, cl2, sup2, old_of_new, forced, free_support = res
        formula = CnfFormula(device)
        formula.init_formula(nv2, cl2)
        smp = cls(formula, device, **kw)
        smp.support = sup2
        smp._recon = {"unsat": False, "old_of_new": old_of_new, "forced": forced,
                      "free_support": free_support, "orig_nv": nv,
                      "orig_support": sorted(support) if support else []}
        return smp

    @classmethod
    def _unsat_sampler(cls, device: torch.device, **kw) -> "GPUSampler":
        formula = CnfFormula(device)
        formula.init_formula(1, [[1], [-1]])
        smp = cls(formula, device, **kw)
        smp.support = []
        smp._recon = {"unsat": True}
        return smp

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    # ----------------------------------------------------------------- one chunk

    def _pair_masks(self, support_mask: "torch.Tensor | None"):
        """(all_vars_id, support_id|None) engine pair-mask IDs for the completion checks.
        Cached; keyed by support-mask IDENTITY with a strong ref held — sample() builds ONE
        support_mask per call, so this registers once per sample() not per chunk."""
        if getattr(self, "_pm_ref", "unset") is not support_mask:
            self._pm_ref = support_mask
            self._pm_all = self.engine.make_pair_mask(None)
            self._pm_sup = (self.engine.make_pair_mask(support_mask)
                            if support_mask is not None else None)
        return self._pm_all, self._pm_sup

    @torch.inference_mode()
    def _run_chunk(self, K: int, B: int, random_var: bool, max_rounds: "int | None",
                   support_mask: torch.Tensor | None, complete: bool, lookahead=False):
        """One chunk of K trajectories — the WHOLE decide+propagate loop in ONE fused kernel
        launch (engine.run_trajectory). Decision order = support vars first (phase 1), then, if
        `complete`, the remaining vars (phase 2); the kernel never straddles the phase boundary.
        `lookahead` in ("soft", "fsoft") enables the in-kernel risk-polarity; `max_rounds` is
        accepted for API compatibility but never applied (the kernel structurally terminates in
        <= |order| decision groups; a cap cripples B=1 on large supports — measured 2x on s832).

        Returns (n_fully, n_support_complete, rounds, free_after_support):
          n_fully            : rows that are active AND have no free var -> guaranteed models.
                               The row selection stays on the GPU in engine slot 0 (slot 1 holds
                               the support-complete selection); read it out with the engine's
                               selection accessors, no (K,) mask ever crosses the bus.
          free_after_support : always 0.0 (unmeasurable mid-trajectory in the fused path).
        """
        eng = self.engine
        # reset broadcast-restores the cached post-units-flush state: no (K,M) state writes
        # beyond one copy, no per-chunk units flush.
        eng.reset(K)
        if random_var:
            priority = torch.randperm(self.N, generator=self._gen, device=self.device)
        else:
            priority = torch.arange(self.N, device=self.device)
        # Picking the next-B-free-vars in priority order happens inside the kernel; polarity is
        # a per-(chunk,k,v) hash bit seeded below (torch RNG stream consumed ONCE per chunk).
        rank = torch.argsort(priority)                  # vars in priority order
        if support_mask is not None:
            in_sup = support_mask[rank]
            sup_part = rank[in_sup]
            order = torch.cat([sup_part, rank[~in_sup]]) if complete else sup_part
            n_support = int(sup_part.numel())
        else:
            order = rank
            n_support = order.numel()                   # single phase: boundary = end
        seed = int(torch.randint(0, 2 ** 62, (1,), generator=self._gen,
                                 device=self.device).item())
        rounds = eng.run_trajectory(order, n_support, B, seed,
                                    soft=(lookahead in ("soft", "fsoft")))
        # Completion checks straight off the assignment BITMAP (one bitwise pass over the words
        # inside the engine; only the COUNT crosses the bus). The former (K, N) free-mask
        # materialization cost ~16x its bytes in strided sector RMWs.
        pm_all, pm_sup = self._pair_masks(support_mask)
        n_fully = eng.select_complete(pm_all, slot=0)
        n_sup = (eng.select_complete(pm_sup, slot=1) if support_mask is not None else n_fully)
        return n_fully, n_sup, rounds, 0.0

    @torch.inference_mode()
    def _autotune(self, K, B_hint, max_rounds, support_mask, complete, lookahead, probe_k=2048):
        """Probe a small (random_var, B) grid and return the (rv, B) with the best THROUGHPUT
        (models / wall) on this formula. Both are strong, formula-dependent levers (polarity stays
        random throughout, so diversity / 2-wise coverage are preserved — validated == CMSGen):
          - niche (support BCP-determines the rest): B=8, order irrelevant — fast, high success.
          - "service" family (search-heavy residual, freeAft≈685): random_var=False (structured order
            follows the gate DAG) + B=1 (avoid over-committing) is the ONLY combo that completes them
            (ActivityService modelOK 0.05%→12%); B≥4 OR random order → ≈0.
          - low-supOK family (7/80.sk): random_var=True.
        Selecting by throughput (not raw count) never trades a fast high-success combo for a slow one:
        a B=1 run that collects more models per chunk but takes 60 rounds loses to a 3-round B=8 run.
        Niche fast-path: if (random, B_hint) already succeeds well, return immediately so the niche
        pays one cheap probe; only the lose-cases explore the full grid (≤6 chunks)."""
        pk = max(1, min(K, probe_k))

        def probe(rv, B):
            self._sync(); t0 = time.perf_counter()
            n, _, _, _ = self._run_chunk(pk, B, rv, max_rounds, support_mask, complete, lookahead)
            self._sync()
            return n, time.perf_counter() - t0

        n0, dt0 = probe(True, B_hint)
        if n0 / pk > 0.1:                       # workable at (random, B_hint) — keep it; B_hint is fast.
            # (Threshold is deliberately low: a small-K probe under-rates large B — fewer rounds means
            #  kernel-launch overhead dominates its wall — so the grid over-favors B=1. If random+B_hint
            #  already produces at a usable rate, don't risk that misjudgment; only truly-stuck formulas
            #  (service family ≈0 here) fall through to the full (order,B) grid.)
            return True, B_hint
        best = (n0 / dt0 if dt0 > 0 else 0.0, True, B_hint)
        for rv in (True, False):
            for B in sorted({1, 4, B_hint}):
                if rv is True and B == B_hint:  # already probed above
                    continue
                n, dt = probe(rv, B)
                rate = n / dt if dt > 0 else 0.0
                if rate > best[0]:
                    best = (rate, rv, B)
        return best[1], best[2]

    def _select_config(self, plan, stats, chunk, exploit=False):
        """Adaptive (bandit) config selection: dynamically switch to the best-THROUGHPUT config as it
        is measured on real full-K chunks. More robust than the small-K `_autotune` probe (which
        under-rates large B because kernel-launch overhead dominates its few rounds, and commits
        permanently): here every measurement is a productive full-K chunk and the choice can change.
        Policy: try each config once, then decaying-epsilon-greedy (explore shrinks as 1/sqrt(chunk),
        floored at 5%) on cumulative models/wall — keeps re-checking alternatives but spends most of
        the budget on the winner (niche → (random,8); service → (structured,1)).
        Exploration is PLAUSIBILITY-FILTERED: only configs within 20x of the best measured rate get
        re-checked. Uniform re-checks ignored what a re-check COSTS — measured on b10 T=25 (64-chunk
        collect run): orac at 102 models/s vs the winner's 29.6k (290x) got 5 uniform picks = 20.4s
        of the 29s wall (70%). Run-to-run rate noise is ~1.9x (b06 measurements), so a 20x-slower
        config is dead on this formula, not noisy — it keeps exactly its one cold-start chunk.
        `exploit=True` skips exploration and returns the current best — used in the post-target
        top-up phase to accumulate a stable measurement window on the winner (min_measure_chunks)."""
        rate = lambda c: (stats[c]["models"] / stats[c]["wall"]) if stats[c]["wall"] > 0 else 0.0
        best = lambda: max(plan, key=rate)
        for c in plan:                                  # cold start: one real chunk per config
            if stats[c]["chunks"] == 0:
                return c
        if exploit:
            return best()
        eps = max(0.05, chunk ** -0.5)
        if self._pyrng.random() < eps:
            b = best()
            cands = [c for c in plan if c is not b and rate(c) >= rate(b) / 20]
            if cands:
                return self._pyrng.choice(cands)
        return best()

    def _reconstruct_els(self, comp_rows: list[list[int]]):
        """Reconstruct original models when the formula was reduced by ELS (simplify_cnf_els).
        Every original var resolves to its representative's value via `rep_of`: forced vars take their
        pinned value, decided/propagated reps take the sampled value, and free (don't-care) reps draw
        ONE random bit per equivalence class (so equivalent vars stay equal — a hard requirement, else
        the projection is invalid). Returns (original_models, support_projections)."""
        r = self._recon
        rep_of = r["rep_of"]                        # rep_of[v-1] = (rep_var, same_polarity)
        new_of_new = r["new_of_new"]                # rep_var -> compacted var (1-based)
        forced_map = r["forced_map"]                # var -> bool
        orig_nv = r["orig_nv"]
        orig_support = r["orig_support"]
        out_models, projs = [], []
        for cm in comp_rows:
            model_dict = {abs(cl): (cl > 0) for cl in cm}   # compacted var -> bool
            free_bits: dict[int, bool] = {}
            model = []
            for v in range(1, orig_nv + 1):
                if v in forced_map:
                    b = forced_map[v]
                else:
                    rv, pol = rep_of[v - 1]
                    if rv in forced_map:
                        rval = forced_map[rv]
                    elif rv in new_of_new:
                        rval = model_dict.get(new_of_new[rv], False)
                    else:                                    # don't-care rep: one bit per class
                        rval = free_bits.get(rv)
                        if rval is None:
                            rval = self._pyrng.random() < 0.5
                            free_bits[rv] = rval
                    b = rval if pol else (not rval)
                model.append(v if b else -v)
            out_models.append(model)
            sset = set(model)
            projs.append(tuple((v + 1) if (v + 1) in sset else -(v + 1) for v in orig_support))
        return out_models, projs

    def _reconstruct(self, comp_rows: list[list[int]]):
        """Map full models of the simplified (compacted) formula back to full ORIGINAL models, and
        return (original_models, support_projections). Forced vars are pinned to their unit-propagated
        value; the compacted vars take their sampled value; don't-care support vars (free_support) are
        sampled uniformly at random (so the support-projection distribution is preserved). Remaining
        unmapped original vars are true don't-cares set to False (any value gives a valid model)."""
        if self._recon.get("els"):
            return self._reconstruct_els(comp_rows)
        r = self._recon
        old_of_new = r["old_of_new"]              # old_of_new[i] = original var of compacted var i+1
        forced_map = {abs(l): (l > 0) for l in r["forced"]}
        free_support = r["free_support"]
        orig_nv = r["orig_nv"]
        orig_support = r["orig_support"]
        out_models, projs = [], []
        for cm in comp_rows:
            val = dict(forced_map)                # original var -> bool
            for cl in cm:
                val[old_of_new[abs(cl) - 1]] = cl > 0
            for v0 in free_support:
                val[v0 + 1] = self._pyrng.random() < 0.5
            model = [(v if val.get(v, False) else -v) for v in range(1, orig_nv + 1)]
            out_models.append(model)
            sset = set(model)
            projs.append(tuple((v + 1) if (v + 1) in sset else -(v + 1) for v in orig_support))
        return out_models, projs

    # --------------------------------------------------- efficient bit-projection output

    def _build_projection_plan(self) -> dict:
        """Precompute a per-output-column plan so a chunk's support projection is a GPU gather +
        constant fill + random fill — NO Python per-row reconstruct (the measured delivery
        bottleneck). One-time per formula; the reconstruction rules (simplify / ELS / plain)
        become tensors over the S output columns:
          gcol[i]   = compacted variable column to gather for output column i (-1 if not a gather)
          flip[i]   = XOR applied to the gathered/random bit (ELS negated-representative polarity)
          const[i]  = constant 0/1 for a forced (unit-propagated) column
          rgroup[i] = random-draw group id (-1 if none); ELS columns sharing a representative share
                      a group so equivalent variables stay EQUAL (a hard validity requirement).
        Output columns default to the original support (the sampled object); with no support they
        are all original variables (the full model). Returns/caches the plan dict."""
        if getattr(self, "_proj_plan", None) is not None:
            return self._proj_plan
        r = self._recon
        if r is not None and r.get("orig_support"):
            cols = list(r["orig_support"])                                 # 0-based support vars
        elif r is not None:
            cols = list(range(r["orig_nv"]))                              # full model
        elif self.support:
            cols = list(self.support)
        else:
            cols = list(range(self.N))
        S = len(cols)
        gcol, flip, const, rgroup = [-1] * S, [0] * S, [0] * S, [-1] * S
        groups: dict = {}

        def _group(key):
            return groups.setdefault(key, len(groups))

        if r is None:                                          # sampler decided ORIGINAL vars
            for i, v0 in enumerate(cols):
                gcol[i] = v0                                   # compacted == original
        elif r.get("els"):
            rep_of, nn, fm = r["rep_of"], r["new_of_new"], r["forced_map"]
            for i, v0 in enumerate(cols):
                v = v0 + 1
                if v in fm:
                    const[i] = 1 if fm[v] else 0
                    continue
                rv, pol = rep_of[v - 1]
                if rv in fm:
                    const[i] = 1 if (fm[rv] if pol else not fm[rv]) else 0
                elif rv in nn:
                    gcol[i], flip[i] = nn[rv] - 1, (0 if pol else 1)
                else:                                          # don't-care rep: one draw per class
                    rgroup[i], flip[i] = _group(("els", rv)), (0 if pol else 1)
        else:                                                  # plain unit-prop simplification
            new_of_old = {int(o): j for j, o in enumerate(r["old_of_new"])}
            fm = {abs(l): (l > 0) for l in r["forced"]}
            for i, v0 in enumerate(cols):
                v = v0 + 1
                if v in fm:                                    # unit-forced -> constant
                    const[i] = 1 if fm[v] else 0
                elif v in new_of_old:                          # survives compacted -> gather
                    gcol[i] = new_of_old[v]
                else:                                          # in no surviving clause == free_support
                    rgroup[i] = _group(("free", v0))
        self._proj_plan = {
            "cols_1based": [v0 + 1 for v0 in cols], "S": S, "n_groups": len(groups),
            "gcol": np.asarray(gcol, dtype=np.int64),
            "flip": np.asarray(flip, dtype=np.uint8),
            "const": np.asarray(const, dtype=np.uint8),
            "rgroup": np.asarray(rgroup, dtype=np.int64),
        }
        return self._proj_plan

    @torch.inference_mode()
    def _project(self, n: int, slot: int = 0):
        """(n, S) uint8 support projection for the rows selected in `slot`, via the cached plan.

        The only bytes that leave the device are the gathered support bits themselves (the
        engine does the bit gather); the constant / polarity-flip / don't-care columns are
        assembled here. No per-row Python reconstruct — that was the measured delivery
        bottleneck — and no (K, N) mask."""
        plan = self._build_projection_plan()
        proj = np.broadcast_to(plan["const"], (n, plan["S"])).copy()   # (n, S) uint8
        g = plan["gcol"] >= 0
        if g.any():
            # gather ONLY the needed var columns' TRUE bits straight from the bitmap
            pos = self.engine.var_bits(slot, plan["gcol"][g], n)       # (n, |g|) uint8
            proj[:, g] = pos ^ plan["flip"][g]
        rm = plan["rgroup"] >= 0
        if plan["n_groups"] and rm.any():
            # torch RNG keeps the don't-care draw stream identical to the historical engine
            bits = torch.randint(0, 2, (n, plan["n_groups"]), generator=self._gen,
                                 device=self.device).to(torch.uint8).cpu().numpy()
            proj[:, rm] = bits[:, plan["rgroup"][rm]] ^ plan["flip"][rm]
        return proj

    @torch.inference_mode()
    def sample_bits(self, n_target: int, *, config=(True, 8, "fused"),
                    max_decision_rounds: "int | None" = 300, max_chunks: "int | None" = None):
        """Collect up to `n_target` support-projection bit-rows efficiently. Only the small
        (n, S) bit matrix crosses to host RAM per chunk (not full N-var models), so a big output
        is cheap. Returns (matrix, cols_1based): matrix is a numpy uint8 (got, S) array of 0/1,
        cols_1based lists the ORIGINAL variable id of each column. Sound: every row is a valid
        support projection (forced pinned, compacted vars from a real model, don't-cares free,
        ELS-equivalent vars kept equal via shared random groups)."""
        rv, B, la = config[0], max(1, config[1]), (config[2] if len(config) > 2 else False)
        plan = self._build_projection_plan()
        support_mask = None
        if self.support:
            support_mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
            support_mask[torch.tensor(self.support, dtype=torch.long, device=self.device)] = True
        # Every chunk is fused; run at the boosted K_fused while K is at its construction value
        # (a caller-lowered K — e.g. the coverage run — is honoured).
        K = self.K_fused if self.K >= self._K_init else self.K
        mats: list = []
        got = chunk = 0
        while got < n_target:
            if max_chunks is not None and chunk >= max_chunks:
                break
            chunk += 1
            n_done, _, _, _ = self._run_chunk(K, B, rv, max_decision_rounds,
                                              support_mask, True, la)
            if n_done == 0:
                if chunk >= 3:                    # cold start + grace: instance is producing nothing
                    break
                continue
            take = min(n_done, n_target - got)
            mats.append(self._project(n_done)[:take])
            got += take
        M = (np.concatenate(mats) if mats else np.zeros((0, plan["S"]), dtype=np.uint8))
        return M, plan["cols_1based"]

    # ----------------------------------------------------------------- public

    @torch.inference_mode()
    def sample(
        self,
        n_target: int,
        *,
        decisions_per_round: int = 1,
        random_var: "bool | str" = True,
        max_decision_rounds: int | None = None,
        max_chunks: int | None = None,
        collect: bool = True,
        verify_clauses: list | None = None,
        support: list | None = "auto",
        complete: bool = True,
        lookahead: bool = False,
        configs: "list[tuple[bool, int]] | None" = None,
        config_policy: str = "adaptive",
        min_measure_chunks: int = 1,
        max_seconds: "float | None" = None,
    ) -> dict:
        """Draw at least `n_target` valid models.

        Stops early on any of the safety valves (all optional):
          - `max_chunks` chunks have run;
          - `max_seconds` of sampling wall have elapsed (checked between chunks — the intended
            "produce N samples, but never spin longer than T" contract);
          - ZERO models after every config has been measured plus a couple of grace chunks —
            the target is unreachable (e.g. a 0%-success instance would otherwise loop forever
            once no chunk cap is given).

        `support`: 0-based variable indices to restrict decisions to (independent-support
        sampling — the rest are determined by propagation). "auto" (default) uses the
        support parsed from the CNF (`self.support`) if any; None decides all variables.

        n_models counts FULLY-assigned models (sound — guaranteed valid). When sampling over
        an independent support, models are projected onto the support for diversity/collection.
        n_support_complete counts trajectories that fixed the whole support without conflict
        (equals n_models when the support is a true independent support). If `verify_clauses`
        is given, every fully-assigned model is independently checked and `all_valid` is set.

        `random_var`: True = random decision order (diverse), False = structured/index order (follows
        the gate DAG — completes search-heavy residuals), "auto" = cheaply probe a (order, B) grid and
        pick the best-throughput combo per formula (recommended; overrides `decisions_per_round` with
        the tuned B — cracks the service family with (structured, B=1) without hurting the niche's B=8).

        `configs`: a PORTFOLIO of (random_var, B) tuples, with `config_policy`:
          - "adaptive" (default): a throughput BANDIT — each chunk runs ONE config at full K, picked by
            its measured models/wall so far, dynamically switching to the winner and re-checking the
            rest (decaying-epsilon). Robust like a portfolio yet ~as fast as the best single config; it
            fixes the autotune's small-K mis-estimate (it measures full-K) and its permanent commit
            (it can switch). Recommended.
          - "static": run ALL configs as sub-batches every chunk (K split) — guarantees per-chunk
            coverage but a slow config's K-independent round count drags every chunk (~1.5-2.7× niche).
        `config_breakdown` reports each config's models/chunks/throughput. Overrides
        random_var/decisions_per_round when set.
        """
        B = max(1, decisions_per_round)
        sup = self.support if (isinstance(support, str) and support == "auto") else support
        support_mask = None
        sup_idx = None
        if sup:
            support_mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
            sup_t = torch.tensor(sup, dtype=torch.long, device=self.device)
            support_mask[sup_t] = True
            sup_idx = sup_t                                 # for projection
        if configs is None and isinstance(random_var, str) and random_var == "auto":
            random_var, B = self._autotune(self.K, B, max_decision_rounds, support_mask,
                                           complete, lookahead)
        # Config portfolio. Two policies (when len>1):
        #  - "adaptive" (default): a THROUGHPUT BANDIT — each chunk runs ONE config at full K, chosen by
        #    its measured models/wall so far, dynamically switching to the winner (and re-checking the
        #    rest). Robust like a portfolio but ~as fast as the best single config — fixes both the
        #    autotune's small-K mis-estimate and its permanent commit.
        #  - "static": run ALL configs as sub-batches every chunk (K split). Guarantees per-chunk
        #    coverage but a slow config's K-independent round count drags every chunk (~1.5-2.7×).
        # Normalise the portfolio to (random_var, B, lookahead) triples. A config may be a 2-tuple
        # (rv, B) — lookahead defaults off — or a 3-tuple (rv, B, lookahead), lookahead ∈
        # {False, "soft"} ("fused"/"fsoft" accepted as aliases from the pre-packed-only era).
        # "soft" = the in-kernel risk-polarity — the big win on the tight/deep CIRCUIT regime
        # (conflict-avoidance polarity → far higher success at no extra memory). The retired
        # modes raise: "full" python lookahead and "orac" needed the deleted PyTorch engine
        # path (both measured out — see docs/findings/INSIGHTS.md).
        def _norm(c):
            la = c[2] if len(c) > 2 else False
            if la in ("full", "orac", True):
                raise ValueError(f"lookahead mode {la!r} was retired with the PyTorch engine "
                                 "path; use False or 'soft'")
            return (c[0], max(1, c[1]), "soft" if la in ("soft", "fsoft") else False)
        plan = [_norm(c) for c in configs] if configs else [_norm((random_var, B, lookahead))]
        # NOTE: the packed K_fused keeps the conservative eng_hold deduction (__init__) even
        # though no base-K engine state exists any more — that headroom covers the fixed 4096×M
        # soundness-check temporary (~3.3 GB on b05-c50's M=201k); skipping it SPILLED the big
        # c50 wide instances under the kill-switch (11.7 GB on the 12 GB 4080). If the ~9% is
        # ever wanted back, it must go with an explicit 4096·M reservation + kill-switch re-run.
        K = self.K
        adaptive = configs is not None and config_policy == "adaptive" and len(plan) > 1
        # Per-config work counters (decs/props) alongside models/wall so the reported dec/s, impl/s,
        # assigns/s can be attributed to the SAME config as the headline samples/s (best_rate) —
        # otherwise they blend over the bandit's exploration (incl. the slow orac chunk), understating
        # the winner ~2x and diverging across GPUs (measured on b06.c25, 2026-07-04).
        cfg_stats = {c: {"models": 0, "attempts": 0, "wall": 0.0, "chunks": 0,
                         "decs": 0, "props": 0} for c in plan}
        K_sub = max(1, K // len(plan)) if (configs and not adaptive and len(plan) > 1) else K
        models: list[list[int]] = []
        attempts = 0
        n_models = 0
        n_support_complete = 0
        rounds_hist: list[int] = []
        free_after_hist: list[float] = []
        chunk = 0
        all_valid = True
        gpu_unsound = 0          # rows that were 'fully+active' yet a clause had sat_count==0 (should be 0)
        distinct: set[tuple[int, ...]] = set()
        keep_rows = collect or verify_clauses is not None

        self._sync()
        bcp_r0 = self.engine.launches
        bcp_o0 = self.engine.ops_total()
        bcp_p0 = self.engine.props_total()
        bcp_d0 = self.engine.decs_total()
        t0 = time.perf_counter()
        while True:
            # Stop once n_target is met — BUT for the adaptive bandit, always measure every config at
            # least once first. Otherwise on a big-K GPU a single chunk can already exceed n_target, so
            # only plan[0] ever runs and a faster config is never discovered (the A100 shallow-circuit
            # regression). Bounded by max_chunks. Cold-start chunks are productive (real models).
            cold = adaptive and any(cfg_stats[c]["chunks"] == 0 for c in plan)
            # Stability: don't stop until SOME config has been measured over >= min_measure_chunks
            # chunks (the bandit piles those onto the winner). A big-K GPU meets n_target in the
            # cold-start (1 chunk/config), so a single noisy chunk would set the headline rate —
            # measured on b06.c25: best_rate_sps swung 131k-249k (1.9x) run-to-run at 1 chunk.
            enough = (not adaptive) or max((cfg_stats[c]["chunks"] for c in plan),
                                           default=0) >= min_measure_chunks
            if not cold and enough and n_models >= n_target:
                break
            if max_chunks is not None and chunk >= max_chunks:
                break
            if max_seconds is not None and time.perf_counter() - t0 > max_seconds:
                break                                   # hard time cap (between chunks)
            if n_models == 0 and not cold and chunk >= len(plan) + 2:
                break                                   # every config measured + grace: hopeless
            chunk += 1
            # Once past n_target but short of the measurement window, EXPLOIT the winner (no eps
            # exploration) so its stable rate is measured on back-to-back chunks, not diluted.
            topup = adaptive and not cold and n_models >= n_target
            chunk_cfgs = ([self._select_config(plan, cfg_stats, chunk, exploit=topup)]
                          if adaptive else plan)
            for cfg in chunk_cfgs:
                rv_c, B_c, la_c = cfg
                # Fused-trajectory chunks (fused AND fsoft — both are single-launch kernels with no
                # (K,N) decide masks — see K_fused) run at the boosted K. Only while K is at its
                # construction value: a caller-lowered K (coverage run) is honoured.
                # every chunk is fused: run at the boosted K_fused while K is at its
                # construction value (a caller-lowered K — coverage run — is honoured)
                K_c = self.K_fused if self.K >= self._K_init else K_sub
                if adaptive:
                    self._sync()
                    # read BEFORE t_cfg so the (synced) counter reads aren't charged to wall
                    p_before = self.engine.props_total()
                    d_before = self.engine.decs_total()
                    t_cfg = time.perf_counter()
                n_done, n_sup, rounds, free_after = self._run_chunk(
                    K_c, B_c, rv_c, max_decision_rounds, support_mask, complete, la_c)
                rounds_hist.append(rounds)
                free_after_hist.append(free_after)
                attempts += K_c
                n_models += n_done
                n_support_complete += n_sup
                cfg_stats[cfg]["models"] += n_done
                cfg_stats[cfg]["attempts"] += K_c
                cfg_stats[cfg]["chunks"] += 1
                if adaptive:
                    self._sync(); cfg_stats[cfg]["wall"] += time.perf_counter() - t_cfg
                    cfg_stats[cfg]["props"] += self.engine.props_total() - p_before
                    cfg_stats[cfg]["decs"] += self.engine.decs_total() - d_before
                if n_done:
                    # GPU soundness (cheap): for a FULLY-assigned row, non-falsified ⟺ TRUE, so
                    # "every clause has sat_count > 0" ⟺ "every clause has a TRUE literal" ⟺ satisfied.
                    # (Holds ONLY at a complete assignment — sat_count>0 on a PARTIAL assignment just
                    # means "not yet violated". We gate on the completed selection, so it's valid.)
                    # Redundant with `active` (cleared the instant a clause hits 0); CPU
                    # `_satisfies()` is the real check. Evaluated entirely on device (one flag per
                    # row): the old path shipped a (rows, M) int32 slice to the host, which on
                    # b05-c50 reached ~3.3 GB and forced the K budget to keep silent headroom.
                    gpu_unsound += self.engine.check_sound(slot=0, max_rows=4096)
                if n_done and keep_rows:
                    # Gather just the variables' TRUE bits — (n, N), half of what the full
                    # (n, 2N) assignment would cost, and no polarity slice on the host.
                    if getattr(self, "_all_vars", None) is None:
                        self._all_vars = np.arange(self.N, dtype=np.int32)
                    pos = self.engine.var_bits(0, self._all_vars, n_done)   # (n_done, N) uint8
                    vidx = np.arange(1, self.N + 1, dtype=np.int32)
                    comp_rows = np.where(pos, vidx, -vidx).tolist()
                    if self._recon is not None and not self._recon.get("unsat"):
                        # reconstruct full ORIGINAL models (forced pinned, don't-cares randomized)
                        out_rows, projs = self._reconstruct(comp_rows)
                    else:
                        out_rows = comp_rows
                        si = sup_idx.tolist() if sup_idx is not None else None
                        projs = [tuple(r[i] for i in si) for r in comp_rows] if si is not None else comp_rows
                    for p in projs:
                        distinct.add(tuple(p))
                    if verify_clauses is not None:
                        for r in out_rows:
                            if not _satisfies(r, verify_clauses):
                                all_valid = False
                    if collect:
                        models.extend(out_rows)
        self._sync()
        elapsed = time.perf_counter() - t0
        bcp_rounds = self.engine.launches - bcp_r0             # kernel launches over this run (delta)
        bcp_ops = self.engine.ops_total() - bcp_o0
        bcp_props = self.engine.props_total() - bcp_p0
        bcp_decs = self.engine.decs_total() - bcp_d0
        bcp_impl = bcp_props - bcp_decs                        # every non-decision assignment

        out = {
            "n_models": n_models, "n_support_complete": n_support_complete, "attempts": attempts,
            "success_rate": (n_models / attempts) if attempts else 0.0,
            "support_success_rate": (n_support_complete / attempts) if attempts else 0.0,
            "elapsed_s": elapsed,
            "samples_per_s": (n_models / elapsed) if elapsed > 0 else 0.0,
            "attempts_per_s": (attempts / elapsed) if elapsed > 0 else 0.0,
            # Lightweight BCP-work stats (free counters) — to tell success-bound from BCP-bound: a low
            # samples/s with high attempts/s = low success; a low attempts/s = each trajectory is slow
            # (deep propagation). ops/attempt = clause-cells touched per trajectory.
            "bcp_rounds": bcp_rounds, "bcp_ops": bcp_ops,
            "bcp_rounds_per_chunk": (bcp_rounds / chunk) if chunk else 0.0,
            "bcp_ops_per_attempt": (bcp_ops / attempts) if attempts else 0.0,
            # CDCL-comparable work rate: literals assigned (decisions + implications) per second —
            # what CDCL solvers report as props/sec (typically ~1e7/s single-core). Split into the
            # branch DECISIONS and the unit-propagation IMPLICATIONS they triggered (impl = assigns
            # - decisions); implications/decision is the propagation fan-out per branch.
            "assigns_per_s": (bcp_props / elapsed) if elapsed > 0 else 0.0,
            "decisions_per_s": (bcp_decs / elapsed) if elapsed > 0 else 0.0,
            "implications_per_s": (bcp_impl / elapsed) if elapsed > 0 else 0.0,
            "bcp_decisions": bcp_decs, "bcp_implications": bcp_impl,
            "mean_free_after_support": (sum(free_after_hist) / len(free_after_hist)) if free_after_hist else 0.0,
            "n_chunks": chunk, "K": K, "B": B, "random_var": bool(random_var),
            "support_size": (len(sup) if sup else None),
            "mean_decision_rounds": (sum(rounds_hist) / len(rounds_hist)) if rounds_hist else 0,
            "max_decision_rounds": max(rounds_hist) if rounds_hist else 0,
            "n_distinct": len(distinct) if keep_rows else None,
            "gpu_sound": gpu_unsound == 0, "gpu_unsound": gpu_unsound,
            "kernel": True,
        }
        if collect:
            out["models"] = models
        if verify_clauses is not None:
            out["all_valid"] = all_valid
        if configs is not None:
            def _rate(s):
                return (s["models"] / s["wall"]) if s["wall"] > 0 else 0.0
            dom = max(cfg_stats.items(), key=lambda kv: kv[1]["models"])[0]   # most models produced
            out["random_var"], out["B"], out["lookahead"] = bool(dom[0]), dom[1], dom[2]
            out["dominant_config"] = [bool(dom[0]), dom[1], dom[2]]
            # Best-THROUGHPUT config = max models/wall (the one a real run commits to after exploring).
            # The benchmark headline uses THIS rate, not the end-to-end blend, so the bandit's slower
            # exploration chunks (and any first-chunk warmup) never drag the reported steady-state
            # throughput. Only meaningful when adaptive (per-config wall is timed); else wall==0.
            # Prefer a WELL-MEASURED config (>= min_measure_chunks) over a lucky 1-chunk rate, then by
            # rate — so the headline rests on the config the top-up phase actually committed to and
            # measured, not a config whose single cold-start chunk happened to clock fast.
            brc, brs = max(cfg_stats.items(),
                           key=lambda kv: (kv[1]["chunks"] >= min_measure_chunks, _rate(kv[1])))
            out["best_rate_config"] = [bool(brc[0]), brc[1], brc[2]]
            out["best_rate_sps"] = _rate(brs) if brs["wall"] > 0 else None
            out["best_rate_success"] = (brs["models"] / brs["attempts"]) if brs["attempts"] else 0.0
            # Work rates for the SAME best-rate config as best_rate_sps (consistent + stable: measured
            # over the winner's chunks, not blended over the exploration). gpap_sample reports these.
            bw = brs["wall"]
            out["best_rate_decisions_per_s"] = (brs["decs"] / bw) if bw > 0 else None
            out["best_rate_implications_per_s"] = ((brs["props"] - brs["decs"]) / bw) if bw > 0 else None
            out["best_rate_assigns_per_s"] = (brs["props"] / bw) if bw > 0 else None
            out["best_rate_chunks"] = brs["chunks"]
            out["config_policy"] = "adaptive" if adaptive else "static"

            def _lab(c):
                return f"rv={c[0]},B={c[1]}" + (f",la={c[2]}" if c[2] else "")
            out["config_breakdown"] = {
                _lab(c): {
                    "models": s["models"], "attempts": s["attempts"], "chunks": s["chunks"],
                    "wall_s": s["wall"],
                    "success_rate": (s["models"] / s["attempts"]) if s["attempts"] else 0.0,
                    "samples_per_s": (s["models"] / s["wall"]) if s["wall"] > 0 else None,
                    "decisions_per_s": (s["decs"] / s["wall"]) if s["wall"] > 0 else None,
                    "implications_per_s": ((s["props"] - s["decs"]) / s["wall"]) if s["wall"] > 0 else None,
                } for c, s in cfg_stats.items()
            }
        return out


def _satisfies(dimacs_model: list[int], clauses: list) -> bool:
    """Independent soundness check: does this full assignment satisfy every clause?"""
    true_set = set(dimacs_model)
    for c in clauses:
        if not any(l in true_set for l in c):
            return False
    return True


def simplify_cnf(nv: int, clauses: list, support: list | None):
    """Unit-propagation simplification before sampling (what Arjun/preprocessors do). Asserts all
    unit clauses, propagates, drops satisfied clauses + falsified literals, and COMPACTS the
    remaining (unassigned) variables to 1..nv'. This shrinks M/N a lot on unit-heavy formulas
    (e.g. these sk_ benchmarks), so the sampler fits a much larger K → higher throughput.

    Returns (nv2, clauses2, support2, old_of_new, forced, free_support) or None if UNSAT:
      nv2          : compacted variable count.
      clauses2     : simplified clauses over compacted vars (1-based).
      support2     : 0-based support vars surviving in the compacted formula (decide these).
      old_of_new   : list; old_of_new[i] = original DIMACS var of compacted var i+1.
      forced       : DIMACS lits fixed by unit propagation (constant in EVERY full model).
      free_support : 0-based original support vars that are neither forced nor in any remaining
                     clause — true don't-cares; to preserve the support-projection distribution
                     they MUST be sampled uniformly at random in reconstruction (not dropped).

    Soundness: unit propagation only asserts already-forced units and removes satisfied clauses /
    falsified literals — it is strictly MODEL-PRESERVING (it does NOT eliminate variables by
    resolution like VE/BCE, which CMSGen disables for exactly that reason). So the set of valid
    support projections is unchanged, provided forced support vars are pinned to their forced value
    and free_support vars are randomized. Validated by tests/test_sampler.py.
    """
    from itertools import chain
    from gpap.sampling.preprocess import unit_closure
    out, assigned, status = unit_closure(clauses, [])   # reads only — no defensive copy needed
    if status == "unsat":
        return None
    nonunit = [c for c in out if len(c) > 1]
    # Vectorised variable compaction (the sorted-set + per-literal remap listcomps cost ~1.5s
    # of pure Python per call on 1M-clause unrolls; this runs on construct AND rebuild).
    lens = np.fromiter(map(len, nonunit), dtype=np.int64, count=len(nonunit))
    flat = (np.array(list(chain.from_iterable(nonunit)), dtype=np.int64) if lens.sum()
            else np.empty(0, dtype=np.int64))
    used_np = np.unique(np.abs(flat))                   # sorted unique surviving vars
    lut = np.zeros((int(used_np[-1]) + 1) if len(used_np) else 1, dtype=np.int64)
    lut[used_np] = np.arange(1, len(used_np) + 1)       # old var -> compacted var (0 = unused)
    nf = (np.sign(flat) * lut[np.abs(flat)]).tolist()
    bounds = np.zeros(len(nonunit) + 1, dtype=np.int64)
    np.cumsum(lens, out=bounds[1:])
    clauses2 = [nf[a:b] for a, b in zip(bounds[:-1].tolist(), bounds[1:].tolist())]

    def _new(v1):                                       # 1-based old var -> compacted var or 0
        return int(lut[v1]) if v1 < len(lut) else 0

    support2 = sorted({_new(v + 1) - 1 for v in (support or []) if _new(v + 1)})
    forced = [(v if assigned[v] else -v) for v in assigned]
    free_support = sorted(v for v in (support or [])
                          if (v + 1) not in assigned and not _new(v + 1))
    return len(used_np), clauses2, support2, used_np.tolist(), forced, free_support


def simplify_cnf_els(nv: int, clauses: list, support: list | None):
    """Stronger simplification: unit-closure + EQUIVALENT-LITERAL SUBSTITUTION (ELS) + compaction.

    On top of `simplify_cnf`'s unit propagation, this collapses mutually-implied literals (binary-
    implication SCCs) to a single representative, REMOVING variables. Since BCP propagation cost
    scales with M (clauses) and K with per-trajectory memory (~M), shrinking the formula speeds up
    the dominant propagate AND lets a larger K fit — it hits the profiled bottleneck twice. On the
    sk_ benchmarks the binary-graph SCCs alone remove 5-38% of variables / 3-30% of clauses.

    ELS is equisatisfiability-preserving and, for PROJECTED sampling, support-projection preserving
    **iff reconstruction is consistent**: every removed variable equals its representative literal
    in EVERY model, so a removed var (even a don't-care one) must be reconstructed from its rep, NOT
    randomized independently — otherwise an equivalent pair could be assigned unequal values, an
    invalid projection. We therefore reconstruct per equivalence-CLASS (one value per representative,
    applied to all members with the right polarity). Validated by tests/test_sampler.py.

    Returns (nv_r, clauses_r, support_r, recon) or None if UNSAT, where recon carries the per-original
    -variable resolver used by GPUSampler._reconstruct:
      rep_of[v-1]  : (rep_var, same_polarity) — original var v takes rep_var's value (negated if not).
      new_of_new   : {rep_var: compacted_idx(1-based)} — reps that survive as decided/propagated vars.
      forced_map   : {var: bool} — vars pinned by either unit-closure pass.
    """
    from gpap.engine.formula import ext_lit_to_lit_index
    from gpap.sampling.preprocess import equivalent_literal_substitution, unit_closure

    # Pass 1: unit closure over the original variables (reads only — no copy).
    out1, assigned1, st1 = unit_closure(clauses, [])
    if st1 == "unsat":
        return None
    nonunit1 = [c for c in out1 if len(c) > 1]

    # Binary-implication edges (2N lit-index scheme) from binary clauses: (l1∨l2) ⇒ ¬l1→l2, ¬l2→l1.
    edges = []
    for c in nonunit1:
        if len(c) == 2:
            i1, i2 = ext_lit_to_lit_index(c[0]), ext_lit_to_lit_index(c[1])
            edges.append((i1 ^ 1, i2))
            edges.append((i2 ^ 1, i1))
    if edges:
        els_clauses, var_map, st = equivalent_literal_substitution(nonunit1, edges, nv)
        if st.get("status") == "unsat":
            return None
    else:
        els_clauses, var_map = nonunit1, {}

    # Pass 2: ELS substitution can turn (a∨a)→(a) into new units — absorb them.
    out2, assigned2, st2 = unit_closure(els_clauses, [])
    if st2 == "unsat":
        return None
    nonunit2 = [c for c in out2 if len(c) > 1]

    used = sorted({abs(l) for c in nonunit2 for l in c})       # surviving representative vars
    new_of_new = {v: i + 1 for i, v in enumerate(used)}
    clauses_r = [[(new_of_new[abs(l)] if l > 0 else -new_of_new[abs(l)]) for l in c]
                 for c in nonunit2]
    forced_map = {**assigned1, **assigned2}                    # var -> bool (disjoint key sets)

    # Per-original-variable resolver: v -> (rep_var, same_polarity).
    rep_of = []
    for v in range(1, nv + 1):
        if v in var_map:                                       # removed by ELS -> rep DIMACS literal
            rl = var_map[v]
            rep_of.append((abs(rl), rl > 0))
        else:
            rep_of.append((v, True))                           # its own rep (forced / model / free)

    def _rep(v0):                                              # surviving decided var for support v0
        v = v0 + 1
        if v in forced_map:
            return None
        rv = rep_of[v - 1][0]
        return (new_of_new[rv] - 1) if rv in new_of_new else None

    support_r = sorted({_rep(v0) for v0 in (support or []) if _rep(v0) is not None})
    recon = {"orig_nv": nv, "orig_support": sorted(support) if support else [],
             "rep_of": rep_of, "new_of_new": new_of_new, "forced_map": forced_map}
    return len(used), clauses_r, support_r, recon


def parse_independent_support(path: str) -> list[int] | None:
    """Read the independent support from `c ind <v1> <v2> ... 0` header lines (UniGen/QuickSampler
    convention). Returns 0-based variable indices, or None if no `c ind` lines are present."""
    sup: set[int] = set()
    found = False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("c ind") or s.startswith("cr "):
                found = True
                for tok in s.split()[2:] if s.startswith("c ind") else s.split()[1:]:
                    try:
                        v = int(tok)
                    except ValueError:
                        continue
                    if v != 0:
                        sup.add(abs(v) - 1)            # DIMACS 1-based → 0-based var
            elif s[0].isdigit() or s[0] == "-":
                break                                  # past the header into clauses
    return sorted(sup) if found else None
