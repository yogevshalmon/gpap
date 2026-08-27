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

K is auto-sized to the GPU (capped by default to stay off the host-RAM spill cliff).
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is what makes that K reachable; it is set in
`gpap/__init__.py`, which always runs before this module (see there for the measurement).
"""
import contextlib
import time

import numpy as np
import torch

from gpap.engine.formula import CnfFormula
from gpap.engine.bcp_engine import BCPEngine
from gpap.engine.calibrate import (
    compute_k_safe,
    dispatch_mode,
    gpu_mem_available,
    resolve_k_cap,
)

# Decision batch the trail is sized for. The trail cap is num_vars + max(B, 64) + 8, so any B
# at or below the floor costs the same; sizing with the floor makes K independent of the
# portfolio's B, which a measurement wants.
SOUND_B_FLOOR = 64


class GPUSampler:
    """Batched GPU sampler: K parallel randomized BCP-completions per chunk."""

    def __init__(
        self,
        formula: CnfFormula,
        device: torch.device,
        *,
        k_cap: "int | str | None" = "auto",
        seed: int | None = None,
    ):
        """`k_cap` bounds the per-chunk batch. THREE values, and the default is the useful one:

            "auto"  (default) size the cap to THIS GPU's VRAM (calibrate.default_k_cap) — what a
                    user who never wants to think about K should get, and what the pipeline and
                    the CLI have always passed. Exactly the historical 49,152 at or below the
                    reference card, larger on a bigger one.
            int     that cap, verbatim — what a cross-GPU A/B needs, since "auto" differs per card.
            None    no cap: the analytic memory model alone decides.

        The default used to be a hardcoded 16,384, which is neither of the three: it pinned a
        direct GPUSampler user to ~1/3 of the batch the same formula gets through the pipeline,
        on every GPU. No in-repo caller relied on it (all of them pass k_cap explicitly)."""
        self.formula = formula
        self.device = device
        self.k_cap = k_cap = resolve_k_cap(k_cap, device)   # resolved once; None = uncapped
        self.N = formula.num_vars
        self.M = formula.num_clauses
        # The engine core (the ONLY execution path — raises off-CUDA / without libgpap;
        # field split 16|16 or 8|24 picked per formula, see gpap.engine).
        self.engine = BCPEngine(formula)
        # K is charged per STATE clause (length >= 3): binaries are occ implication edges and
        # units live in the flush bitmap — neither costs per-row bytes (see gpap_row_bytes).
        self.M_state = self.engine.num_long
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
        # THE batch size, from the engine's own memory model (gpap.engine.calibrate ->
        # cpp/src/calibrate.cpp). One number, one model: `row_bytes` is the exact allocation list
        # of what a chunk reserves, so there is nothing left to "boost".
        #
        # There used to be TWO models here — a conservative `K` plus a `K_fused` boost that
        # recovered the headroom the conservative one wasted. Both are gone because the reason
        # for them is: the old per-row model charged for a separate sat array, a byte-per-literal
        # assignment and PyTorch-era scatter temporaries that no longer exist, while omitting the
        # trail, which made it 1.57x over. The boost was a correction factor for a wrong model.
        self.K = compute_k_safe(self.M_state, self.N, gpu_mem_available(device), cap=k_cap,
                                decisions_per_round=SOUND_B_FLOOR)
        self._K_init = self.K
        # PIN the propagate kernel to the batch the chunks will actually run, rather than letting
        # the engine re-dispatch per reset(). The two agree on every full chunk, but the sampler
        # also issues small resets — the warmup's pinned 1024, a short final chunk — and those
        # must not silently switch kernel mid-run: `engine.mode` is recorded as row provenance and
        # a measurement whose kernel changes underneath it is not a measurement. The RULE lives in
        # engine.calibrate.dispatch_mode; this only chooses its input. `auto_dispatch` guards it:
        # a kernel pinned any other way ($GPAP_KERNEL, an explicit mode) must not be overridden.
        if self.engine.auto_dispatch and device.type == "cuda":
            self.engine.set_mode(dispatch_mode(self.M, self.K, device))

    # ---------------------------------------------------------------- THE batch size
    # Read `batch_size`; set it with pin_batch() / pinned_batch(). `K` is the calibrated value
    # and `batch_size` is what the next chunk runs — they differ only while a pin is in force.
    @property
    def batch_size(self) -> int:
        """Trajectories the next chunk will run — the number that sets throughput."""
        return self.K

    def pin_batch(self, n: int) -> int:
        """Force every chunk to EXACTLY `n` rows, and return it.

        Use it whenever a bound is REQUIRED rather than merely intended — above all when rows are
        reconstructed to host RAM, where an unbounded chunk is tens of GB (that is exactly how
        tests/test_circuits.py once reached 28 GB asking for 1000 models).

        Pinning ABOVE the calibrated value is allowed and is how the no-change gate fixes K across
        builds, but it overrides the memory model. On platforms that SPILL to host RAM instead
        of failing the allocation, an oversized K degrades silently rather than erroring, so size
        it yourself (docs/BENCHMARKING.md §2, kill-switch)."""
        self.K = max(1, int(n))
        return self.K

    @contextlib.contextmanager
    def pinned_batch(self, n: int):
        """`pin_batch(n)` for the duration of the block, restoring the calibrated value after."""
        k = self.K
        self.pin_batch(n)
        try:
            yield self
        finally:
            self.K = k

    # Byte cap for one (rows, M) int32 soundness-check slice (see sample()); bounds the
    # transient regardless of M so the K budget needs no silent headroom for it.
    _SOUND_SLICE_BYTES = 256e6

    @classmethod
    def from_cnf_file(cls, path: str, device: torch.device, *, simplify: bool = False,
                      els: bool = False, **kw) -> "GPUSampler":
        """Build a sampler from a DIMACS file. `simplify=True` first applies unit-propagation
        simplification (shrinks M -> larger K -> more throughput) and samples the compacted
        formula, reconstructing full original models at collection time (see simplify_cnf).
        `els=True` additionally does equivalent-literal substitution (simplify_cnf_els): removes
        more variables/clauses -> faster propagate + bigger K (implies simplify)."""
        if not (simplify or els):
            formula = CnfFormula(device)
            formula.parse_cnf_file(path)            # ONE parse; support comes off the same core
            smp = cls(formula, device, **kw)
            smp.support = formula.support
            return smp
        from gpap.engine import read_dimacs
        nv, clauses, support = read_dimacs(path)
        return cls._from_simplified(nv, clauses, support, device, els=els, **kw)

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
                return cls._unsat_sampler(device, nv, support, **kw)
            nv2, cl2, sup2, recon = res
            formula = CnfFormula(device)
            formula.init_formula(nv2, cl2)
            smp = cls(formula, device, **kw)
            smp.support = sup2
            smp._recon = {"unsat": False, "els": True, **recon}
            return smp
        res = simplify_cnf(nv, clauses, support)
        if res is None:                                  # UNSAT after unit propagation
            return cls._unsat_sampler(device, nv, support, **kw)
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
    def _unsat_sampler(cls, device: torch.device, orig_nv: int = 1,
                       orig_support: list | None = None, **kw) -> "GPUSampler":
        """Stand-in sampler for a formula simplify proved UNSAT: emits nothing, forever.

        It still records the ORIGINAL shape (nv + support). No row will ever be projected, but a
        caller asking for bit output needs the COLUMN LIST to hand back a well-formed (0, S)
        matrix — without it `_build_projection_plan` had no `orig_nv` to read and `gpap sample
        unsat.cnf --out f.bits` died with a KeyError instead of writing an empty file."""
        formula = CnfFormula(device)
        formula.init_formula(1, [[1], [-1]])
        smp = cls(formula, device, **kw)
        smp.support = []
        smp._recon = {"unsat": True, "orig_nv": orig_nv,
                      "orig_support": sorted(orig_support) if orig_support else []}
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
    def _run_chunk(self, K: int, B: int, random_var: bool,
                   support_mask: torch.Tensor | None, complete: bool, lookahead=False):
        """One chunk of K trajectories — the WHOLE decide+propagate loop in ONE fused kernel
        launch (engine.run_trajectory). Decision order = support vars first (phase 1), then, if
        `complete`, the remaining vars (phase 2); the kernel never straddles the phase boundary.
        `lookahead` in ("soft", "fsoft") enables the in-kernel risk-polarity. There is
        deliberately no round cap: the kernel structurally terminates in <= |order| decision
        groups, and a cap cripples B=1 on large supports (measured 2x on s832).

        Returns (n_fully, n_support_complete, rounds):
          n_fully : rows that are active AND have no free var -> guaranteed models.
                    The row selection stays on the GPU in engine slot 0 (slot 1 holds
                    the support-complete selection); read it out with the engine's
                    selection accessors, no (K,) mask ever crosses the bus.
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
        return n_fully, n_sup, rounds

    def _select_config(self, plan, stats, chunk, exploit=False):
        """Adaptive (bandit) config selection: dynamically switch to the best-THROUGHPUT config as it
        is measured on real full-K chunks. More robust than a small-K pre-probe (which
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

    def projection_cols(self) -> list[int]:
        """The 0-based ORIGINAL variables each output column stands for — the SHAPE of this
        sampler's output, without building the plan that fills it. Split out so a caller that
        collected nothing can still report a correctly-shaped (0, S) result: the plan itself is a
        per-column Python pass (~0.2 us/column) and there is no reason to pay it for zero rows."""
        r = self._recon
        if r is not None and r.get("orig_support"):
            return list(r["orig_support"])                                 # 0-based support vars
        if r is not None:
            return list(range(r["orig_nv"]))                               # full model
        if self.support:
            return list(self.support)
        return list(range(self.N))

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
        cols = self.projection_cols()
        S = len(cols)
        gcol, flip, const, rgroup = [-1] * S, [0] * S, [0] * S, [-1] * S
        groups: dict = {}

        def _group(key):
            return groups.setdefault(key, len(groups))

        if r is not None and r.get("unsat"):
            pass                                               # no row will ever be projected;
            # the plan carries only the SHAPE, so an UNSAT formula yields a (0, S) matrix.
        elif r is None:                                        # sampler decided ORIGINAL vars
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
    def sample_bits(self, n_target: int, *, config=(True, 8, False),
                    max_chunks: "int | None" = None):
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
        K = self.batch_size
        mats: list = []
        got = chunk = 0
        while got < n_target:
            if max_chunks is not None and chunk >= max_chunks:
                break
            chunk += 1
            n_done, _, _ = self._run_chunk(K, B, rv, support_mask, True, la)
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
        random_var: bool = True,
        max_chunks: int | None = None,
        collect: bool = True,
        verify_clauses: list | None = None,
        support: list | None = "auto",
        complete: bool = True,
        lookahead: bool = False,
        configs: "list[tuple[bool, int]] | None" = None,
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

        `random_var`: True = random decision order (diverse), False = structured/index order
        (follows the gate DAG — completes search-heavy residuals). Per-formula adaptation is the
        portfolio bandit's job (`configs`), which measures both orders on real full-K chunks.

        `configs`: a PORTFOLIO of (random_var, B) tuples, run as a throughput BANDIT — each chunk
        runs ONE config at full K, picked by its measured models/wall so far, dynamically switching
        to the winner and re-checking the rest (decaying-epsilon). Robust like a portfolio yet ~as
        fast as the best single config; unlike the retired small-K pre-probe it measures at
        full K, and it can change its mind. A one-entry portfolio leaves it nothing to select, which
        is how a MEASUREMENT pins the algorithm (see pipeline.SINGLE_PORTFOLIOS).
        `config_breakdown` reports each config's models/chunks/throughput. Overrides
        random_var/decisions_per_round when set.
        """
        if isinstance(random_var, str):
            raise ValueError(f"random_var={random_var!r} is not a mode: pass a bool. Per-formula "
                             "adaptation is the portfolio bandit (configs=...); the old 'auto' "
                             "probe was retired — the bandit measures at full K and can change "
                             "its mind, which are exactly what the probe could not do.")
        B = max(1, decisions_per_round)
        sup = self.support if (isinstance(support, str) and support == "auto") else support
        support_mask = None
        sup_idx = None
        if sup:
            support_mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
            sup_t = torch.tensor(sup, dtype=torch.long, device=self.device)
            support_mask[sup_t] = True
            sup_idx = sup_t                                 # for projection
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
        # The 4096-row soundness check reads sat_count in bounded slices (_SOUND_SLICE_BYTES),
        # so it needs no reservation in the batch budget.
        K = self.batch_size
        # More than one config => the bandit picks per chunk. Exactly one => nothing to pick, and
        # per-config timing is skipped (that is the pinned-algorithm measurement mode).
        adaptive = configs is not None and len(plan) > 1
        # Per-config work counters (decs/props) alongside models/wall so the reported dec/s, impl/s,
        # assigns/s can be attributed to the SAME config as the headline samples/s (best_rate) —
        # otherwise they blend over the bandit's exploration (incl. the slow orac chunk), understating
        # the winner ~2x and diverging across GPUs.
        cfg_stats = {c: {"models": 0, "attempts": 0, "wall": 0.0, "chunks": 0,
                         "decs": 0, "props": 0} for c in plan}
        models: list[list[int]] = []
        attempts = 0
        n_models = 0
        n_support_complete = 0
        rounds_hist: list[int] = []
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
            # only plan[0] ever runs and a faster config is never discovered (this was a real
            # regression on shallow circuits with a large-VRAM card). Bounded by max_chunks. Cold-start chunks are productive (real models).
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
            chunk_cfgs = [self._select_config(plan, cfg_stats, chunk, exploit=topup)
                          if adaptive else plan[0]]
            for cfg in chunk_cfgs:
                rv_c, B_c, la_c = cfg
                K_c = self.batch_size
                if adaptive:
                    self._sync()
                    # read BEFORE t_cfg so the (synced) counter reads aren't charged to wall
                    p_before = self.engine.props_total()
                    d_before = self.engine.decs_total()
                    t_cfg = time.perf_counter()
                n_done, n_sup, rounds = self._run_chunk(
                    K_c, B_c, rv_c, support_mask, complete, la_c)
                rounds_hist.append(rounds)
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
            "n_chunks": chunk, "K": K, "B": B, "random_var": bool(random_var),
            "support_size": (len(sup) if sup else None),
            "mean_decision_rounds": (sum(rounds_hist) / len(rounds_hist)) if rounds_hist else 0,
            "max_decision_rounds": max(rounds_hist) if rounds_hist else 0,
            "n_distinct": len(distinct) if keep_rows else None,
            "gpu_sound": gpu_unsound == 0, "gpu_unsound": gpu_unsound,
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

            def _lab(c):
                # bool()/str() rather than the raw values: a config may be authored with the
                # pipeline's VarOrder/Lookahead enums, whose repr would otherwise leak into the
                # results-JSON KEYS ("rv=VarOrder.RANDOM,B=8") and break comparison against rows
                # recorded before those enums existed.
                return f"rv={bool(c[0])},B={c[1]}" + (f",la={str(c[2])}" if c[2] else "")
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
