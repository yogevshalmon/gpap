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
                        more dead-ends). DEFAULT 8 — the same number every other entry point
                        already used (`gpap::SampleOptions::batch`, `sample_bits`,
                        `pipeline.sample_clauses(b=)`, `gpap bench --b`, and the first entry of
                        DEFAULT_PORTFOLIO). It was 1 here and only here, so a direct
                        GPUSampler user got a different sampler from every other face.
  random_var          : pick a random free var per sample (diverse) vs. the first free
                        one (cheap, low-index-biased).

K is auto-sized to the GPU (capped by default to stay off the host-RAM spill cliff).
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is what makes that K reachable; it is set in
`gpap/__init__.py`, which always runs before this module (see there for the measurement).
"""
import contextlib

import numpy as np
import torch

from gpap.engine import _core as _gcore
from gpap.engine.formula import CnfFormula
from gpap.engine.bcp_engine import BCPEngine
from gpap.engine.calibrate import resolve_k_cap


# The core spells "no limit" as 0, so `max_chunks or 0` / `max_seconds or 0.0` turned a SPENT
# budget into an UNBOUNDED run -- the exact inversion of what the caller asked for, and silent.
# `None` is the only spelling of "no limit" here; a 0 is a real, exhausted budget.
_NO_LIMIT = 0


def _chunk_budget(v):
    """`None` = unbounded. The core cannot express "run zero chunks", so a non-positive budget
    is refused rather than silently promoted to unbounded."""
    if v is None:
        return _NO_LIMIT
    if int(v) < 1:
        raise ValueError(f"max_chunks must be >= 1 (got {v}); pass None for no limit")
    return int(v)


def _time_budget(v):
    """`None` = unbounded. A SPENT budget (0.0, the natural value of `limit - elapsed`) clamps to
    a tiny positive number so the core's `time_limit_s > 0` valve still fires at all, instead of
    reading as no limit. The valve is checked BEFORE each chunk, so a spent budget buys ZERO
    further chunks — which is what a caller who has already used its time asked for."""
    if v is None:
        return _NO_LIMIT
    return float(v) if float(v) > 0 else 1e-9


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

            "auto"  (default) the library's batch cap (calibrate.default_k_cap) — what a
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
        # TWO streams, deliberately. SAMPLING (the per-chunk shuffle and trajectory seed) is
        # the CORE sampler's own deterministic RNG, seeded below and shared with the C++ face.
        # `_proj_gen` drives DELIVERY (the don't-care bits the projection fill draws). They must
        # not share a stream: delivery draws would otherwise shift every later chunk's
        # permutation, making `--out` change which models a run finds at the same --seed. The
        # core enforces its half by construction (its kept-row gathers draw nothing); this
        # generator keeps the host half honest.
        self._proj_gen = torch.Generator(device=device)
        self._proj_gen.manual_seed((seed if seed is not None else 0) ^ 0x5EED_0117)
        self._seed = seed
        import random as _random
        self._pyrng = _random.Random(seed)
        self.support: list[int] | None = None   # 0-based independent-support vars (from `c ind`)
        self._recon: dict | None = None          # set when built with simplify=True (see _from_simplified)
        # THE chunk loop lives in the core (cpp/src/sample.cpp, `gpap_sampler_*`), shared
        # verbatim with the C++ face — this class marshals configs and knobs in, stats and kept
        # rows out. The handle owns the batch (K = min(analytic bound, cap) over the exact
        # per-row allocation list, gpap_compute_k_safe), the SAMPLING RNG, and the kernel pin
        # (the propagate variant is pinned to the chunk batch at create, so a short warmup
        # reset cannot switch kernels mid-measurement; an explicit $GPAP_KERNEL / mode pin is
        # never overridden — gpap_engine_auto guards it, in the core).
        self._smp = _gcore.Sampler(self.engine._core, seed=(seed if seed is not None else 0),
                                   k_cap=(-1 if k_cap is None else int(k_cap)))
        self._sup_pushed = "unset"     # last support handed to the core (see _push_support)

    # ---------------------------------------------------------------- THE batch size
    # Read `batch_size`; set it with pin_batch() / pinned_batch(). `K` is held by the CORE
    # sampler — assigning it routes to gpap_sampler_set_k, so a pin binds the actual chunk loop.
    @property
    def K(self) -> int:
        """Rows per chunk — the core sampler's resolved batch."""
        return self._smp.k

    @K.setter
    def K(self, n: int) -> None:
        self._smp.set_k(int(n))

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

    # ------------------------------------------------------------- the core chunk loop
    # reset+trajectory+selection, the config bandit, the stop valves and the delivery/verify
    # accounting ALL live in cpp/src/sample.cpp (`gpap_sampler_*`), shared with the C++ face.
    # tests/test_sample_core.py pins that loop's own contract at the ABI level; nothing here
    # re-implements any of it.

    def _push_support(self, sup) -> None:
        """Hand the decision phase-1 variable set to the core, skipping an unchanged one.

        Re-setting the SAME support is already a no-op IN THE CORE — that is where the rule
        lives, because a pair mask lives until the engine dies (see gpap.h) and both faces
        re-push on every call. This cache only saves the ctypes round-trip, so it is allowed to
        be COARSER than the core's test (which canonicalises the set) and never finer: a key
        miss costs one no-op call, and there is no key hit the core would have acted on."""
        key = tuple(sup) if sup else None
        if self._sup_pushed != key:
            self._smp.set_support(list(sup) if sup else None)
            self._sup_pushed = key

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

    def _fill_projection(self, raw, plan, n=None):
        """(n, S) uint8 support projection from `raw` — the (n, |gather|) bit rows the CORE
        kept (plan["gcol"]'s >= 0 columns, in that order) — via the cached plan: constant
        columns broadcast, gathered columns XOR their polarity flip, don't-care groups drawn
        from `_proj_gen` (ELS-equivalent variables share a draw — a hard validity requirement).
        The gather itself happened in the core, billed to deliver_s; only this fill is host
        work, and only IT consumes delivery randomness."""
        n = raw.shape[0] if n is None else n
        plan_g = plan["gcol"] >= 0
        proj = np.broadcast_to(plan["const"], (n, plan["S"])).copy()   # (n, S) uint8
        if plan_g.any() and n:
            proj[:, plan_g] = raw[:n, :int(plan_g.sum())] ^ plan["flip"][plan_g]
        rm = plan["rgroup"] >= 0
        if plan["n_groups"] and rm.any() and n:
            # torch RNG keeps the don't-care stream on `_proj_gen`, never the sampling one.
            bits = torch.randint(0, 2, (n, plan["n_groups"]), generator=self._proj_gen,
                                 dtype=torch.uint8, device=self.device).cpu().numpy()
            proj[:, rm] = bits[:, plan["rgroup"][rm]] ^ plan["flip"][rm]
        return proj

    def _gather_cols(self, plan):
        """The 0-based compacted columns the core must keep for this plan — the gather set, or
        ONE dummy column when the plan gathers nothing: set_columns(empty) means EVERY variable
        in the ABI, which is the opposite of "none", and `_fill_projection` reads a raw matrix
        only through the gather mask, so a dummy column is never looked at."""
        g = plan["gcol"][plan["gcol"] >= 0]
        if len(g) or self.N == 0:       # N == 0: "every variable" IS zero columns — no dummy
            return g
        return np.zeros(1, dtype=np.int64)

    def reseed(self, seed: int) -> None:
        """Restore EVERY RNG stream to a known point.

        The warmup calls this so the timed run is deterministic from `seed` alone. It exists as a
        method because there are THREE streams to restore and they live in three places — the
        core's sampling stream (`gpap_sampler_reseed`), the delivery fill (`_proj_gen`) and the
        host reconstruction fills (`_pyrng`) — so a caller that reseeds only the one it knows
        about silently loses that determinism for the others."""
        self._smp.reseed(seed)
        self._proj_gen.manual_seed(seed ^ 0x5EED_0117)
        import random as _random
        self._pyrng = _random.Random(seed)

    def sample_bits(self, n_target: int, *, config=(True, 8, False),
                    max_chunks: "int | None" = None, max_seconds: "float | None" = None):
        """Collect up to `n_target` support-projection bit-rows efficiently. Only the small
        (n, S) bit matrix crosses to host RAM per chunk (not full N-var models), so a big output
        is cheap. Returns (matrix, cols_1based): matrix is a numpy uint8 (got, S) array of 0/1,
        cols_1based lists the ORIGINAL variable id of each column. Sound: every row is a valid
        support projection (forced pinned, compacted vars from a real model, don't-cares free,
        ELS-equivalent vars kept equal via shared random groups).

        `max_seconds` bounds the wall (checked between chunks), like `sample`'s. It is not
        optional in spirit: this is a SAMPLING loop, so without it a `--time-limit`ed run keeps
        sampling here long after its budget was spent. Prefer `sample(out_cap=...)`, which takes
        these rows off chunks the timed run executes anyway; this entry point remains for topping
        up a short collection and for callers with no timed run.

        The loop is the core's (`gpap_sampler_run`, keep_cap = n_target): its zero-yield stop is
        TOTAL-based — it gives up only when NOTHING has been found after the grace window, where
        the old Python loop abandoned a delivering fill on one late empty chunk.
        """
        rv, B, la = config[0], max(1, config[1]), (config[2] if len(config) > 2 else False)
        plan = self._build_projection_plan()
        self._push_support(self.support)
        self._smp.set_columns(self._gather_cols(plan))
        self._smp.run(int(n_target), [(rv, B, la in ("soft", "fsoft"))],
                      max_chunks=_chunk_budget(max_chunks),
                      time_limit_s=_time_budget(max_seconds),
                      keep_cap=int(n_target))
        take = min(int(self._smp.config_stats(0).kept_rows), int(n_target))
        raw = (self._smp.bits(0, rows=take) if take
               else np.zeros((0, 1), dtype=np.uint8))
        return self._fill_projection(raw, plan, take), plan["cols_1based"]

    # ----------------------------------------------------------------- public

    @torch.inference_mode()
    def sample(
        self,
        n_target: int,
        *,
        decisions_per_round: int = 8,
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
        out_cap: int = 0,
        verify_sound: bool = False,
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

        `decisions_per_round` (B) defaults to **8**, matching `gpap::SampleOptions::batch`,
        `sample_bits`, `pipeline.sample_clauses(b=)`, `gpap bench --b` and DEFAULT_PORTFOLIO's
        first entry — it was 1 on this method alone, so a direct library caller silently got a
        different sampler from the one every other face runs. B is also a UNIFORMITY knob and
        the change moves the default the safe way: against the exact oracle, B=8's median TV is
        0.156 to B=1's 0.315, and B >= |support| is exactly uniform
        (docs/findings/UNIFORMITY_FINDINGS.md). Pass `decisions_per_round=1` for the old
        behaviour — B=1 still wins on success rate on tight/deep circuits, which is why the
        portfolios carry it as an arm.

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

        `out_cap` > 0 also KEEPS up to that many support-projection bit-rows as the run goes, in
        `out_bits` / `out_cols` — the rows a caller writes to a file. The core keeps the projection's
        gather columns off chunks this run executes anyway (billed to deliver_s), against the
        second full sampling pass a post-hoc collection costs; the host fill finishes the rows. Rows are bucketed PER CONFIG and only the winning config's are returned,
        so the emitted set comes from ONE sampler rather than a mixture of the bandit's arms (a
        mixture is a uniformity question, not a delivery one). Its wall is accumulated separately
        as `out_project_s` and excluded from every rate — the per-config stamp alone was not
        enough, because a pinned single config has no per-config wall at all.
        Measurements: docs/findings/CLI_DELIVERY_FINDINGS.md.

        `verify_sound` (default OFF) re-checks on the device that EVERY emitted model satisfies
        EVERY clause, and reports `sound_check` / `unsound_rows`. It is opt-in because it is a
        self-test, not part of producing samples: a row is only counted when it is active and
        fully assigned, which already means no clause was emptied. Turn it on in tests, in a
        campaign whose rows will be published, or when something looks wrong. Its wall is
        accumulated into `verify_sound_s` and EXCLUDED from every rate, exactly as the output
        projection is, so enabling it cannot move samples/s. Cost is per completed row and
        scales with the formula, so a large instance pays more than a small one (measured
        a fraction of a percent of the chunk it rides on).
        """
        if isinstance(random_var, str):
            raise ValueError(f"random_var={random_var!r} is not a mode: pass a bool. Per-formula "
                             "adaptation is the portfolio bandit (configs=...); the old 'auto' "
                             "probe was retired — the bandit measures at full K and can change "
                             "its mind, which are exactly what the probe could not do.")
        B = max(1, decisions_per_round)
        sup = self.support if (isinstance(support, str) and support == "auto") else support
        sup_idx = list(sup) if sup else None

        # Normalise the portfolio to (random_var, B, lookahead) triples. A config may be a
        # 2-tuple (rv, B) — lookahead defaults off — or a 3-tuple, lookahead ∈ {False, "soft"}
        # ("fused"/"fsoft" accepted as aliases from the pre-packed-only era). The retired modes
        # raise: "full" python lookahead and "orac" needed the deleted PyTorch engine path.
        def _norm(c):
            la = c[2] if len(c) > 2 else False
            if la in ("full", "orac", True):
                raise ValueError(f"lookahead mode {la!r} was retired with the PyTorch engine "
                                 "path; use False or 'soft'")
            return (c[0], max(1, c[1]), "soft" if la in ("soft", "fsoft") else False)
        plan = [_norm(c) for c in configs] if configs else [_norm((random_var, B, lookahead))]
        K = self.batch_size
        adaptive = configs is not None and len(plan) > 1
        keep_rows = collect or verify_clauses is not None

        # What the core keeps per chunk: full compacted models for collect/verify (host
        # reconstruction needs every variable), the projection plan's gather set for an
        # out_cap-only run, NOTHING for a pure timed run — keep_cap=0 means the core never
        # gathers a row, which is the emission-symmetry property the rival comparison rests on.
        self._push_support(sup)
        pplan = self._build_projection_plan() if out_cap else None
        if keep_rows:
            self._smp.set_columns(None)
        elif out_cap:
            self._smp.set_columns(self._gather_cols(pplan))
        keep_cap = -1 if keep_rows else (int(out_cap) if out_cap else 0)

        # Work counters bracket the run (the engine keeps counting; only deltas are ours).
        self._sync()
        bcp_r0 = self.engine.launches
        bcp_o0 = self.engine.ops_total()
        bcp_p0 = self.engine.props_total()
        bcp_d0 = self.engine.decs_total()

        # ONE call: the loop, the bandit, the valves and the accounting are the core's
        # (cpp/src/sample.cpp) — see the class docstring and tests/test_sample_core.py.
        st = self._smp.run(
            int(n_target), [(rv_c, b_c, la_c == "soft") for rv_c, b_c, la_c in plan],
            max_chunks=_chunk_budget(max_chunks), time_limit_s=_time_budget(max_seconds),
            min_measure_chunks=int(min_measure_chunks), keep_cap=keep_cap,
            verify=bool(verify_sound), support_only=not complete)
        self._sync()
        cbd = [self._smp.config_stats(i) for i in range(len(plan))]

        n_models, attempts, chunk = int(st.n_models), int(st.attempts), int(st.chunks)
        proj_s, verify_s = float(st.deliver_s), float(st.verify_s)
        gpu_unsound, sound_rows = int(st.unsound_rows), int(st.sound_rows_checked)

        # ---- host-side collection: reconstruct / verify / dedup off the kept buckets ----
        # ALL configs' buckets, pooled — collection is a correctness readout, so every model
        # found belongs in it (delivery via out_cap below stays winner-only, the ONE-sampler
        # rule; a mixture is a uniformity question and that is a delivery decision).
        models: list[list[int]] = []
        distinct: set = set()
        all_valid = True
        if keep_rows and n_models:
            vidx = np.arange(1, self.N + 1, dtype=np.int32)
            for i in range(len(plan)):
                kept = int(cbd[i].kept_rows)
                if not kept:
                    continue
                pos = self._smp.bits(i, rows=kept)          # (kept, N) uint8, compacted vars
                comp_rows = np.where(pos, vidx, -vidx).tolist()
                if self._recon is not None and not self._recon.get("unsat"):
                    # reconstruct full ORIGINAL models (forced pinned, don't-cares randomized)
                    out_rows, projs = self._reconstruct(comp_rows)
                else:
                    out_rows = comp_rows
                    projs = ([tuple(r[j] for j in sup_idx) for r in comp_rows]
                             if sup_idx is not None else comp_rows)
                for pr in projs:
                    distinct.add(tuple(pr))
                if verify_clauses is not None:
                    for r in out_rows:
                        if not _satisfies(r, verify_clauses):
                            all_valid = False
                if collect:
                    models.extend(out_rows)

        # `elapsed` is the core's SAMPLING wall: kept-row gathers are delivery and the
        # soundness re-check is a self-test, both billed separately (`out_project_s` /
        # `verify_sound_s`), so neither asking for rows nor asking for verification can move
        # samples/s, dec/s or impl/s.
        elapsed = float(st.sampling_s)
        bcp_rounds = self.engine.launches - bcp_r0             # kernel launches (delta)
        bcp_ops = self.engine.ops_total() - bcp_o0
        bcp_props = self.engine.props_total() - bcp_p0
        bcp_decs = self.engine.decs_total() - bcp_d0
        bcp_impl = bcp_props - bcp_decs                        # every non-decision assignment

        out = {
            "n_models": n_models, "n_support_complete": int(st.n_support_complete),
            "attempts": attempts,
            "success_rate": (n_models / attempts) if attempts else 0.0,
            "support_success_rate": (int(st.n_support_complete) / attempts) if attempts else 0.0,
            "elapsed_s": elapsed,
            "samples_per_s": (n_models / elapsed) if elapsed > 0 else 0.0,
            "attempts_per_s": (attempts / elapsed) if elapsed > 0 else 0.0,
            # Lightweight BCP-work stats (free counters) — to tell success-bound from BCP-bound.
            "bcp_rounds": bcp_rounds, "bcp_ops": bcp_ops,
            "bcp_rounds_per_chunk": (bcp_rounds / chunk) if chunk else 0.0,
            "bcp_ops_per_attempt": (bcp_ops / attempts) if attempts else 0.0,
            # CDCL-comparable work rate: literals assigned (decisions + implications) per second.
            "assigns_per_s": (bcp_props / elapsed) if elapsed > 0 else 0.0,
            "decisions_per_s": (bcp_decs / elapsed) if elapsed > 0 else 0.0,
            "implications_per_s": (bcp_impl / elapsed) if elapsed > 0 else 0.0,
            "bcp_decisions": bcp_decs, "bcp_implications": bcp_impl,
            "n_chunks": chunk, "K": K, "B": B, "random_var": bool(random_var),
            "support_size": (len(sup) if sup else None),
            "mean_decision_rounds": float(st.rounds_mean),
            "max_decision_rounds": int(st.rounds_max),
            "n_distinct": len(distinct) if keep_rows else None,
            # Why the run ended — the core's stop valve, by name (new with the core loop; the
            # legacy keys above still carry everything recorded rows compare on).
            "stop": _gcore.SAMPLE_STOP.get(int(st.stop), "?"),
            # Tri-state, never a bool: "off" is not "unsound", and it also covers
            # REQUESTED-BUT-NEVER-REACHED (verify on, nothing completed, check never ran —
            # `sound_rows_checked` is the evidence a verdict is a statement about).
            "sound_check": ("off" if not (verify_sound and sound_rows) else
                            ("ok" if gpu_unsound == 0 else "UNSOUND")),
            "sound_rows_checked": (sound_rows if verify_sound else None),
            "unsound_rows": (gpu_unsound if verify_sound else None),
            "verify_sound_s": verify_s,
        }
        if collect:
            out["models"] = models
        if verify_clauses is not None:
            out["all_valid"] = all_valid
        best_cfg = plan[0]
        if configs is not None:
            dom = plan[int(st.dominant_config)]                   # most models produced
            out["random_var"], out["B"], out["lookahead"] = bool(dom[0]), dom[1], dom[2]
            out["dominant_config"] = [bool(dom[0]), dom[1], dom[2]]
            # Best-THROUGHPUT config — the core's pick (well-measured >= min_measure_chunks
            # first, then models/wall), the same index whose kept rows out_cap returns below.
            bi = int(st.best_config)
            brc, brs = plan[bi], cbd[bi]
            best_cfg = brc
            out["best_rate_config"] = [bool(brc[0]), brc[1], brc[2]]
            out["best_rate_success"] = (brs.models / brs.attempts) if brs.attempts else 0.0
            out["best_rate_chunks"] = int(brs.chunks)
            # Per-config walls now exist for every run (the core stamps them cheaply), but the
            # best_rate_* RATES keep their documented shape: only an ADAPTIVE run publishes a
            # per-config steady-state rate — a pinned single config's headline stays the
            # caller-side blend, exactly what every recorded row was produced under.
            bw = float(brs.wall_s) if adaptive else 0.0
            out["best_rate_sps"] = (brs.models / bw) if bw > 0 else None
            out["best_rate_decisions_per_s"] = (brs.decs / bw) if bw > 0 else None
            out["best_rate_implications_per_s"] = ((brs.props - brs.decs) / bw) if bw > 0 else None
            out["best_rate_assigns_per_s"] = (brs.props / bw) if bw > 0 else None

            def _lab(c):
                # bool()/str() rather than the raw values: a config may be authored with the
                # pipeline's VarOrder/Lookahead enums, whose repr would otherwise leak into the
                # results-JSON KEYS and break comparison against recorded rows.
                return f"rv={bool(c[0])},B={c[1]}" + (f",la={str(c[2])}" if c[2] else "")
            out["config_breakdown"] = {
                _lab(c): {
                    "models": int(sd.models), "attempts": int(sd.attempts),
                    "chunks": int(sd.chunks),
                    "wall_s": float(sd.wall_s) if adaptive else 0.0,
                    "success_rate": (sd.models / sd.attempts) if sd.attempts else 0.0,
                    "samples_per_s": (sd.models / sd.wall_s)
                                     if (adaptive and sd.wall_s > 0) else None,
                    "decisions_per_s": (sd.decs / sd.wall_s)
                                       if (adaptive and sd.wall_s > 0) else None,
                    "implications_per_s": ((sd.props - sd.decs) / sd.wall_s)
                                          if (adaptive and sd.wall_s > 0) else None,
                } for c, sd in zip(plan, cbd)
            }
        if out_cap:
            # The WINNING config's rows only (see `out_cap` in the docstring). Always
            # well-shaped — a run that completed nothing returns (0, S) over the same columns.
            bi = int(st.best_config)
            kept = min(int(cbd[bi].kept_rows), int(out_cap))
            if kept:
                raw = self._smp.bits(bi, rows=kept)
                if keep_rows:                       # bucket holds ALL vars: slice the gathers
                    gsel = pplan["gcol"][pplan["gcol"] >= 0]
                    raw = raw[:, gsel] if len(gsel) else raw[:, :0]
                out["out_bits"] = self._fill_projection(raw, pplan, kept)
                out["out_cols"] = pplan["cols_1based"]
            else:
                cols = [v + 1 for v in self.projection_cols()]
                out["out_bits"] = np.zeros((0, len(cols)), dtype=np.uint8)
                out["out_cols"] = cols
            out["out_config"] = [bool(best_cfg[0]), best_cfg[1], best_cfg[2]]
            out["out_project_s"] = proj_s
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
