"""GPU-parallel SAT solution sampler via exact BCP-completion.

`GPUSampler` runs K independent randomized DPLL-completions per chunk: each trajectory picks the
next B free variables, gives each a random per-sample polarity and propagates to fixpoint, all
in one fused kernel launch (`engine.run_trajectory`, the C++ core `libgpap`; CUDA only). A
conflicted trajectory is discarded; a fully-assigned conflict-free one is a guaranteed-valid
model (every clause keeps a non-falsified literal), so no rounding or repair is needed.

K is auto-sized to the GPU; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, set in
`gpap/__init__.py`, is what makes that K reachable.
"""
import contextlib

import numpy as np
import torch

from gpap.engine import _core as _gcore
from gpap.engine.formula import CnfFormula
from gpap.engine.bcp_engine import BCPEngine
from gpap.engine.calibrate import resolve_k_cap


# The core spells "no limit" as 0; here `None` does — a caller's 0 is a spent budget.
_NO_LIMIT = 0


def _chunk_budget(v):
    """`None` = unbounded. A non-positive budget is refused (the core cannot run zero chunks)."""
    if v is None:
        return _NO_LIMIT
    if int(v) < 1:
        raise ValueError(f"max_chunks must be >= 1 (got {v}); pass None for no limit")
    return int(v)


def _time_budget(v):
    """`None` = unbounded. A spent budget (0.0) clamps to a tiny positive value so the core's
    `time_limit_s > 0` valve still fires (checked before each chunk, so it buys zero chunks)."""
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
        """`k_cap` bounds the per-chunk batch: "auto" (default) is the library's batch cap
        `calibrate.default_k_cap`, an int is that cap verbatim (what a cross-GPU A/B needs), and
        None means no cap, leaving it to the analytic memory model."""
        self.formula = formula
        self.device = device
        self.k_cap = k_cap = resolve_k_cap(k_cap, device)
        self.N = formula.num_vars
        self.M = formula.num_clauses
        self.engine = BCPEngine(formula)
        # K is charged per STATE clause (length >= 3): binaries are occ implication edges and
        # units live in the flush bitmap — neither costs per-row bytes (see gpap_row_bytes).
        self.M_state = self.engine.num_long
        self._max_cls = formula.max_clause_len or 1
        self._total_lits = formula.total_lits
        # Two streams: SAMPLING is the core's own RNG (shared with the C++ face), `_proj_gen`
        # drives DELIVERY (the don't-care bits the projection fill draws). Do not merge them:
        # delivery draws would shift later chunks' permutations, so `--out` would change which
        # models a run finds at the same --seed.
        self._proj_gen = torch.Generator(device=device)
        self._proj_gen.manual_seed((seed if seed is not None else 0) ^ 0x5EED_0117)
        self._seed = seed
        import random as _random
        self._pyrng = _random.Random(seed)
        self.support: list[int] | None = None   # 0-based independent-support vars (from `c ind`)
        self._recon: dict | None = None
        # The handle owns the batch (K = min(analytic bound, cap), gpap_compute_k_safe), the
        # sampling RNG, and the kernel pin (pinned to the chunk batch so a short warmup reset
        # cannot switch kernels mid-run; an explicit mode pin is never overridden).
        self._smp = _gcore.Sampler(self.engine._core, seed=(seed if seed is not None else 0),
                                   k_cap=(-1 if k_cap is None else int(k_cap)))
        self._sup_pushed = "unset"     # last support handed to the core (see _push_support)

    # Read `batch_size`, set it with pin_batch() / pinned_batch(); assigning `K` routes to
    # gpap_sampler_set_k, so a pin binds the actual chunk loop.
    @property
    def K(self) -> int:
        """Rows per chunk: the core sampler's resolved batch."""
        return self._smp.k

    @K.setter
    def K(self, n: int) -> None:
        self._smp.set_k(int(n))

    @property
    def batch_size(self) -> int:
        """Trajectories the next chunk will run — the number that sets throughput."""
        return self.K

    def pin_batch(self, n: int) -> int:
        """Force every chunk to exactly `n` rows, and return it — whenever a bound is required
        rather than merely intended, above all when rows are reconstructed to host RAM. Pinning
        above the calibrated value overrides the memory model, and platforms that spill to host
        RAM degrade silently instead of erroring (docs/BENCHMARKING.md §2)."""
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
        """Build a sampler from a DIMACS file. `simplify=True` samples the unit-propagation
        -compacted formula (smaller M -> larger K), reconstructing original models at collection
        time; `els=True` also does equivalent-literal substitution and implies simplify."""
        if not (simplify or els):
            formula = CnfFormula(device)
            formula.parse_cnf_file(path)
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
        # `clauses` is read-only here and downstream: no deep copy (seconds/call on big unrolls).
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
        if res is None:
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
        """Stand-in sampler for a formula simplify proved UNSAT: emits nothing, forever. It still
        records the original shape (nv + support), because a caller asking for bit output needs
        the column list to get back a well-formed (0, S) matrix."""
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

    # reset+trajectory+selection, the bandit, the stop valves and the delivery/verify accounting
    # all live in cpp/src/sample.cpp (`gpap_sampler_*`), shared with the C++ face.

    def _push_support(self, sup) -> None:
        """Hand the decision variable set to the core, skipping an unchanged one. Re-setting the
        same support is already a no-op in the core; this cache only saves the ctypes round-trip,
        so it may be coarser than the core's test and never finer."""
        key = tuple(sup) if sup else None
        if self._sup_pushed != key:
            self._smp.set_support(list(sup) if sup else None)
            self._sup_pushed = key

    def _reconstruct_els(self, comp_rows: list[list[int]]):
        """Reconstruct original models when ELS reduced the formula (simplify_cnf_els). Every var
        resolves through `rep_of`: forced vars take their pinned value, decided reps the sampled
        value, free reps one random bit per equivalence class so equivalent vars stay equal.
        Returns (models, support_projections)."""
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
        """Map compacted models back to original ones; returns (models, support_projections).
        Forced vars are pinned to their unit-propagated value, compacted vars take their sampled
        value, don't-care support vars are drawn uniformly (so the projection distribution is
        preserved), and remaining unmapped vars are set to False."""
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


    def projection_cols(self) -> list[int]:
        """The 0-based original variables each output column stands for: the shape of the output
        without the plan that fills it, so a caller that collected nothing still reports (0, S)."""
        r = self._recon
        if r is not None and r.get("orig_support"):
            return list(r["orig_support"])
        if r is not None:
            return list(range(r["orig_nv"]))
        if self.support:
            return list(self.support)
        return list(range(self.N))

    def _build_projection_plan(self) -> dict:
        """Per-output-column plan (cached) making a chunk's support projection a GPU gather +
        constant fill + random fill, with no Python per-row reconstruct: `gcol[i]` is the
        compacted column to gather (-1 if not), `flip[i]` the XOR on the gathered/random bit,
        `const[i]` the constant for a unit-forced column, `rgroup[i]` the random-draw group (-1
        if none) — ELS columns sharing a representative share a group. Columns default to the
        original support, else all original variables."""
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
            pass       # no row is ever projected; the plan carries only the (0, S) shape
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
                if v in fm:
                    const[i] = 1 if fm[v] else 0
                elif v in new_of_old:
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
        """(n, S) uint8 support projection from `raw`, the (n, |gather|) bit rows the core kept
        (plan["gcol"]'s >= 0 columns, in order): constant columns broadcast, gathered columns XOR
        their polarity flip, don't-care groups draw from `_proj_gen` (ELS-equivalent variables
        share a draw). Only this fill consumes delivery randomness."""
        n = raw.shape[0] if n is None else n
        plan_g = plan["gcol"] >= 0
        proj = np.broadcast_to(plan["const"], (n, plan["S"])).copy()
        if plan_g.any() and n:
            proj[:, plan_g] = raw[:n, :int(plan_g.sum())] ^ plan["flip"][plan_g]
        rm = plan["rgroup"] >= 0
        if plan["n_groups"] and rm.any() and n:
            bits = torch.randint(0, 2, (n, plan["n_groups"]), generator=self._proj_gen,
                                 dtype=torch.uint8, device=self.device).cpu().numpy()
            proj[:, rm] = bits[:, plan["rgroup"][rm]] ^ plan["flip"][rm]
        return proj

    def _gather_cols(self, plan):
        """The 0-based compacted columns the core must keep: the gather set, or one dummy column
        when the plan gathers nothing, since set_columns(empty) means EVERY variable in the ABI
        (the dummy is never read: raw is indexed only through the gather mask)."""
        g = plan["gcol"][plan["gcol"] >= 0]
        if len(g) or self.N == 0:       # N == 0: "every variable" IS zero columns — no dummy
            return g
        return np.zeros(1, dtype=np.int64)

    def reseed(self, seed: int) -> None:
        """Restore all three RNG streams — the core's sampling stream, the delivery fill
        (`_proj_gen`) and the host reconstruction fills (`_pyrng`) — so a run is deterministic
        from `seed`."""
        self._smp.reseed(seed)
        self._proj_gen.manual_seed(seed ^ 0x5EED_0117)
        import random as _random
        self._pyrng = _random.Random(seed)

    def sample_bits(self, n_target: int, *, config=(True, 8, False),
                    max_chunks: "int | None" = None, max_seconds: "float | None" = None):
        """Collect up to `n_target` support-projection bit-rows; only the (n, S) bit matrix
        crosses to host RAM per chunk, never full N-var models. Returns (matrix, cols_1based):
        a numpy uint8 (got, S) array of 0/1 and the original variable id of each column; every
        row is a valid support projection. `max_seconds` bounds the sampling wall between
        chunks. Prefer `sample(out_cap=...)`, which takes these rows off chunks the run runs
        anyway."""
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
        """Draw at least `n_target` valid models. Stops early on any optional valve: `max_chunks`
        chunks run, `max_seconds` of sampling wall elapsed (checked between chunks), or zero
        models after every config was measured plus a couple of grace chunks.

        `support`: 0-based variables to restrict decisions to (the rest follow by propagation);
        "auto" (default) uses the support parsed from the CNF, None decides all variables.
        `n_models` counts fully-assigned models, every one guaranteed valid;
        `n_support_complete` counts trajectories that fixed the whole support without conflict
        (equal to `n_models` for a true independent support). With `verify_clauses` every model
        is independently checked and `all_valid` is set. `samples_per_s` is n_models over the
        sampling wall alone. `random_var`: True = random decision order, else index order.

        `decisions_per_round` (B, default 8, as on every other face) is a throughput AND a
        uniformity knob: larger B is more uniform, B >= |support| exactly so, while B=1 wins on
        success rate on tight/deep circuits (docs/findings/UNIFORMITY_FINDINGS.md).

        `configs`: a portfolio of (random_var, B[, lookahead]) tuples run as a throughput bandit,
        each chunk running ONE config at full K picked by its measured models/wall so far; a
        one-entry portfolio pins the algorithm, which is how a measurement is taken
        (pipeline.SINGLE_PORTFOLIOS). `config_breakdown` reports each config's
        models/chunks/throughput. Overrides random_var/decisions_per_round.

        `out_cap` > 0 keeps up to that many support-projection bit-rows as the run goes, in
        `out_bits` / `out_cols`, gathered off chunks the run executes anyway rather than in a
        second pass; only the winning config's bucket is returned, so the emitted set is ONE
        sampler and not a mixture of arms. Its wall is `out_project_s`, excluded from every rate
        (a pinned single config has no per-config wall, docs/findings/CLI_DELIVERY_FINDINGS.md).

        `verify_sound` (default off) re-checks on the device that every emitted model satisfies
        every clause, reporting `sound_check` ("ok" / "UNSOUND" / "off" — a tri-state, since
        "not run" is neither verdict), `sound_rows_checked` and `unsound_rows`. Its wall is
        `verify_sound_s`, excluded from every rate exactly as `out_project_s` is.
        """
        if isinstance(random_var, str):
            raise ValueError(f"random_var={random_var!r} is not a mode: pass a bool. Per-formula "
                             "adaptation is the portfolio bandit (configs=...); the old 'auto' "
                             "probe was retired — the bandit measures at full K and can change "
                             "its mind, which are exactly what the probe could not do.")
        B = max(1, decisions_per_round)
        sup = self.support if (isinstance(support, str) and support == "auto") else support
        sup_idx = list(sup) if sup else None

        # Normalise to (random_var, B, lookahead) triples: a 2-tuple leaves lookahead off, a
        # 3-tuple takes False or "soft" ("fused"/"fsoft" are aliases). "full"/"orac" raise.
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

        # What the core keeps per chunk: full compacted models for collect/verify, the plan's
        # gather set for an out_cap-only run, nothing for a pure timed run. keep_cap=0 means the
        # core never gathers a row — the emission-symmetry property the rival comparison needs.
        self._push_support(sup)
        pplan = self._build_projection_plan() if out_cap else None
        if keep_rows:
            self._smp.set_columns(None)
        elif out_cap:
            self._smp.set_columns(self._gather_cols(pplan))
        keep_cap = -1 if keep_rows else (int(out_cap) if out_cap else 0)

        self._sync()
        bcp_r0 = self.engine.launches
        bcp_o0 = self.engine.ops_total()
        bcp_p0 = self.engine.props_total()
        bcp_d0 = self.engine.decs_total()

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

        # Host-side collection over ALL configs' buckets, pooled: collection is a correctness
        # readout, so every model found belongs in it. Delivery via out_cap stays winner-only.
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

        # `elapsed` is the core's SAMPLING wall: kept-row gathers (`out_project_s`) and the
        # soundness re-check (`verify_sound_s`) are billed separately, so neither moves a rate.
        elapsed = float(st.sampling_s)
        bcp_rounds = self.engine.launches - bcp_r0
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
            "bcp_rounds": bcp_rounds, "bcp_ops": bcp_ops,
            "bcp_rounds_per_chunk": (bcp_rounds / chunk) if chunk else 0.0,
            "bcp_ops_per_attempt": (bcp_ops / attempts) if attempts else 0.0,
            # CDCL-comparable work rate: literals assigned per second.
            "assigns_per_s": (bcp_props / elapsed) if elapsed > 0 else 0.0,
            "decisions_per_s": (bcp_decs / elapsed) if elapsed > 0 else 0.0,
            "implications_per_s": (bcp_impl / elapsed) if elapsed > 0 else 0.0,
            "bcp_decisions": bcp_decs, "bcp_implications": bcp_impl,
            "n_chunks": chunk, "K": K, "B": B, "random_var": bool(random_var),
            # None = no support declared; 0 = declared and propagation determined all of it.
            "support_size": (None if sup is None else len(sup)),
            "mean_decision_rounds": float(st.rounds_mean),
            "max_decision_rounds": int(st.rounds_max),
            "n_distinct": len(distinct) if keep_rows else None,
            "stop": _gcore.SAMPLE_STOP.get(int(st.stop), "?"),
            # Tri-state, never a bool: "off" is not "unsound", and it also covers
            # requested-but-never-reached, whose evidence is `sound_rows_checked`.
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
            dom = plan[int(st.dominant_config)]
            out["random_var"], out["B"], out["lookahead"] = bool(dom[0]), dom[1], dom[2]
            out["dominant_config"] = [bool(dom[0]), dom[1], dom[2]]
            # Best-throughput config: the core's pick (>= min_measure_chunks first, then
            # models/wall), the same index whose kept rows out_cap returns below.
            bi = int(st.best_config)
            brc, brs = plan[bi], cbd[bi]
            best_cfg = brc
            out["best_rate_config"] = [bool(brc[0]), brc[1], brc[2]]
            out["best_rate_success"] = (brs.models / brs.attempts) if brs.attempts else 0.0
            out["best_rate_chunks"] = int(brs.chunks)
            # Only an ADAPTIVE run publishes a per-config steady-state rate.
            bw = float(brs.wall_s) if adaptive else 0.0
            out["best_rate_sps"] = (brs.models / bw) if bw > 0 else None
            out["best_rate_decisions_per_s"] = (brs.decs / bw) if bw > 0 else None
            out["best_rate_implications_per_s"] = ((brs.props - brs.decs) / bw) if bw > 0 else None
            out["best_rate_assigns_per_s"] = (brs.props / bw) if bw > 0 else None

            def _lab(c):
                # bool()/str(), not the raw values: a config may be authored with the pipeline's
                # VarOrder/Lookahead enums, whose repr would leak into results-JSON keys.
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
            # The winning config's rows only; a run that completed nothing returns (0, S).
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
    """Unit-propagation simplification: assert all units, propagate, drop satisfied clauses and
    falsified literals, compact the survivors to 1..nv'. Smaller M/N means a larger K. Returns
    (nv2, clauses2, support2, old_of_new, forced, free_support), or None if UNSAT. `support2` is
    0-based over the compacted formula, `old_of_new[i]` the original DIMACS var of compacted var
    i+1, `forced` the DIMACS lits fixed by propagation, `free_support` the 0-based support vars
    neither forced nor in any remaining clause — true don't-cares reconstruction must randomize
    uniformly and never drop. Sound because unit propagation is strictly model-preserving (no
    VE/BCE-style elimination), given that pinning (tests/sampling/test_sampler.py).
    """
    from itertools import chain
    from gpap.sampling.preprocess import unit_closure
    out, assigned, status = unit_closure(clauses, [])
    if status == "unsat":
        return None
    nonunit = [c for c in out if len(c) > 1]
    # Vectorised variable compaction: the listcomp version costs seconds of pure Python per call
    # on 1M-clause unrolls, and this runs on construct AND rebuild.
    lens = np.fromiter(map(len, nonunit), dtype=np.int64, count=len(nonunit))
    flat = (np.array(list(chain.from_iterable(nonunit)), dtype=np.int64) if lens.sum()
            else np.empty(0, dtype=np.int64))
    used_np = np.unique(np.abs(flat))
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
    """Unit-closure + equivalent-literal substitution (ELS) + compaction. On top of
    `simplify_cnf`, this collapses mutually-implied literals (binary-implication SCCs) to one
    representative, removing variables; cost scales with M and K with per-trajectory memory (~M),
    so a smaller formula hits the bottleneck twice. ELS preserves support projections iff
    reconstruction is consistent: a removed variable equals its representative in every model, so
    it must be reconstructed from that rep and never randomized independently — hence one draw
    per equivalence class (tests/sampling/test_sampler.py). Returns (nv_r, clauses_r, support_r, recon),
    or None if UNSAT; recon is the resolver used by GPUSampler._reconstruct:
      rep_of[v-1]  : (rep_var, same_polarity) — v takes rep_var's value, negated if not.
      new_of_new   : {rep_var: compacted_idx(1-based)} — reps surviving as decided vars.
      forced_map   : {var: bool} — vars pinned by either unit-closure pass.
    """
    from gpap.engine.formula import ext_lit_to_lit_index
    from gpap.sampling.preprocess import equivalent_literal_substitution, unit_closure

    # Pass 1: unit closure over the original variables (reads only — no copy).
    out1, assigned1, st1 = unit_closure(clauses, [])
    if st1 == "unsat":
        return None
    nonunit1 = [c for c in out1 if len(c) > 1]

    # Binary-implication edges (2N lit-index scheme): (l1∨l2) ⇒ ¬l1→l2, ¬l2→l1.
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

    used = sorted({abs(l) for c in nonunit2 for l in c})
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
            rep_of.append((v, True))

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
    """Read the independent support from `c ind <v1> ... 0` header lines (UniGen convention);
    returns 0-based variable indices, or None if no `c ind` line is present."""
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
                        sup.add(abs(v) - 1)
            elif s[0].isdigit() or s[0] == "-":
                break                                  # past the header into clauses
    return sorted(sup) if found else None
