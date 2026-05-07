"""AlphaFold2.3-multimer reward backend for VIDD's --task=ab path.

Bonobo-style: the binder template is supplied as a pre-made multi-chain PDB
(target chain + binder chain on disk), referenced via ``args.template_pdb``.
No NBB2 binder pre-folding at runtime. Build templates offline (e.g. with
``ProDifEvo-Refinement/scripts/generate_template.py``).

Two underlying models are constructed lazily:

* ``AFModel(protocol="binder", use_multimer=True)`` for predicting the
  antibody:antigen complex (used when ``iptm`` is requested). Uses
  ``args.template_pdb`` as the multi-chain template.
* ``AFModel(protocol="hallucination", use_multimer=False)`` for predicting
  the antibody monomer (used when no antigen / no iptm).

Uses mber-open's ``AFModel`` (subclass of colabdesign's ``mk_af_model`` with
mber-open mixins) so string-position rm_binder/seq/sc args are parsed as
binder-chain positions to mask. Stock colabdesign would treat a truthy
string as ``True`` (mask all binder), giving materially different
predictions vs bonobo.

* Race-condition-free: each AF worker is owned by exactly one
  ``ThreadPoolExecutor`` task at a time (one task per worker, processing a
  shard sequentially) — see ``_reward_metrics_parallel``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Sequence

import jax.numpy as jnp
import numpy as np

# evodiff utils provides set_diversity for ``calc_diversity`` (mirrors
# protein_eval_bind_colabdesign.py's import path).
from evaluations.protein_utils import set_diversity


# Lazy module handles: colabdesign / mber-open are heavy and may not be
# importable on hosts without GPUs. Defer to first call.
_AF_FACTORY = None
_CLEAR_MEM = None


def _lazy_import_colabdesign():
    """Import the mber-open AFModel + colabdesign clear_mem on first use.

    Stock colabdesign's ``_prep_binder`` doesn't accept string-position
    rm_binder/seq/sc args; mber-open's override does, and that's the bonobo
    parity we need.
    """
    global _AF_FACTORY, _CLEAR_MEM
    if _AF_FACTORY is not None:
        return _AF_FACTORY, _CLEAR_MEM
    try:
        from mber.models.colabdesign.model import AFModel
        from colabdesign import clear_mem
    except ImportError as e:
        raise ImportError(
            "mber + colabdesign are required for the AF2 reward backend. Install with:\n"
            "  pip install -e mber-open/\n"
            "  pip install 'colabdesign @ git+https://github.com/sokrypton/ColabDesign.git@d024c4e'\n"
            "and ensure jax/flax are installed."
        ) from e
    _AF_FACTORY = AFModel
    _CLEAR_MEM = clear_mem
    return _AF_FACTORY, _CLEAR_MEM


ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

# Bonobo-style hardcoded CDR mask: covers Chothia-style CDR-H1 (H27-H35),
# CDR-H2 (H48-H58), CDR-H3 (H96-H107) — 32 positions on the binder chain.
# Used identically for rm_binder, rm_binder_seq, rm_binder_sc so AF
# re-hallucinates CDR backbone, sequence identity, and sidechains while
# keeping the framework templated. Hardcoded (not exposed as a CLI arg) to
# match bonobo exactly — this is load-bearing for ipTM parity.
DEFAULT_RM_BINDER_POSITIONS = ",".join(
    [f"H{i}" for i in range(27, 36)]   # CDR-H1: H27..H35
    + [f"H{i}" for i in range(48, 59)] # CDR-H2: H48..H58
    + [f"H{i}" for i in range(96, 108)]# CDR-H3: H96..H107
)


# ============================================================
# Metric extraction helpers (operate on colabdesign ``aux`` dict)
# ============================================================

def _get_log(aux: dict) -> dict:
    return aux.get("log", {})


def _per_residue_plddt(aux: dict) -> np.ndarray:
    plddt = aux.get("plddt")
    if plddt is None:
        raise KeyError("AF2 aux missing 'plddt'")
    arr = np.asarray(plddt)
    return arr.reshape(-1) * (100.0 if arr.max() <= 1.0 else 1.0)


def af2_to_ptm(aux: dict) -> float:
    return float(_get_log(aux).get("ptm", 0.0))


def af2_to_iptm(aux: dict) -> float:
    log = _get_log(aux)
    if "i_ptm" in log:
        return float(log["i_ptm"])
    if "iptm" in log:
        return float(log["iptm"])
    raise KeyError("AF2 aux['log'] does not contain i_ptm.")


def af2_to_plddt(aux: dict, binder_offset: int = 0, binder_len: Optional[int] = None) -> float:
    pl = _per_residue_plddt(aux)
    sl = pl[binder_offset:] if binder_len is None else pl[binder_offset:binder_offset + binder_len]
    if sl.size == 0:
        return 0.0
    return float(sl.mean()) / 100.0


def af2_to_cdr_plddt(
    aux: dict,
    cdr_indices: Sequence[int],
    binder_offset: int = 0,
) -> float:
    pl = _per_residue_plddt(aux)
    if len(cdr_indices) == 0:
        return 0.0
    cdr_global = np.asarray(cdr_indices, dtype=int) + binder_offset
    cdr_global = cdr_global[cdr_global < pl.size]
    if cdr_global.size == 0:
        return 0.0
    return float(pl[cdr_global].mean()) / 100.0


def af2_to_radius(aux: dict, binder_offset: int = 0, binder_len: Optional[int] = None) -> float:
    """Radius-of-gyration reward (matches ProteinEvalMetricsColabDesign).

    Returns -elu(rg - rg_threshold), so smaller (more compact) binders score higher.
    """
    atom_positions = aux.get("atom_positions")
    if atom_positions is None:
        raise KeyError("AF2 aux missing 'atom_positions'")
    # CA atom is always index 1 in colabdesign atom ordering. We pull the
    # binder slice and compute its radius of gyration.
    ca = jnp.asarray(atom_positions)[:, 1]
    if binder_len is not None:
        ca = ca[binder_offset:binder_offset + binder_len]
    elif binder_offset:
        ca = ca[binder_offset:]
    rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
    rg_th = 2.38 * ca.shape[0] ** 0.365
    import jax  # local import to avoid module-load side effects
    rg_value = jax.nn.elu(rg - rg_th).item()
    return float(-rg_value)


# ============================================================
# Per-GPU AF worker for multi-GPU parallelism
# ============================================================

class _AFWorker:
    """A single AF2 model instance pinned to one JAX device.

    Concurrency note: workers are NOT internally thread-safe — each worker
    mutates ``model.aux`` on every predict call. The dispatcher in
    ``AbAF2RewardCal._reward_metrics_parallel`` shards sequences such that
    each worker is owned by exactly one ThreadPoolExecutor task at a time.
    """

    def __init__(
        self,
        jax_device,
        af_params_dir: str,
        num_recycles: int,
        use_multimer: bool,
        af_models: Sequence[int],
        antigen_pdb: Optional[str],
        antigen_chain: str,
        binder_chain: str = "H",
        hotspot: Optional[str] = None,
    ):
        self.jax_device = jax_device
        self.af_params_dir = af_params_dir
        self.num_recycles = num_recycles
        self.use_multimer = use_multimer
        self.af_models = list(af_models)
        self.antigen_pdb = antigen_pdb
        self.antigen_chain = antigen_chain
        self.binder_chain = binder_chain
        self.hotspot = hotspot

        self._complex_model = None
        self._complex_target_len = 0
        self._complex_binder_len = 0
        self._template_initialized = False
        self._monomer_model = None
        self._monomer_len = 0

    def _ensure_complex_model(self, ab_len: int) -> None:
        """Build the non-templated binder model (full hallucination of binder)."""
        mk_model, clear_mem = _lazy_import_colabdesign()
        if self._complex_model is None or self._complex_binder_len != ab_len:
            if self._complex_model is not None:
                clear_mem()
            self._complex_model = mk_model(
                protocol="binder",
                debug=False,
                data_dir=self.af_params_dir,
                use_multimer=self.use_multimer,
                num_recycles=self.num_recycles,
            )
            self._complex_model._prep_binder(
                pdb_filename=self.antigen_pdb,
                chain=self.antigen_chain,
                binder_len=ab_len,
                hotspot=self.hotspot,
                seed=0,
                rm_target=False,
                rm_target_seq=False,
                rm_target_sc=False,
                rm_template_ic=True,
                rm_binder=True,
                rm_binder_seq=True,
                rm_binder_sc=True,
            )
            self._complex_binder_len = ab_len
            self._complex_target_len = int(self._complex_model._target_len)

    def init_template(self, template_pdb_path: str, rm_binder_str: str) -> None:
        """One-shot template init: build a binder model with the combined target+binder PDB,
        masking the binder template at the bonobo-style CDR positions.

        ``rm_binder``, ``rm_binder_seq``, ``rm_binder_sc`` are all set to the same
        position string (per bonobo).
        """
        import jax
        mk_model, clear_mem = _lazy_import_colabdesign()
        with jax.default_device(self.jax_device):
            if self._complex_model is not None:
                clear_mem()
            self._complex_model = mk_model(
                protocol="binder",
                debug=False,
                data_dir=self.af_params_dir,
                use_multimer=self.use_multimer,
                num_recycles=self.num_recycles,
            )
            self._complex_model._prep_binder(
                pdb_filename=template_pdb_path,
                chain=self.antigen_chain,
                binder_chain=self.binder_chain,
                hotspot=self.hotspot,
                seed=0,
                rm_target=False,
                rm_target_seq=False,
                rm_target_sc=False,
                rm_template_ic=True,
                rm_binder=rm_binder_str,
                rm_binder_seq=rm_binder_str,
                rm_binder_sc=rm_binder_str,
            )
            self._complex_target_len = int(self._complex_model._target_len)
            self._complex_binder_len = int(self._complex_model._binder_len)
            self._template_initialized = True

    def _ensure_monomer_model(self, ab_len: int) -> None:
        mk_model, clear_mem = _lazy_import_colabdesign()
        if self._monomer_model is None or self._monomer_len != ab_len:
            if self._monomer_model is not None:
                clear_mem()
            self._monomer_model = mk_model(
                protocol="hallucination",
                use_templates=False,
                num_recycles=self.num_recycles,
                data_dir=self.af_params_dir,
                use_multimer=False,
            )
            self._monomer_model._prep_hallucination(length=ab_len)
            self._monomer_len = ab_len

    def predict_complex(self, ab_seq: str) -> tuple:
        import jax
        with jax.default_device(self.jax_device):
            if not self._template_initialized:
                self._ensure_complex_model(len(ab_seq))
            self._complex_model.predict(seq=ab_seq, models=self.af_models, verbose=False)
            aux = dict(self._complex_model.aux)
            aux["log"] = dict(self._complex_model.aux.get("log", {}))
            aux["_pdb_str"] = self._complex_model.save_pdb()
            return aux, self._complex_target_len

    def predict_monomer(self, ab_seq: str) -> dict:
        import jax
        with jax.default_device(self.jax_device):
            self._ensure_monomer_model(len(ab_seq))
            self._monomer_model.set_seq(ab_seq)
            self._monomer_model.predict(models=self.af_models, verbose=False)
            aux = dict(self._monomer_model.aux)
            aux["log"] = dict(self._monomer_model.aux.get("log", {}))
            aux["_pdb_str"] = self._monomer_model.save_pdb()
            return aux


# ============================================================
# AF2 reward calculator
# ============================================================

class AbAF2RewardCal:
    """AF2-multimer reward calculator for antibody CDR design.

    Public surface matches ``ProteinEvalMetricsColabDesign``:
        ``reward_metrics(S_sp, ori_pdb_file, save_pdb, save_pdb_name, return_all_reward_term)``
        ``calc_diversity(S_sp)``
    """

    def __init__(self, args, device, result_save_folder: str = ""):
        self.args = args
        self.device = device

        self.gen_protein_folder = os.path.join(result_save_folder, "saved_proteins")
        os.makedirs(self.gen_protein_folder, exist_ok=True)

        self.metrics_name = args.reward.split(",")
        self.metrics_weight = [float(x) for x in args.reward_weight.split(",")]
        if len(self.metrics_name) != len(self.metrics_weight):
            raise ValueError(
                f"Mismatch: {len(self.metrics_name)} reward names vs "
                f"{len(self.metrics_weight)} weights in --reward / --reward_weight."
            )

        self.cdr_indices: List[int] = _parse_cdr_indices(args.cdr_indices)

        # Antigen / target.
        self.antigen_pdb = args.antigen_pdb
        self.antigen_chain = args.antigen_chain or "A"
        self.needs_complex = "iptm" in self.metrics_name
        if self.needs_complex and not self.antigen_pdb:
            raise ValueError(
                "Reward 'iptm' requires --antigen_pdb pointing at the antigen structure."
            )

        # AF2 weights / runtime.
        self.af_params_dir = os.path.expanduser(
            args.af_params_dir or os.environ.get("AF_PARAMS_DIR", "~/.mber/af_params")
        )
        self.num_recycles = int(args.num_recycles)
        self.af_models = [0]
        self.use_multimer = True

        # Bonobo-style binder templating: pre-made target+binder PDB on disk.
        # Required for binder runs (iptm metric). The template PDB must contain
        # the target on ``antigen_chain`` and the binder on chain "H".
        self.template_pdb: Optional[str] = getattr(args, "template_pdb", None) or None
        if self.needs_complex and self.template_pdb is None:
            raise ValueError(
                "Binder runs (iptm reward) require --template_pdb pointing at a "
                "pre-made multi-chain PDB containing target + binder. Generate one "
                "with ProDifEvo-Refinement/scripts/generate_template.py."
            )
        if self.template_pdb is not None and not os.path.exists(self.template_pdb):
            raise FileNotFoundError(f"--template_pdb not found: {self.template_pdb}")

        self.hotspot: Optional[str] = getattr(args, "hotspot", None) or None
        # rm_binder positions are hardcoded to match bonobo (Chothia
        # H27-H35,H48-H58,H96-H107). Bonobo doesn't expose this as a CLI knob,
        # so we don't either — it's load-bearing for parity.
        self.rm_binder_positions: str = DEFAULT_RM_BINDER_POSITIONS
        self._binder_chain = "H"

        self._serial_template_initialized = False

        # Lazy-built colabdesign model handles for the serial path.
        self._complex_model = None
        self._monomer_model = None
        self._complex_target_len = 0
        self._monomer_len = 0

        # Multi-GPU AF parallelism.
        self.af_gpu_ids = _parse_gpu_ids(getattr(args, "af_gpu_ids", ""))
        self._workers: List[_AFWorker] = []

        self._timings = {
            "n_sequences": 0,
            "reward_seconds": 0.0,
            "n_calls": 0,
        }

    # --------------------------------------------------------
    # Worker pool (multi-GPU)
    # --------------------------------------------------------
    def _ensure_workers(self) -> None:
        if self._workers or not self.af_gpu_ids:
            return
        import jax
        all_devices = jax.devices()
        try:
            picked = [all_devices[i] for i in self.af_gpu_ids]
        except IndexError as e:
            raise ValueError(
                f"--af_gpu_ids {self.af_gpu_ids} out of range; "
                f"jax.devices() has {len(all_devices)} devices."
            ) from e
        logging.info(
            f"[AF2] Building {len(picked)} AF workers on devices "
            f"{[str(d) for d in picked]}"
        )
        self._workers = [
            _AFWorker(
                jax_device=dev,
                af_params_dir=self.af_params_dir,
                num_recycles=self.num_recycles,
                use_multimer=self.use_multimer,
                af_models=self.af_models,
                antigen_pdb=self.antigen_pdb,
                antigen_chain=self.antigen_chain,
                binder_chain=self._binder_chain,
                hotspot=self.hotspot,
            )
            for dev in picked
        ]
        if self.template_pdb is not None:
            logging.info(
                f"[AF2] Templating {len(self._workers)} workers from "
                f"{self.template_pdb} (rm_binder={self.rm_binder_positions}, "
                f"hotspot={self.hotspot})"
            )
            for w in self._workers:
                w.init_template(self.template_pdb, self.rm_binder_positions)

    # --------------------------------------------------------
    # Serial-mode prediction (single-GPU / no af_gpu_ids)
    # --------------------------------------------------------
    def _ensure_serial_template(self) -> None:
        if self._serial_template_initialized:
            return
        mk_model, clear_mem = _lazy_import_colabdesign()
        if self._complex_model is not None:
            clear_mem()
        logging.info(
            f"[AF2] Serial template init from {self.template_pdb} "
            f"(rm_binder={self.rm_binder_positions}, hotspot={self.hotspot})"
        )
        self._complex_model = mk_model(
            protocol="binder",
            debug=False,
            data_dir=self.af_params_dir,
            use_multimer=self.use_multimer,
            num_recycles=self.num_recycles,
        )
        self._complex_model._prep_binder(
            pdb_filename=self.template_pdb,
            chain=self.antigen_chain,
            binder_chain=self._binder_chain,
            hotspot=self.hotspot,
            seed=0,
            rm_target=False,
            rm_target_seq=False,
            rm_target_sc=False,
            rm_template_ic=True,
            rm_binder=self.rm_binder_positions,
            rm_binder_seq=self.rm_binder_positions,
            rm_binder_sc=self.rm_binder_positions,
        )
        self._complex_target_len = int(self._complex_model._target_len)
        self._serial_template_initialized = True

    def _ensure_monomer_model(self, ab_len: int) -> None:
        mk_model, clear_mem = _lazy_import_colabdesign()
        if self._monomer_model is None or self._monomer_len != ab_len:
            if self._monomer_model is not None:
                clear_mem()
            self._monomer_model = mk_model(
                protocol="hallucination",
                use_templates=False,
                num_recycles=self.num_recycles,
                data_dir=self.af_params_dir,
                use_multimer=False,
            )
            self._monomer_model._prep_hallucination(length=ab_len)
            self._monomer_len = ab_len

    def _predict_complex(self, ab_seq: str) -> dict:
        self._ensure_serial_template()
        self._complex_model.predict(seq=ab_seq, models=self.af_models, verbose=False)
        aux = dict(self._complex_model.aux)
        aux["log"] = dict(self._complex_model.aux.get("log", {}))
        aux["_pdb_str"] = self._complex_model.save_pdb()
        return aux

    def _predict_monomer(self, ab_seq: str) -> dict:
        self._ensure_monomer_model(len(ab_seq))
        self._monomer_model.set_seq(ab_seq)
        self._monomer_model.predict(models=self.af_models, verbose=False)
        aux = dict(self._monomer_model.aux)
        aux["log"] = dict(self._monomer_model.aux.get("log", {}))
        aux["_pdb_str"] = self._monomer_model.save_pdb()
        return aux

    # --------------------------------------------------------
    # Metric calculation
    # --------------------------------------------------------
    def _metric_value(
        self,
        metric: str,
        complex_aux: Optional[dict],
        monomer_aux: Optional[dict],
        binder_offset: int,
        binder_len: int,
    ) -> float:
        conf_aux = complex_aux if complex_aux is not None else monomer_aux
        if metric == "iptm":
            if complex_aux is None:
                raise ValueError("ipTM requested but no complex prediction was run.")
            return af2_to_iptm(complex_aux)
        if metric == "ptm":
            return af2_to_ptm(conf_aux)
        if metric == "plddt":
            return af2_to_plddt(conf_aux, binder_offset=binder_offset, binder_len=binder_len)
        if metric == "cdr_plddt":
            return af2_to_cdr_plddt(conf_aux, self.cdr_indices, binder_offset=binder_offset)
        if metric == "radius":
            return af2_to_radius(conf_aux, binder_offset=binder_offset, binder_len=binder_len)
        raise NotImplementedError(f"Reward metric '{metric}' not implemented for AF2 antibody backend.")

    # --------------------------------------------------------
    # Public API: VIDD-shaped reward_metrics + calc_diversity
    # --------------------------------------------------------
    def reward_metrics(
        self,
        S_sp,
        ori_pdb_file: Optional[str] = None,
        save_pdb: bool = False,
        save_pdb_name: str = "",
        return_all_reward_term: bool = False,
        mask_for_loss=None,
    ):
        """Score a batch of tokenized antibody sequences with AF2.

        ``S_sp`` is a (bs, gen_len) integer tensor with token IDs in 0..20
        (ALPHABET ordering). ``mask_for_loss`` mirrors the RERD contract: a
        (bs, gen_len) tensor where positions == 1 are kept during decode. If
        omitted, every position is kept.

        Returns aggregates only, or (aggregates, per-metric) when
        ``return_all_reward_term`` is True — matching VIDD's existing
        ``ProteinEvalMetricsColabDesign.reward_metrics`` contract.
        """
        t0 = time.perf_counter()
        n_seqs_this_call = 0
        try:
            ab_sequences: List[str] = []
            for _it, ssp in enumerate(S_sp):
                seq_string = "".join(
                    ALPHABET[x] for _ix, x in enumerate(ssp)
                    if mask_for_loss is None or mask_for_loss[_it][_ix] == 1
                )
                ab_sequences.append(seq_string)
            n_seqs_this_call = len(ab_sequences)

            if self.af_gpu_ids and len(self.af_gpu_ids) > 1:
                agg, per_metric = self._reward_metrics_parallel(
                    ab_sequences=ab_sequences, save_pdb=save_pdb, save_pdb_name=save_pdb_name,
                )
            else:
                agg, per_metric = self._reward_metrics_serial(
                    ab_sequences=ab_sequences, save_pdb=save_pdb, save_pdb_name=save_pdb_name,
                )

            if return_all_reward_term:
                return agg, per_metric
            return agg
        finally:
            self._timings["reward_seconds"] += time.perf_counter() - t0
            self._timings["n_sequences"] += n_seqs_this_call
            self._timings["n_calls"] += 1
            # JAX device pinning + JAX/CUDA init may shift torch.cuda.current_device()
            # at the OS level. Restore so the diffusion driver's subsequent forward
            # passes don't allocate fresh tensors on the wrong device.
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.set_device(0)
            except ImportError:
                pass

    def calc_diversity(self, S_sp):
        return set_diversity(S_sp.detach().cpu().numpy())

    # --------------------------------------------------------
    # Reward orchestration
    # --------------------------------------------------------
    def _reward_metrics_serial(
        self,
        ab_sequences: List[str],
        save_pdb: bool,
        save_pdb_name: str,
    ):
        agg_list: List[float] = []
        per_metric_list: List[List[float]] = []

        for _it, ab_seq in enumerate(ab_sequences):
            complex_aux: Optional[dict] = None
            monomer_aux: Optional[dict] = None
            binder_offset = 0

            if self.needs_complex:
                complex_aux = self._predict_complex(ab_seq)
                pdb_str = complex_aux["_pdb_str"]
                binder_offset = self._complex_target_len
            else:
                monomer_aux = self._predict_monomer(ab_seq)
                pdb_str = monomer_aux["_pdb_str"]

            ab_len = len(ab_seq)
            values = [
                self._metric_value(m, complex_aux, monomer_aux, binder_offset, ab_len)
                for m in self.metrics_name
            ]
            agg = sum(v * w for v, w in zip(values, self.metrics_weight))
            agg_list.append(agg)
            per_metric_list.append(values)

            if save_pdb:
                pdb_path = os.path.join(self.gen_protein_folder, f"{save_pdb_name}_repeat{_it}.pdb")
                with open(pdb_path, "w") as f:
                    f.write(pdb_str)

        return agg_list, per_metric_list

    def _reward_metrics_parallel(
        self,
        ab_sequences: List[str],
        save_pdb: bool,
        save_pdb_name: str,
    ):
        from concurrent.futures import ThreadPoolExecutor

        self._ensure_workers()
        n_workers = len(self._workers)

        # Dispatch SHARDS (one task per worker) rather than one task per sequence.
        # Each AF model mutates internal state (model.aux) on every predict() call;
        # if two ThreadPoolExecutor threads land on the same worker concurrently,
        # one thread's aux read can capture the other thread's prediction.
        # Sharding guarantees at most one in-flight predict per worker.
        shards: List[List[int]] = [[] for _ in range(n_workers)]
        for i in range(len(ab_sequences)):
            shards[i % n_workers].append(i)

        def _shard_task(worker_idx: int, idx_list: List[int]):
            worker = self._workers[worker_idx]
            out = []
            for idx in idx_list:
                ab_seq = ab_sequences[idx]
                if self.needs_complex:
                    aux, target_len = worker.predict_complex(ab_seq)
                    out.append((idx, aux, None, target_len))
                else:
                    aux = worker.predict_monomer(ab_seq)
                    out.append((idx, None, aux, 0))
            return out

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_shard_task, w, shards[w])
                for w in range(n_workers) if shards[w]
            ]
            all_results = []
            for f in futures:
                all_results.extend(f.result())
        results = sorted(all_results, key=lambda x: x[0])

        agg_list: List[float] = []
        per_metric_list: List[List[float]] = []

        for idx, complex_aux, monomer_aux, target_len in results:
            ab_seq = ab_sequences[idx]
            ab_len = len(ab_seq)
            binder_offset = target_len if self.needs_complex else 0
            pdb_str = (complex_aux or monomer_aux)["_pdb_str"]

            values = [
                self._metric_value(m, complex_aux, monomer_aux, binder_offset, ab_len)
                for m in self.metrics_name
            ]
            agg = sum(v * w for v, w in zip(values, self.metrics_weight))
            agg_list.append(agg)
            per_metric_list.append(values)

            if save_pdb:
                pdb_path = os.path.join(self.gen_protein_folder, f"{save_pdb_name}_repeat{idx}.pdb")
                with open(pdb_path, "w") as f:
                    f.write(pdb_str)

        return agg_list, per_metric_list


# ============================================================
# Helpers
# ============================================================

def _parse_cdr_indices(spec) -> List[int]:
    if spec is None or spec == "":
        return []
    if isinstance(spec, (list, tuple)):
        return sorted(int(x) for x in spec)
    if isinstance(spec, str):
        return sorted(int(x) for x in spec.split(",") if x.strip())
    raise TypeError(f"Unsupported cdr_indices spec: {type(spec).__name__}")


def _parse_gpu_ids(spec) -> List[int]:
    if spec is None or spec == "":
        return []
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec]
    if isinstance(spec, str):
        return [int(x) for x in spec.split(",") if x.strip()]
    raise TypeError(f"Unsupported af_gpu_ids spec: {type(spec).__name__}")
