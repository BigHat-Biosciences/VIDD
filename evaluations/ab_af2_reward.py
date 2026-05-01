"""AlphaFold2.3-multimer reward backend for antibody (VHH) CDR design in VIDD.

Ported from ProDifEvo-Refinement/ab_af2_reward.py and adapted to VIDD's
``reward_metrics`` calling convention used by ``ProteinEvalMetricsColabDesign``.

Backend choices:
* ``mk_afdesign_model(protocol="binder", use_multimer=True)`` for predicting the
  antibody:antigen complex (used when ``iptm`` is in the metric list). The
  antigen PDB is supplied as the target; the antibody is the binder of length
  ``args.gen_len``.
* ``mk_afdesign_model(protocol="hallucination", use_multimer=False)`` for the
  monomer fallback when no antigen / iptm is requested.

Differences vs the ProDifEvo-Refinement version:
* Public surface mirrors ``ProteinEvalMetricsColabDesign`` so it slots into
  VIDD's ``initialize_eval_model`` without changes to the diffusion driver:
    reward_metrics(S_sp, ori_pdb_file=None, save_pdb=False,
                   save_pdb_name="", return_all_reward_term=False)
      -> record_reward                  if return_all_reward_term is False
      -> (record_reward, each_reward)   otherwise
  ``record_reward`` is a list[float] of weighted aggregates; ``each_reward``
  is a list[list[float]] in the order of ``args.reward.split(",")``.
* Decodes ``S_sp`` token tensors with VIDD's ALPHABET ordering
  ('ACDEFGHIKLMNPQRSTVWYX'), matching ``protein_eval_bind_colabdesign.py``.
* Carries a ``calc_diversity`` passthrough.
* Keeps the ``_timings`` dict, multi-GPU AF worker pool, NBB2 templating, and
  lazy colabdesign import from the original.
"""

from __future__ import annotations

import os
import logging
import tempfile
import time
from typing import List, Optional, Sequence

import numpy as np

from evaluations.protein_utils import set_diversity


# Lazy module handles: colabdesign / NBB2 are heavy and may not be importable
# in non-AF2 environments. Defer to first use.
_AF_FACTORY = None
_CLEAR_MEM = None
_NBB2_CLS = None


def _lazy_import_colabdesign():
    """Import colabdesign on first use, surfacing a clear install hint if missing."""
    global _AF_FACTORY, _CLEAR_MEM
    if _AF_FACTORY is not None:
        return _AF_FACTORY, _CLEAR_MEM
    try:
        from colabdesign import mk_afdesign_model, clear_mem
    except ImportError as e:
        raise ImportError(
            "colabdesign is required for the AF2 antibody reward backend. Install with:\n"
            "  pip install 'colabdesign @ git+https://github.com/sokrypton/ColabDesign.git@d024c4e'\n"
            "and ensure jax/flax are installed."
        ) from e
    _AF_FACTORY = mk_afdesign_model
    _CLEAR_MEM = clear_mem
    return _AF_FACTORY, _CLEAR_MEM


def _lazy_import_nbb2():
    """Import NanoBodyBuilder2 on first use."""
    global _NBB2_CLS
    if _NBB2_CLS is not None:
        return _NBB2_CLS
    try:
        from ImmuneBuilder import NanoBodyBuilder2
    except ImportError as e:
        raise ImportError(
            "ImmuneBuilder is required for --use_template (NanoBodyBuilder2 binder pre-folding). "
            "Install with: pip install ImmuneBuilder"
        ) from e
    _NBB2_CLS = NanoBodyBuilder2
    return _NBB2_CLS


# VIDD's S_sp tensor uses indices 0-19 over 20 standard AAs in alphabetical
# order, with index 20 = 'X'. This matches protein_eval_bind_colabdesign.py.
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


# ============================================================
# PDB combine helper (target + NBB2-folded binder -> multi-chain PDB)
# ============================================================

def _combine_target_and_binder_pdb(
    target_pdb_path: str,
    binder_pdb_str: str,
    target_chain: str,
    binder_chain: str = "H",
) -> str:
    """Concatenate target ATOMs and binder ATOMs into a single multi-chain PDB string.

    Vendored from mber-open's pdb_utils.combine_structures, simplified to operate
    on a target PDB path + a binder PDB string. The binder chain is rewritten to
    ``binder_chain`` (default 'H'); the target keeps its existing chain ID.
    """
    with open(target_pdb_path, "r") as f:
        target_pdb = f.read()

    lines = ["HEADER    PROTEIN", "TITLE     COMBINED TARGET+BINDER (NBB2 TEMPLATE)"]
    atom_count = 1

    for line in target_pdb.splitlines():
        if line.startswith("ATOM"):
            lines.append(f"ATOM  {atom_count:5d}{line[11:]}")
            atom_count += 1
    lines.append(f"TER   {atom_count:5d}      {target_chain}")

    binder_atoms = 0
    for line in binder_pdb_str.splitlines():
        if line.startswith("ATOM"):
            new_line = f"ATOM  {atom_count:5d}{line[11:21]}{binder_chain}{line[22:]}"
            lines.append(new_line)
            atom_count += 1
            binder_atoms += 1
    if binder_atoms > 0:
        lines.append(f"TER   {atom_count:5d}      {binder_chain}")
    lines.append("END")
    return "\n".join(lines)


# ============================================================
# Metric extraction helpers (operate on colabdesign ``aux`` dict)
# ============================================================

def _get_log(aux: dict) -> dict:
    return aux.get("log", {})


def _per_residue_plddt(aux: dict) -> np.ndarray:
    """Return per-residue pLDDT as a 1D float array on [0, 100]."""
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
    raise KeyError(
        "AF2 aux['log'] does not contain i_ptm. Ensure use_multimer=True and that "
        "the binder protocol was used for complex prediction."
    )


def af2_to_plddt(aux: dict, binder_offset: int = 0, binder_len: Optional[int] = None) -> float:
    """Mean pLDDT over the binder slice, scaled to [0, 1]."""
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
    """Mean pLDDT over CDR positions in the binder, scaled to [0, 1]."""
    pl = _per_residue_plddt(aux)
    if len(cdr_indices) == 0:
        return 0.0
    cdr_global = np.asarray(cdr_indices, dtype=int) + binder_offset
    cdr_global = cdr_global[cdr_global < pl.size]
    if cdr_global.size == 0:
        return 0.0
    return float(pl[cdr_global].mean()) / 100.0


def af2_to_radius(aux: dict, binder_offset: int = 0, binder_len: Optional[int] = None) -> float:
    """Radius-of-gyration penalty from VIDD's protein_eval_bind_colabdesign.

    Returns the *positive* rg value (callers can negate via reward weights to
    turn it into a maximization target, matching the convention in VIDD scripts
    where ``radius`` weight is small and negative-acting through ``rg_value_max``).
    """
    import jax  # local — only needed for elu fallback
    import jax.numpy as jnp
    ca = np.asarray(aux["atom_positions"])  # [L, 37, 3]
    # CA index in residue_constants.atom_order is 1
    ca = ca[:, 1, :]
    if binder_len is not None:
        ca = ca[binder_offset:binder_offset + binder_len]
    else:
        ca = ca[binder_offset:]
    if ca.shape[0] == 0:
        return 0.0
    rg = float(jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8))
    rg_th = 2.38 * (ca.shape[0] ** 0.365)
    rg_value = float(jax.nn.elu(rg - rg_th))
    return rg_value


# ============================================================
# Per-GPU AF worker for multi-GPU parallelism
# ============================================================

class _AFWorker:
    """A single AF2 model instance pinned to one JAX device."""

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
    ):
        self.jax_device = jax_device
        self.af_params_dir = af_params_dir
        self.num_recycles = num_recycles
        self.use_multimer = use_multimer
        self.af_models = list(af_models)
        self.antigen_pdb = antigen_pdb
        self.antigen_chain = antigen_chain
        self.binder_chain = binder_chain

        self._complex_model = None
        self._complex_target_len = 0
        self._complex_binder_len = 0
        self._template_initialized = False
        self._monomer_model = None
        self._monomer_len = 0

    def _ensure_complex_model(self, ab_len: int) -> None:
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
                hotspot=None,
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

    def init_template(self, combined_pdb_path: str, rm_binder_str: str, ab_len: int) -> None:
        """One-shot template init: build the binder model on the combined PDB
        and bake in the CDR mask via ``rm_binder``. Subsequent calls to
        ``predict_complex`` are pure forward passes."""
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
                pdb_filename=combined_pdb_path,
                chain=self.antigen_chain,
                binder_chain=self.binder_chain,
                hotspot=None,
                seed=0,
                rm_target=False,
                rm_target_seq=False,
                rm_target_sc=False,
                rm_template_ic=True,
                rm_binder=rm_binder_str,
                rm_binder_seq=True,
                rm_binder_sc=True,
            )
            self._complex_binder_len = ab_len
            self._complex_target_len = int(self._complex_model._target_len)
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
# AF2 antibody reward calculator (VIDD-shaped public surface)
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

        # Reward metrics + weights (same parsing as ProteinEvalMetricsColabDesign).
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

        # NBB2 templating.
        self.use_template = bool(args.use_template)
        if self.use_template and not self.needs_complex:
            raise ValueError(
                "--use_template requires complex prediction (i.e. 'iptm' in --reward)."
            )
        self.nbb2_weights_dir = os.path.expanduser(
            args.nbb2_weights_dir or os.environ.get("NBB2_WEIGHTS_DIR", "~/.mber/nbb2_weights")
        )
        self._binder_chain = "H"

        self.seed_sequence = args.antibody_sequence
        if self.use_template and not self.seed_sequence:
            raise ValueError(
                "--use_template requires --antibody_sequence (NBB2 folds it once at startup)."
            )

        self._template_initialized_serial = False

        # Lazy-built colabdesign model handles for the serial path.
        self._complex_model = None
        self._monomer_model = None
        self._complex_target_len = 0
        self._complex_binder_len = 0
        self._monomer_len = 0

        # Lazy NBB2 model handle.
        self._nbb2_model = None

        # Multi-GPU AF parallelism.
        self.af_gpu_ids = _parse_gpu_ids(getattr(args, "af_gpu_ids", ""))
        self._workers: List[_AFWorker] = []

        # Cumulative timing/counts. Updated on every reward_metrics call.
        self._timings = {
            "n_sequences": 0,
            "reward_seconds": 0.0,
            "n_calls": 0,
        }

    # --------------------------------------------------------
    # Model construction
    # --------------------------------------------------------
    def _ensure_complex_model(self, ab_len: int) -> None:
        mk_model, clear_mem = _lazy_import_colabdesign()
        if self._complex_model is None or self._complex_binder_len != ab_len:
            if self._complex_model is not None:
                clear_mem()
            logging.info(
                f"[AF2] Building binder/multimer model (params={self.af_params_dir}, "
                f"recycles={self.num_recycles}, ab_len={ab_len})"
            )
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
                hotspot=None,
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

    def _ensure_monomer_model(self, ab_len: int) -> None:
        mk_model, clear_mem = _lazy_import_colabdesign()
        if self._monomer_model is None or self._monomer_len != ab_len:
            if self._monomer_model is not None:
                clear_mem()
            logging.info(
                f"[AF2] Building hallucination/monomer model (recycles={self.num_recycles}, ab_len={ab_len})"
            )
            self._monomer_model = mk_model(
                protocol="hallucination",
                use_templates=False,
                num_recycles=self.num_recycles,
                data_dir=self.af_params_dir,
                use_multimer=False,
            )
            self._monomer_model._prep_hallucination(length=ab_len)
            self._monomer_len = ab_len

    def _ensure_workers(self) -> None:
        """Lazy-build a pool of _AFWorker instances pinned to ``af_gpu_ids``.

        When ``use_template`` is set, also runs the one-shot template init on
        each worker (NBB2-fold the seed once, combine with antigen, then init).
        """
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
            )
            for dev in picked
        ]

        if self.use_template:
            combined_pdb = self._build_template_pdb()
            rm_binder_str = self._cdr_position_string()
            ab_len = len(self.seed_sequence)
            try:
                logging.info(
                    f"[AF2] Templating {len(self._workers)} workers (rm_binder={rm_binder_str})"
                )
                for w in self._workers:
                    w.init_template(combined_pdb, rm_binder_str, ab_len)
            finally:
                try:
                    os.remove(combined_pdb)
                except OSError:
                    pass

    def _build_template_pdb(self) -> str:
        if not self.seed_sequence:
            raise ValueError("Templating requires a seed sequence.")
        binder_pdb_str = self._fold_binder_with_nbb2(self.seed_sequence)
        combined_pdb_str = _combine_target_and_binder_pdb(
            target_pdb_path=self.antigen_pdb,
            binder_pdb_str=binder_pdb_str,
            target_chain=self.antigen_chain,
            binder_chain=self._binder_chain,
        )
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp:
            tmp.write(combined_pdb_str)
            return tmp.name

    def _cdr_position_string(self) -> str:
        if not self.cdr_indices:
            return ""
        return ",".join(f"{self._binder_chain}{i + 1}" for i in self.cdr_indices)

    def _ensure_nbb2(self) -> None:
        if self._nbb2_model is not None:
            return
        NBB2 = _lazy_import_nbb2()
        os.makedirs(self.nbb2_weights_dir, exist_ok=True)
        logging.info(f"[NBB2] Loading NanoBodyBuilder2 from {self.nbb2_weights_dir}")
        self._nbb2_model = NBB2(numbering_scheme="raw", weights_dir=self.nbb2_weights_dir)

    def _init_serial_template(self) -> None:
        """One-shot template init for the serial-mode AF model."""
        if self._template_initialized_serial:
            return
        mk_model, clear_mem = _lazy_import_colabdesign()
        if self._complex_model is not None:
            clear_mem()

        ab_len = len(self.seed_sequence)
        rm_binder_str = self._cdr_position_string()
        logging.info(
            f"[AF2] One-shot template init (serial): ab_len={ab_len}, "
            f"rm_binder={rm_binder_str}"
        )
        self._complex_model = mk_model(
            protocol="binder",
            debug=False,
            data_dir=self.af_params_dir,
            use_multimer=self.use_multimer,
            num_recycles=self.num_recycles,
        )

        combined_pdb = self._build_template_pdb()
        try:
            self._complex_model._prep_binder(
                pdb_filename=combined_pdb,
                chain=self.antigen_chain,
                binder_chain=self._binder_chain,
                hotspot=None,
                seed=0,
                rm_target=False,
                rm_target_seq=False,
                rm_target_sc=False,
                rm_template_ic=True,
                rm_binder=rm_binder_str,
                rm_binder_seq=True,
                rm_binder_sc=True,
            )
        finally:
            try:
                os.remove(combined_pdb)
            except OSError:
                pass

        self._complex_binder_len = ab_len
        self._complex_target_len = int(self._complex_model._target_len)
        self._template_initialized_serial = True

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------
    def _fold_binder_with_nbb2(self, ab_seq: str) -> str:
        self._ensure_nbb2()
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            import torch
            with torch.no_grad():
                nb = self._nbb2_model.predict({"H": ab_seq})
            nb.save(tmp_path)
            with open(tmp_path, "r") as f:
                return f.read()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _predict_complex(self, ab_seq: str) -> dict:
        if self.use_template:
            self._init_serial_template()
            self._complex_model.predict(seq=ab_seq, models=self.af_models, verbose=False)
        else:
            self._ensure_complex_model(len(ab_seq))
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
    ):
        """Score a batch of tokenized antibody sequences with AF2.

        ``S_sp`` is a (bs, gen_len) integer tensor with token IDs in 0..20
        (ALPHABET ordering). Returns aggregates only, or (aggregates, per-metric)
        when ``return_all_reward_term`` is True — matching VIDD's existing
        ``ProteinEvalMetricsColabDesign.reward_metrics`` contract.
        """
        t0 = time.perf_counter()
        n_seqs_this_call = 0
        try:
            ab_sequences = _decode_sequences(S_sp)
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

        def _task(idx: int):
            worker = self._workers[idx % n_workers]
            ab_seq = ab_sequences[idx]
            if self.needs_complex:
                aux, target_len = worker.predict_complex(ab_seq)
                return idx, aux, None, target_len
            aux = worker.predict_monomer(ab_seq)
            return idx, None, aux, 0

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_task, i) for i in range(len(ab_sequences))]
            results = sorted([f.result() for f in futures], key=lambda x: x[0])

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

def _decode_sequences(S_sp) -> List[str]:
    """Decode VIDD's (bs, gen_len) token tensor to AA strings using ALPHABET.

    Out-of-range tokens (e.g. mask) decode to 'X' so that AF2 still gets a
    valid amino-acid string; in CDR-only design those positions will already
    have been filled by the diffusion sampler before reward_metrics is called.
    """
    out: List[str] = []
    for ssp in S_sp:
        chars: List[str] = []
        for x in ssp:
            xi = int(x)
            chars.append(ALPHABET[xi] if 0 <= xi < len(ALPHABET) else "X")
        out.append("".join(chars))
    return out


def _parse_cdr_indices(spec) -> List[int]:
    """Parse --cdr_indices (csv str / list / None) into a sorted list of 0-based ints."""
    if spec is None or spec == "":
        return []
    if isinstance(spec, (list, tuple)):
        return sorted(int(x) for x in spec)
    if isinstance(spec, str):
        return sorted(int(x) for x in spec.split(",") if x.strip())
    raise TypeError(f"Unsupported cdr_indices spec: {type(spec).__name__}")


def _parse_gpu_ids(spec) -> List[int]:
    """Parse --af_gpu_ids ('' / '1,2,3' / list) into a list of ints."""
    if spec is None or spec == "":
        return []
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec]
    if isinstance(spec, str):
        return [int(x) for x in spec.split(",") if x.strip()]
    raise TypeError(f"Unsupported af_gpu_ids spec: {type(spec).__name__}")
