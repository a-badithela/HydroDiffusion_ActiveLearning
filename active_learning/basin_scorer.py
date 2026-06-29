"""
Runs trained model checkpoints over a set of basins and returns per-basin
predictions for use by acquisition functions. Supports encdec_lstm (deterministic)
and all diffusion model variants (diffusion_lstm, diffusion_ssm, diffusion_unet,
diffusion_ssm_unet, diffusion_ssm_lstm, decoder_only_lstm, decoder_only_ssm).
Model type and architecture params are inferred automatically from cfg.json.

Forcing note: datasets_npy.py stores 5 CAMELS variables tripled to 15 virtual
channels. There is no real multi-source forcing — the three "sources" are
identical.  We hardcode daymet (channels [2,5,8,11,14]) to match how the
SLURM jobs train the models.

Memory note: checkpoints are loaded and scored one at a time, then deleted,
so GPU memory stays bounded regardless of ensemble size.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from papercode.datasets_npy import (
    CamelsNPY,
    load_npy_data,
)
from papercode.train_npy import _build_model, load_norms

RAW_DIR = "/projects/standard/kumarv/renga/Public/DATA/camels_us_531/RAW"

# Channels in the 15-virtual-channel tensor that correspond to each forcing
# source.  Because datasets_npy.py triples the same 5 CAMELS variables, all
# three selections return identical data.  We use 'daymet' to match the
# default SLURM configuration.
_DAYMET_IDX = [2, 5, 8, 11, 14]

# Minimal cfg needed to build and run encdec_lstm.  Callers may override any key.
DEFAULT_CFG: Dict = {
    "model_name":       "encdec_lstm",
    "hidden_size":      256,
    "dropout":          0.3,
    "forecast_horizon": 8,
    "seq_length":       365,
    "forcing_source":   "daymet",   # hardcoded; see module docstring
    "concat_static":    True,
    "no_static":        False,
    "DEVICE":           "cpu",
}


class BasinScorer:
    """
    Runs an ensemble of trained checkpoints over a given set of basins.

    Args:
        checkpoint_paths: Paths to checkpoint .pt files saved by train_npy.py.
            Each must contain keys 'model' and 'ema'.
        cfg: Optional overrides for DEFAULT_CFG (model_name, hidden_size, etc.).
        device: PyTorch device string ('cpu', 'cuda:0', …).
    """

    def __init__(
        self,
        checkpoint_paths: List[str],
        cfg: Optional[Dict] = None,
        device: str = "cpu",
    ):
        self.ckpt_paths = [Path(p) for p in checkpoint_paths]

        # Three-layer config: defaults → cfg.json from checkpoint dir → caller overrides.
        # main.py writes cfg.json next to every checkpoint.pt with all training params
        # as strings. Reading it here ensures _build_model gets the right model_name and
        # architecture params without requiring callers to know them.
        self.cfg = {**DEFAULT_CFG}
        cfg_json = self.ckpt_paths[0].parent / "cfg.json"
        if cfg_json.exists():
            with open(cfg_json) as f:
                saved = json.load(f)
            # cfg.json stores every value as a string; convert back for keys that need numbers/bools.
            # SSM models (decoder_only_ssm) also need d_model/d_state/n_layers as ints and
            # all the learning-rate / S4D coupling params as floats — otherwise _build_model fails.
            _int_keys   = {"hidden_size", "forecast_horizon", "seq_length",
                           "ddim_steps", "num_samples", "time_emb_dim",
                           "batch_size", "lstm_nlayers",
                           "d_model", "d_state", "n_layers", "static_dim",
                           "warmup"}
            _float_keys = {"dropout", "ssm_dropout",
                           "lr", "lr_min", "lr_dt", "learning_rate",
                           "min_dt", "max_dt", "cfr", "cfi",
                           "wd", "weight_decay"}
            _bool_keys  = {"concat_static", "no_static"}
            for key, val in saved.items():
                if key in _int_keys:
                    self.cfg[key] = int(val)
                elif key in _float_keys:
                    self.cfg[key] = float(val)
                elif key in _bool_keys:
                    self.cfg[key] = (val.lower() == "true"
                                     if isinstance(val, str) else bool(val))
                else:
                    self.cfg[key] = val
        if cfg:
            self.cfg.update(cfg)

        self.cfg["DEVICE"] = device

        # Load raw data once — reused for every scoring call.
        data, dates, basins = load_npy_data(
            npy_path=f"{RAW_DIR}/data.npy",
            dates_path=f"{RAW_DIR}/dates.npy",
            basin_list_path=f"{RAW_DIR}/Basin_List.npy",
        )
        self._data = data
        self._dates = dates
        self._all_basins = np.array([str(b).zfill(8) for b in basins])
        # Loaded, not recomputed: must match the exact seed-only stats this
        # checkpoint trained with (train_npy.py::save_norms).
        self._scalar = load_norms(self.ckpt_paths[0].parent)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_to_basins(self, basin_ids: List[str]):
        target = set(str(b).zfill(8) for b in basin_ids)
        mask = np.array([b in target for b in self._all_basins])
        data = self._data[mask]
        basins = self._all_basins[mask]
        # Pooled seed-derived scale, not each basin's own true Q stats — basin_ids
        # here may be candidate basins with no sensor data.
        n = basins.shape[0]
        q_means = np.full((n, 1), float(self._scalar["output_mean"][0]), dtype=np.float32)
        q_stds = np.full((n, 1), float(self._scalar["output_stds"][0]), dtype=np.float32)
        return data, basins, q_means, q_stds

    def _load_model_with_ema(self, ckpt_path: Path) -> torch.nn.Module:
        device = self.cfg["DEVICE"]
        model = _build_model(self.cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
        ema.load_state_dict(ckpt["ema"])
        ema.copy_to(model.parameters())   # overwrite params with EMA-averaged values
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        basin_ids: List[str],
        split_start: str = "1990-10-01",
        split_end: str = "1995-09-30",
        stride: int = 90,
        batch_size: int = 256,
    ) -> Dict[str, np.ndarray]:
        """
        Run all checkpoints over the given basins for [split_start, split_end].

        Returns
        -------
        Dict mapping basin_id → np.ndarray of shape
        (n_models, n_windows, forecast_horizon).  Windows are in chronological
        order (DataLoader shuffle=False).
        """
        data, basins, q_means, q_stds = self._filter_to_basins(basin_ids)
        fh = self.cfg["forecast_horizon"]
        device = self.cfg["DEVICE"]

        ds = CamelsNPY(
            data=data,
            dates=self._dates,
            basins=basins,
            scalar=self._scalar,
            q_means=q_means,
            q_stds=q_stds,
            split_start=split_start,
            split_end=split_end,
            seq_length=self.cfg["seq_length"],
            forecast_horizon=fh,
            stride=stride,
            concat_static=self.cfg["concat_static"],
            no_static=self.cfg["no_static"],
            include_dates=False,
            is_train=False,
        )

        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # Mirrors the same branching in evaluate_npy.py lines 143 and 404-412.
        is_diffusion    = self.cfg["model_name"] in [
            "diffusion_lstm", "diffusion_unet", "diffusion_ssm",
            "diffusion_ssm_unet", "diffusion_ssm_lstm",
            "decoder_only_lstm", "decoder_only_ssm",
        ]
        is_decoder_only = self.cfg["model_name"] in ["decoder_only_lstm", "decoder_only_ssm"]

        # basin_model_preds[basin_id][model_idx] = list of (fh,) arrays (one per window)
        basin_model_preds: Dict[str, Dict[int, list]] = defaultdict(
            lambda: defaultdict(list)
        )

        for m_idx, ckpt_path in enumerate(self.ckpt_paths):
            print(
                f"[BasinScorer] checkpoint {m_idx + 1}/{len(self.ckpt_paths)}: "
                f"{ckpt_path.name}"
            )
            model = self._load_model_with_ema(ckpt_path)

            for batch in loader:
                if self.cfg["no_static"]:
                    x, _y, _qm, _qs, batch_basins, *_ = batch
                    static_attrs = None
                else:
                    x, static_attrs, _y, _qm, _qs, batch_basins, *_ = batch
                    static_attrs = static_attrs.to(device)

                x = x.to(device)

                # Select the 5 daymet channels from the 15-virtual-channel tensor.
                # This mirrors the idx_map logic in train_npy.validate_epoch.
                x = x[:, :, _DAYMET_IDX]   # (B, L+H, 5)

                x_past = x[:, :-fh, :]     # (B, L, 5)

                if is_diffusion:
                    # All diffusion models use H future days. Matches evaluate_npy.py:404-407.
                    x_future = x[:, -fh:, :]

                    # Concat static into x_past only for non-decoder_only variants.
                    # Matches evaluate_npy.py:409-411.
                    if (self.cfg["concat_static"]
                            and static_attrs is not None
                            and not is_decoder_only):
                        stat_p = static_attrs.unsqueeze(1).expand(-1, x_past.size(1), -1)
                        x_past = torch.cat([x_past, stat_p], dim=-1)  # (B, L, 32)

                    # stat_f is expanded over the future time dimension for both model types.
                    stat_f = (static_attrs.unsqueeze(1).expand(-1, x_future.size(1), -1)
                              if static_attrs is not None else None)

                    # eta=0 → deterministic DDIM; one call per batch is sufficient for scoring.
                    preds = model.sample_ddim(
                        x_past            = x_past,
                        static_attributes = stat_f,
                        future_pcp        = x_future,
                        num_steps         = int(self.cfg.get("ddim_steps", 3)),
                        eta               = 0.0,
                    )  # (B, H)

                else:
                    x_future = x[:, -fh:, :]   # (B, H, 5)
                    if self.cfg["concat_static"] and static_attrs is not None:
                        # static_attrs: (B, 27) → expand to (B, T, 27) and concatenate
                        stat_p = static_attrs.unsqueeze(1).expand(-1, x_past.size(1), -1)
                        stat_f = static_attrs.unsqueeze(1).expand(-1, x_future.size(1), -1)
                        x_past = torch.cat([x_past, stat_p], dim=-1)     # (B, L, 32)
                        x_future = torch.cat([x_future, stat_f], dim=-1) # (B, H, 32)

                    preds = model(x_past, x_future, None)  # (B, H, 1)
                    if preds.dim() == 3:
                        preds = preds.squeeze(-1)           # (B, H)

                preds_np = preds.cpu().numpy()
                for i, bid in enumerate(batch_basins):
                    basin_model_preds[bid][m_idx].append(preds_np[i])

            # Free GPU memory before loading the next checkpoint.
            del model
            torch.cuda.empty_cache()

        # Assemble into (n_models, n_windows, fh) per basin.
        result: Dict[str, np.ndarray] = {}
        n_models = len(self.ckpt_paths)
        for bid, model_dict in basin_model_preds.items():
            if not model_dict:
                continue
            n_windows = len(next(iter(model_dict.values())))
            arr = np.zeros((n_models, n_windows, fh), dtype=np.float32)
            for m_idx, windows in model_dict.items():
                arr[m_idx] = np.stack(windows, axis=0)
            result[bid] = arr

        # Basins that produced no windows (too-short split) get empty arrays.
        for bid in basin_ids:
            zp = str(bid).zfill(8)
            if zp not in result:
                result[zp] = np.zeros((n_models, 0, fh), dtype=np.float32)

        return result
