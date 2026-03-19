import numpy as np
import pandas as pd
from typing import Optional
import os
import sys
from pathlib import Path

import torch

from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.base import Model
from .pytorch_master_ts import DailyBatchSamplerRandom

# Add project root so top-level WaveFormer.py / wavelet_gpu.py are importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from WaveFormer import WaveFormerModel


class WaveFormerQlibModel(Model):
    """
    Wraps WaveFormerModel (from the project root) as a qlib-compatible Model,
    exposing fit() / predict() for SignalRecord / PortAnaRecord.

    Wavelet denoising is now a GPU-native layer *inside* WaveFormer itself
    (see wavelet_gpu.GpuWaveletDenoiser).  The old CPU-based WaveletDenoiser
    is no longer used here.
    """

    def __init__(
        self,
        d_feat: int = 158,
        d_model: int = 256,
        t_nhead: int = 4,
        s_nhead: int = 2,
        gate_input_start_index: int = 158,
        gate_input_end_index: int = 221,
        T_dropout_rate: float = 0.5,
        S_dropout_rate: float = 0.5,
        beta: float = 5.0,
        n_epochs: int = 5,
        lr: float = 1e-5,
        GPU: Optional[int] = 0,
        seed: int = 0,
        train_stop_loss_thred: float = 0.95,
        save_path: str = "model/",
        save_prefix: str = "waveformer_",
        # GPU-native wavelet denoising parameters (optimized: BayesShrink + light blend)
        use_wavelet_denoise: bool = False,
        wavelet: str = "haar",
        denoise_level: Optional[int] = 1,
        threshold_mode: str = "soft",
        threshold_method: str = "bayes",
        threshold_scale: float = 0.3,
        denoise_blend: float = 0.25,
        denoise_finest_only: bool = True,
        level_dependent_scale: bool = True,
        use_edge_pad: bool = True,
        use_boundary_smooth: bool = False,
        boundary_smooth_win: int = 1,
    ):
        self.inner = WaveFormerModel(
            d_feat=d_feat,
            d_model=d_model,
            t_nhead=t_nhead,
            s_nhead=s_nhead,
            T_dropout_rate=T_dropout_rate,
            S_dropout_rate=S_dropout_rate,
            beta=beta,
            gate_input_start_index=gate_input_start_index,
            gate_input_end_index=gate_input_end_index,
            n_epochs=n_epochs,
            lr=lr,
            GPU=GPU,
            seed=seed,
            train_stop_loss_thred=train_stop_loss_thred,
            save_path=save_path,
            save_prefix=save_prefix,
            # GPU wavelet denoising is part of the model graph
            use_wavelet_denoise=use_wavelet_denoise,
            wavelet=wavelet,
            denoise_level=denoise_level,
            threshold_mode=threshold_mode,
            threshold_method=threshold_method,
            threshold_scale=threshold_scale,
            denoise_blend=denoise_blend,
            denoise_finest_only=denoise_finest_only,
            level_dependent_scale=level_dependent_scale,
            use_edge_pad=use_edge_pad,
            use_boundary_smooth=use_boundary_smooth,
            boundary_smooth_win=boundary_smooth_win,
        )
        self.fitted = False

    # ------------------------------------------------------------------
    # qlib Model interface
    # ------------------------------------------------------------------

    def fit(self, dataset: DatasetH):
        dl_train = dataset.prepare("train", col_set=["feature", "label"],
                                   data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"],
                                   data_key=DataHandlerLP.DK_L)

        self.inner.fit(dl_train, dl_valid)

        # Always persist a copy so --only_backtest can reload without retraining
        try:
            save_dir = Path(self.inner.save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{self.inner.save_prefix}_{self.inner.seed}.pkl"
            ckpt = save_dir / fname
            torch.save(self.inner.model.state_dict(), ckpt)
            print(f"[WaveFormerQlibModel] Model saved → {ckpt}")
        except Exception as e:
            print(f"[WaveFormerQlibModel] Warning: could not save model: {e}")

        self.fitted = True

    def predict(self, dataset: DatasetH) -> pd.Series:
        if not self.fitted:
            raise ValueError("Model is not fitted yet. Call fit() or load_model() first.")

        dl_test = dataset.prepare("test", col_set=["feature", "label"],
                                  data_key=DataHandlerLP.DK_I)
        pred_series, metrics = self.inner.predict(dl_test)

        print(
            f"[WaveFormerQlibModel] Test  IC={metrics['IC']:.4f}  "
            f"ICIR={metrics['ICIR']:.3f}  "
            f"RIC={metrics['RIC']:.4f}  "
            f"RICIR={metrics['RICIR']:.3f}"
        )

        if isinstance(pred_series, pd.DataFrame) and pred_series.shape[1] == 1:
            pred_series = pred_series.iloc[:, 0]
        return pred_series

    # ------------------------------------------------------------------
    # Convenience: load pre-trained parameters (for --only_backtest)
    # ------------------------------------------------------------------

    def load_model(self, param_path: str):
        if not os.path.exists(param_path):
            raise FileNotFoundError(
                f"[WaveFormerQlibModel] Model file not found: {param_path}\n"
                "Run training first (without --only_backtest)."
            )
        self.inner.load_param(param_path)
        print(f"[WaveFormerQlibModel] Parameters loaded from {param_path}")
        self.fitted = True
