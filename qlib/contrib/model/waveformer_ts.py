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

# 将工程根目录 (/root/WaveFormer) 加入 sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from WaveFormer import WaveFormerModel


class WaveFormerQlibModel(Model):
    """
    将仓库里的 WaveFormerModel 封装成 qlib 可调用的 Model：
    - fit(self, dataset: DatasetH)
    - predict(self, dataset: DatasetH) -> pd.Series
    """

    def __init__(
        self,
        d_feat: int = 158,
        d_model: int = 256,
        num_label_heads: int = 1,
        head_loss_weights=None,
        primary_head_index: int = 0,
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
        print_valid_ic: bool = False,
        use_valid_for_ckpt_selection: bool = False,
        use_master_label_process: bool = False,
        use_master_valid_loss: bool = False,
        save_master_suffix: bool = False,
        # wavelet denoising config (GPU-native, inside model)
        use_wavelet_denoise: bool = False,
        wavelet: str = "haar",
        denoise_level: Optional[int] = 1,
        threshold_method: str = "bayes",
        threshold_mode: str = "soft",
        threshold_scale: float = 0.3,
        denoise_blend: float = 0.25,
        adaptive_blend: bool = False,
        adaptive_target_ratio: float = 0.08,
        adaptive_sharpness: float = 12.0,
        adaptive_blend_min: float = 0.03,
        adaptive_blend_max: float = 0.55,
        denoise_finest_only: bool = True,
        level_dependent_scale: bool = True,
        use_edge_pad: bool = True,
        use_boundary_smooth: bool = False,
        boundary_smooth_win: int = 1,
        wavelet_collect_stats: bool = True,
    ):
        self.inner = WaveFormerModel(
            d_feat=d_feat,
            d_model=d_model,
            num_label_heads=num_label_heads,
            head_loss_weights=head_loss_weights,
            primary_head_index=primary_head_index,
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
            log_valid_ic=print_valid_ic,
            use_valid_for_ckpt_selection=use_valid_for_ckpt_selection,
            use_master_label_process=use_master_label_process,
            use_master_valid_loss=use_master_valid_loss,
            save_master_suffix=save_master_suffix,
            use_wavelet_denoise=use_wavelet_denoise,
            wavelet=wavelet,
            denoise_level=denoise_level,
            threshold_method=threshold_method,
            threshold_mode=threshold_mode,
            threshold_scale=threshold_scale,
            denoise_blend=denoise_blend,
            adaptive_blend=adaptive_blend,
            adaptive_target_ratio=adaptive_target_ratio,
            adaptive_sharpness=adaptive_sharpness,
            adaptive_blend_min=adaptive_blend_min,
            adaptive_blend_max=adaptive_blend_max,
            denoise_finest_only=denoise_finest_only,
            level_dependent_scale=level_dependent_scale,
            use_edge_pad=use_edge_pad,
            use_boundary_smooth=use_boundary_smooth,
            boundary_smooth_win=boundary_smooth_win,
            wavelet_collect_stats=wavelet_collect_stats,
        )
        self.fitted = False

    def _init_data_loader(self, data, shuffle: bool = True, drop_last: bool = True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        return torch.utils.data.DataLoader(data, sampler=sampler, drop_last=drop_last)

    def fit(self, dataset: DatasetH):
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        # Follow upstream MASTER URL logic: train-only loop (no valid-set selection path).
        self.inner.fit(dl_train, None)

        self.fitted = True

    def predict(self, dataset: DatasetH):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        pred_series, _metrics = self.inner.predict(dl_test)

        if isinstance(pred_series, pd.DataFrame) and pred_series.shape[1] == 1:
            pred_series = pred_series.iloc[:, 0]
        return pred_series

    def load_model(self, param_path: str):
        self.inner.load_param(param_path)
        self.fitted = True
