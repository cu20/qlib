import sys
import copy
from pathlib import Path
from typing import Optional

DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME))
sys.path.append(str(DIRNAME.parent.parent.parent))         # qlib root
sys.path.append(str(DIRNAME.parent.parent.parent.parent))  # project root (/root/WaveFormer)

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
import yaml
import argparse
import numpy as np


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="./workflow_config_waveformer_Alpha158_lowturn.yaml")
    parser.add_argument("--only_backtest", action="store_true")
    parser.add_argument("--log_dir", type=str, default="logs")

    # ablation
    parser.add_argument("--ablation", action="store_true",
                        help="run ablation: baseline vs denoise (2-way or 3-way)")
    parser.add_argument("--ablation_mode", type=str, default="2way",
                        choices=["2way", "3way"],
                        help="2way: baseline vs after-norm denoise; "
                             "3way: baseline vs before-norm denoise vs after-norm denoise")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="list of seeds for multi-seed experiments")

    # wavelet denoising knobs (used when not in ablation mode)
    parser.add_argument("--wavelet", type=str, default="haar")
    parser.add_argument("--denoise_level", type=int, default=1)
    parser.add_argument("--threshold_method", type=str, default="bayes",
                        choices=["bayes", "visu"])
    parser.add_argument("--threshold_mode", type=str, default="soft",
                        choices=["soft", "hard", "semisoft"])
    parser.add_argument("--threshold_scale", type=float, default=0.3)
    parser.add_argument("--denoise_blend", type=float, default=0.25)
    parser.add_argument("--no_denoise_finest_only", action="store_true")
    parser.add_argument("--no_level_dependent_scale", action="store_true")
    parser.add_argument("--use_edge_pad", action="store_true", default=True)
    parser.add_argument("--no_edge_pad", dest="use_edge_pad", action="store_false")
    parser.add_argument("--use_boundary_smooth", action="store_true", default=False)
    parser.add_argument("--boundary_smooth_win", type=int, default=1)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_once(config: dict, dataset, seed: int, only_backtest: bool,
             use_wavelet_denoise: bool, save_prefix_suffix: str,
             log_dir: str) -> dict:
    """
    Train (or only backtest) for a single seed and return metrics dict.

    If a saved model checkpoint already exists for this group+seed, training
    is skipped and the checkpoint is loaded directly for backtesting.
    """
    cfg = copy.deepcopy(config)
    cfg["task"]["model"]["kwargs"]["seed"] = seed
    cfg["task"]["model"]["kwargs"]["use_wavelet_denoise"] = use_wavelet_denoise

    # Each group gets its own save_prefix so model files don't collide:
    #   model/csi300_baseline_0.pkl  /  csi300_pre_norm_0.pkl  /  csi300_after_norm_0.pkl
    group = save_prefix_suffix if save_prefix_suffix else (
        "denoise" if use_wavelet_denoise else "baseline"
    )
    base_prefix = cfg["task"]["model"]["kwargs"].get("save_prefix", "csi300")
    group_prefix = f"{base_prefix}_{group}" if group else base_prefix
    cfg["task"]["model"]["kwargs"]["save_prefix"] = group_prefix

    save_dir  = Path(cfg["task"]["model"]["kwargs"].get("save_path", "model/"))
    ckpt_path = save_dir / f"{group_prefix}_{seed}.pkl"

    exp_name = f"waveformer_{group}_seed{seed}"

    with R.start(experiment_name=exp_name):
        model = init_instance_by_config(cfg["task"]["model"])

        if ckpt_path.exists():
            print(f"[run_once] Found existing checkpoint: {ckpt_path}  → skip training")
            model.load_model(str(ckpt_path))
        elif not only_backtest:
            print(f"[run_once] No checkpoint found at {ckpt_path}  → start training")
            model.fit(dataset)
        else:
            raise FileNotFoundError(
                f"--only_backtest requested but checkpoint not found: {ckpt_path}"
            )

        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        sar = SigAnaRecord(recorder, ana_long_short=False, ann_scaler=252)
        sar.generate()

        port_cfg = cfg.get("port_analysis_config") or cfg.get("task", {}).get(
            "port_analysis_config"
        )
        if port_cfg is None:
            import yaml as _yaml
            with open(args_global.config) as f:
                raw = _yaml.safe_load(f)
            port_cfg = raw.get("port_analysis_config")

        par = PortAnaRecord(recorder, config=port_cfg)
        par.generate()

        metrics = recorder.list_metrics()

    return metrics


# ---------------------------------------------------------------------------
# Group run (multiple seeds)
# ---------------------------------------------------------------------------

def run_group(config: dict, dataset, seeds: list, only_backtest: bool,
              use_wavelet_denoise: bool, save_prefix_suffix: str,
              log_dir: str) -> dict:
    """
    Run multiple seeds and return averaged metrics.
    """
    all_metrics = []
    for seed in seeds:
        m = run_once(config, dataset, seed, only_backtest,
                     use_wavelet_denoise, save_prefix_suffix, log_dir)
        all_metrics.append(m)

    # Average across seeds
    avg = {}
    keys = all_metrics[0].keys()
    for k in keys:
        vals = [m[k] for m in all_metrics if m.get(k) is not None]
        if vals:
            try:
                avg[k] = float(np.mean([float(v) for v in vals]))
            except (TypeError, ValueError):
                avg[k] = vals[-1]
    return avg


# ---------------------------------------------------------------------------
# Print ablation table
# ---------------------------------------------------------------------------

def _fmt(v):
    if v is None:
        return "  N/A  "
    try:
        return f"{float(v):+.4f}"
    except (TypeError, ValueError):
        return str(v)


def print_ablation_table(baseline: dict, after_norm: dict,
                         pre_norm: Optional[dict] = None,
                         log_dir: str = ""):
    KEYS = [
        ("IC",               "IC"),
        ("ICIR",             "ICIR"),
        ("Rank IC",          "Rank IC"),
        ("Rank ICIR",        "Rank ICIR"),
        ("1day.excess_return_with_cost.annualized_return", "Ann Ret (w/ cost)"),
        ("1day.excess_return_with_cost.information_ratio", "IR (w/ cost)"),
        ("1day.excess_return_with_cost.max_drawdown",      "Max DD (w/ cost)"),
    ]

    def _row(label, d):
        vals = [_fmt(d.get(k)) for k, _ in KEYS]
        return f"  {label:<26}" + "  ".join(vals)

    header = "  " + " " * 26 + "  ".join(f"{lbl:>12}" for _, lbl in KEYS)
    sep = "-" * len(header)

    lines = [
        "",
        "=" * 80,
        "  Ablation Results",
        "=" * 80,
        header,
        sep,
        _row("Baseline (no denoise)", baseline),
    ]
    if pre_norm is not None:
        lines.append(_row("Denoise before norm", pre_norm))
    lines.append(_row("Denoise after norm", after_norm))
    lines.append(sep)

    # delta vs baseline
    def _delta(d):
        out = []
        for k, _ in KEYS:
            try:
                out.append(f"{float(d.get(k, 0)) - float(baseline.get(k, 0)):+.4f}")
            except (TypeError, ValueError):
                out.append("  N/A  ")
        return out

    lines.append(f"  {'Delta (after_norm - baseline)':<26}" +
                 "  ".join(f"{v:>12}" for v in _delta(after_norm)))
    if pre_norm is not None:
        lines.append(f"  {'Delta (pre_norm - baseline)':<26}" +
                     "  ".join(f"{v:>12}" for v in _delta(pre_norm)))
    lines.append("=" * 80)
    lines.append("")

    output = "\n".join(lines)
    print(output)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(log_dir) / "ablation_results.txt"
        out_path.write_text(output, encoding="utf-8")
        print(f"[ablation] Results saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

args_global = None   # module-level ref for port_cfg fallback

if __name__ == "__main__":
    args = parse_args()
    args_global = args

    provider_uri = "~/.qlib/qlib_data/cn_data"
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Build dataset (and cache handler)
    # ------------------------------------------------------------------
    h_conf = config["task"]["dataset"]["kwargs"]["handler"]
    h_path = (
        DIRNAME
        / f'handler_{config["task"]["dataset"]["kwargs"]["segments"]["train"][0].strftime("%Y%m%d")}'
        f'_{config["task"]["dataset"]["kwargs"]["segments"]["test"][1].strftime("%Y%m%d")}.pkl'
    )
    if not h_path.exists():
        h = init_instance_by_config(h_conf)
        h.to_pickle(h_path, dump_all=True)
        print("Saved preprocessed handler to", h_path)
    config["task"]["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
    dataset = init_instance_by_config(config["task"]["dataset"])

    # ------------------------------------------------------------------
    # Override wavelet knobs from CLI (used when not in ablation mode)
    # ------------------------------------------------------------------
    wv_kwargs = {
        "wavelet":              args.wavelet,
        "denoise_level":        args.denoise_level,
        "threshold_method":     args.threshold_method,
        "threshold_mode":       args.threshold_mode,
        "threshold_scale":      args.threshold_scale,
        "denoise_blend":        args.denoise_blend,
        "denoise_finest_only":  not args.no_denoise_finest_only,
        "level_dependent_scale": not args.no_level_dependent_scale,
        "use_edge_pad":         args.use_edge_pad,
        "use_boundary_smooth":  args.use_boundary_smooth,
        "boundary_smooth_win":  args.boundary_smooth_win,
    }
    for k, v in wv_kwargs.items():
        config["task"]["model"]["kwargs"][k] = v

    _common = dict(
        config=config,
        seeds=args.seeds,
        only_backtest=args.only_backtest,
        log_dir=args.log_dir,
    )

    # ------------------------------------------------------------------
    # Ablation mode
    # ------------------------------------------------------------------
    if args.ablation:
        # --- Group 1: baseline (no denoising) ---
        baseline_metrics = run_group(
            dataset=dataset,
            use_wavelet_denoise=False,
            save_prefix_suffix="baseline",
            **_common,
        )

        pre_norm_metrics = None

        # --- Group 2 (3-way only): denoise before normalisation ---
        if args.ablation_mode == "3way":
            # Inject WaveletDenoiseProcessor into infer_processors before RobustZScoreNorm
            h_conf_orig = copy.deepcopy(
                config["task"]["dataset"]["kwargs"].get("_h_conf_orig", h_conf)
            )
            # Build a modified handler config with WaveletDenoiseProcessor prepended
            h_pre_conf = copy.deepcopy(h_conf_orig)
            pre_proc = {
                "class": "WaveletDenoiseProcessor",
                "module_path": "wavelet_processor",
                "kwargs": {
                    "level": 1,
                    "threshold_method": "bayes",
                    "threshold_scale": 0.5,
                    "blend": 1.0,
                    "finest_only": True,
                },
            }
            existing = h_pre_conf.get("kwargs", {}).get("infer_processors", [])
            h_pre_conf.setdefault("kwargs", {})["infer_processors"] = [pre_proc] + existing

            # Build pre-norm dataset with a separate handler cache
            h_pre_path = DIRNAME / (h_path.stem + "_pre_norm.pkl")
            if not h_pre_path.exists():
                h_pre = init_instance_by_config(h_pre_conf)
                h_pre.to_pickle(h_pre_path, dump_all=True)
                print("Saved pre-norm handler to", h_pre_path)

            config_pre = copy.deepcopy(config)
            config_pre["task"]["dataset"]["kwargs"]["handler"] = f"file://{h_pre_path}"
            dataset_pre_norm = init_instance_by_config(config_pre["task"]["dataset"])

            pre_norm_metrics = run_group(
                config=config_pre,
                dataset=dataset_pre_norm,
                use_wavelet_denoise=False,
                save_prefix_suffix="pre_norm",
                seeds=args.seeds,
                only_backtest=args.only_backtest,
                log_dir=args.log_dir,
            )

        # --- Group 3: denoise after normalisation (in-model) ---
        after_norm_metrics = run_group(
            dataset=dataset,
            use_wavelet_denoise=True,
            save_prefix_suffix="after_norm",
            **_common,
        )

        print_ablation_table(
            baseline=baseline_metrics,
            after_norm=after_norm_metrics,
            pre_norm=pre_norm_metrics,
            log_dir=args.log_dir,
        )

    # ------------------------------------------------------------------
    # Normal single run
    # ------------------------------------------------------------------
    else:
        run_group(
            dataset=dataset,
            use_wavelet_denoise=config["task"]["model"]["kwargs"].get(
                "use_wavelet_denoise", False
            ),
            save_prefix_suffix="",
            **_common,
        )
