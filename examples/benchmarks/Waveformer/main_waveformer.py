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
import json


# 与 3-way 消融中 pre_norm 组一致（归一化前 CPU 小波去噪）
# blend 与 wavelet_gpu 一致: out = (1-blend)*raw + blend*denoised
# 默认此前 blend=1.0 会几乎完全替换为去噪后序列，易压掉有效 alpha；默认改为轻量混合。
_WAVELET_PRE_NORM_BASE = {
    "class": "WaveletDenoiseProcessor",
    "module_path": "wavelet_processor",
    "kwargs": {
        "level": 1,
        "threshold_method": "bayes",
        "threshold_scale": 0.35,
        "blend": 0.22,
        "finest_only": True,
    },
}


def _make_wavelet_pre_norm_proc(override: Optional[dict] = None) -> dict:
    proc = copy.deepcopy(_WAVELET_PRE_NORM_BASE)
    if override:
        proc["kwargs"].update(override)
    return proc


def _inject_wavelet_pre_norm(
    h_conf: dict, proc: Optional[dict] = None
) -> dict:
    if proc is None:
        proc = _make_wavelet_pre_norm_proc()
    hc = copy.deepcopy(h_conf)
    procs = list(hc.get("kwargs", {}).get("infer_processors", []))
    if procs and isinstance(procs[0], dict) and procs[0].get("class") == "WaveletDenoiseProcessor":
        return hc
    hc.setdefault("kwargs", {})["infer_processors"] = [proc] + procs
    return hc


def _handler_pickle_path(train_start, test_end, suffix: str = "") -> Path:
    t0 = train_start.strftime("%Y%m%d")
    t1 = test_end.strftime("%Y%m%d")
    return DIRNAME / f"handler_{t0}_{t1}{suffix}.pkl"


def _build_dataset_from_handler_conf(full_config: dict, h_conf: dict, pkl_path: Path):
    if not pkl_path.exists():
        h = init_instance_by_config(h_conf)
        h.to_pickle(pkl_path, dump_all=True)
        print("Saved preprocessed handler to", pkl_path)
    cfg = copy.deepcopy(full_config)
    cfg["task"]["dataset"]["kwargs"]["handler"] = f"file://{pkl_path}"
    return init_instance_by_config(cfg["task"]["dataset"])


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="./workflow_config_waveformer_Alpha158.yaml")
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

    # 若未传参，则保留 YAML 里 task.model.kwargs 的 wavelet 设置（不强行覆盖）
    parser.add_argument("--wavelet", type=str, default=None)
    parser.add_argument("--denoise_level", type=int, default=None)
    parser.add_argument("--threshold_method", type=str, default=None)
    parser.add_argument("--threshold_mode", type=str, default=None)
    parser.add_argument("--threshold_scale", type=float, default=None)
    parser.add_argument("--denoise_blend", type=float, default=None)
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
             use_wavelet_denoise: bool, save_prefix_suffix: Optional[str],
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
    #
    # In normal single-run mode we pass save_prefix_suffix=None to keep raw
    # YAML save_prefix unchanged (e.g. directly evaluate model/csi300_best_0.pkl
    # with a pre-norm dataset pipeline, without creating *_baseline suffix).
    if save_prefix_suffix is None:
        group = ""
    else:
        group = save_prefix_suffix if save_prefix_suffix else (
            "after_norm" if use_wavelet_denoise else "baseline"
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

    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        out_file = log_path / f"{group_prefix}_seed{seed}_metrics.json"
        payload = {
            "experiment_name": exp_name,
            "group": group,
            "group_prefix": group_prefix,
            "seed": seed,
            "checkpoint": str(ckpt_path),
            "only_backtest": bool(only_backtest),
            "metrics": metrics,
        }
        out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"[run_once] Metrics saved to {out_file}")

    return metrics


# ---------------------------------------------------------------------------
# Group run (multiple seeds)
# ---------------------------------------------------------------------------

def run_group(config: dict, dataset, seeds: list, only_backtest: bool,
              use_wavelet_denoise: bool, save_prefix_suffix: Optional[str],
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
        ("1day.excess_return_without_cost.annualized_return", "Ann Ret (w/o cost)"),
        ("1day.excess_return_without_cost.information_ratio", "IR (w/o cost)"),
        ("1day.excess_return_without_cost.max_drawdown",      "Max DD (w/o cost)"),
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
    if args.threshold_method is not None and args.threshold_method not in ("bayes", "visu"):
        raise ValueError("--threshold_method must be bayes or visu")
    if args.threshold_mode is not None and args.threshold_mode not in ("soft", "hard", "semisoft"):
        raise ValueError("--threshold_mode must be soft, hard, or semisoft")
    args_global = args

    provider_uri = "~/.qlib/qlib_data/cn_data"
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 可选：覆盖 WaveletDenoiseProcessor 的 kwargs（仅 pre_norm 时生效，见 _WAVELET_PRE_NORM_BASE）
    pre_norm_override = config["task"].pop("pre_norm_wavelet", None)
    pre_norm_proc = _make_wavelet_pre_norm_proc(pre_norm_override)

    # 非 Dataset 参数：从 kwargs 取出，避免传入 MASTERTSDatasetH 报错
    use_pre_flag = config["task"]["dataset"]["kwargs"].pop("use_pre_norm_wavelet", False)

    segs = config["task"]["dataset"]["kwargs"]["segments"]
    tr0, te1 = segs["train"][0], segs["test"][1]

    # 无 WaveletDenoiseProcessor 的 handler（消融 baseline / after_norm 共用）
    h_baseline_conf = copy.deepcopy(config["task"]["dataset"]["kwargs"]["handler"])
    config["_h_conf_orig"] = copy.deepcopy(h_baseline_conf)

    p_baseline = _handler_pickle_path(tr0, te1, "")
    dataset_baseline = _build_dataset_from_handler_conf(config, h_baseline_conf, p_baseline)

    # 默认「去噪」走 pre_norm 时，单独构建带 WaveletDenoiseProcessor 的数据集
    dataset_pre_norm_single = None
    if use_pre_flag:
        p_pre = _handler_pickle_path(tr0, te1, "_pre_norm")
        h_pre_conf = _inject_wavelet_pre_norm(h_baseline_conf, pre_norm_proc)
        dataset_pre_norm_single = _build_dataset_from_handler_conf(config, h_pre_conf, p_pre)

    # 消融里 baseline/after_norm 仍用无 pre 处理器的数据
    dataset = dataset_baseline

    # ------------------------------------------------------------------
    # Wavelet: YAML 为主；仅当命令行显式给出数值/字符串时覆盖（避免 CLI 默认盖掉 gentle yaml）
    # ------------------------------------------------------------------
    _wv_def = {
        "wavelet": "haar",
        "denoise_level": 1,
        "threshold_method": "bayes",
        "threshold_mode": "soft",
        "threshold_scale": 0.3,
        "denoise_blend": 0.25,
    }
    mkw = config["task"]["model"]["kwargs"]
    for k, dflt in _wv_def.items():
        v = getattr(args, k)
        if v is not None:
            mkw[k] = v
        elif k not in mkw:
            mkw[k] = dflt
    mkw["denoise_finest_only"] = not args.no_denoise_finest_only
    mkw["level_dependent_scale"] = not args.no_level_dependent_scale
    mkw["use_edge_pad"] = args.use_edge_pad
    mkw["use_boundary_smooth"] = args.use_boundary_smooth
    mkw["boundary_smooth_win"] = args.boundary_smooth_win

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
            h_conf_orig = copy.deepcopy(config["_h_conf_orig"])
            h_pre_conf = _inject_wavelet_pre_norm(h_conf_orig, pre_norm_proc)
            p_pre = _handler_pickle_path(tr0, te1, "_pre_norm")
            dataset_pre_norm = _build_dataset_from_handler_conf(
                config, h_pre_conf, p_pre
            )

            pre_norm_metrics = run_group(
                config=config,
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
        if use_pre_flag and dataset_pre_norm_single is not None:
            in_model = config["task"]["model"]["kwargs"].get(
                "use_wavelet_denoise", False
            )
            if in_model:
                print(
                    "[warn] use_pre_norm_wavelet is true and use_wavelet_denoise is true; "
                    "using pre_norm dataset only (in-model denoise disabled to avoid double denoise)."
                )
            run_group(
                dataset=dataset_pre_norm_single,
                use_wavelet_denoise=False,
                save_prefix_suffix=None,
                **_common,
            )
        else:
            run_group(
                dataset=dataset_baseline,
                use_wavelet_denoise=config["task"]["model"]["kwargs"].get(
                    "use_wavelet_denoise", False
                ),
                save_prefix_suffix=None,
                **_common,
            )
