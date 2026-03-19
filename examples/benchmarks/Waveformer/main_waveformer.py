"""
WaveFormer 训练 / 回测 主入口

用法示例
--------
# 正常训练 + 回测（无去噪基线）
python main_waveformer.py

# 启用小波去噪
python main_waveformer.py --wavelet haar --denoise_level 1 --threshold_scale 0.5

# 消融实验：自动跑「有去噪」和「无去噪」两组，输出性能差值表
python main_waveformer.py --ablation

# 只回测（跳过训练，需要已有 model/*.pkl）
python main_waveformer.py --only_backtest

# 消融 + 只回测
python main_waveformer.py --ablation --only_backtest

# 指定多个 seed
python main_waveformer.py --seeds 0 1 2

# 带标签记录到日志
python main_waveformer.py --tag haar_l1_s05 --notes "保守去噪参数"

# 长期收益预测（小波去噪更适合）：回看 20 天，预测 20 日收益
python main_waveformer.py --step_len 20 --label_horizon 20 --ablation
"""

import sys
from pathlib import Path

DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME))
sys.path.append(str(DIRNAME.parent.parent.parent))

import argparse
import copy
import os
import pprint as pp
from typing import Optional

import numpy as np
import yaml
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="WaveFormer training / backtest entry point"
    )
    parser.add_argument(
        "--only_backtest", action="store_true",
        help="跳过训练，直接加载已有模型进行回测"
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="消融实验：自动跑「无去噪」和「有去噪」两组并打印对比表"
    )
    parser.add_argument(
        "--wavelet", type=str, default="haar",
        choices=["haar", "db1"],
        help="小波类型（消融实验时为有去噪组的配置，默认 haar）"
    )
    parser.add_argument(
        "--denoise_level", type=int, default=1,
        help="DWT 分解层数（默认 1；T=8 时 haar 最多支持 level=3）"
    )
    parser.add_argument(
        "--threshold_scale", type=float, default=0.3,
        help="阈值缩放系数（默认 0.3，轻量去噪）"
    )
    parser.add_argument(
        "--threshold_mode", type=str, default="soft",
        choices=["soft", "hard", "semisoft"],
        help="阈值模式（默认 soft；semisoft 保留更多小系数）"
    )
    parser.add_argument(
        "--threshold_method", type=str, default="bayes",
        choices=["bayes", "visu"],
        help="阈值方法：bayes= BayesShrink（自适应，较保守）；visu= VisuShrink（默认 bayes）"
    )
    parser.add_argument(
        "--denoise_blend", type=float, default=0.25,
        help="去噪混合比例：output=(1-blend)*raw+blend*denoised，0=无去噪效果，1=全去噪（默认 0.25，轻量）"
    )
    parser.add_argument(
        "--no_denoise_finest_only", action="store_true",
        help="对所有层做阈值（默认仅最细层）"
    )
    parser.add_argument(
        "--no_level_dependent_scale", action="store_true",
        help="所有层使用相同阈值（默认粗层更小）"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0],
        help="随机种子列表（默认 0）"
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="本次实验标签，写入日志（如 haar_l1_s05）"
    )
    parser.add_argument(
        "--notes", type=str, default="",
        help="本次实验备注，写入日志"
    )
    parser.add_argument(
        "--log_dir", type=str, default=None,
        help="日志目录（默认 Waveformer/logs/，与运行路径无关）"
    )
    parser.add_argument(
        "--config", type=str,
        default="./workflow_config_waveformer_Alpha158_lowturn.yaml",
        help="yaml 配置文件路径"
    )
    parser.add_argument(
        "--step_len", type=int, default=8,
        help="回看窗口长度（交易日数）。默认 8；长期预测建议 20–40"
    )
    parser.add_argument(
        "--label_horizon", type=int, default=5,
        help="预测标签 horizon：Ref($close,-X)/Ref($close,-1)-1 中 X。默认 5（约 4 日收益）；长期建议 20（约 1 月）"
    )
    parser.add_argument(
        "--n_drop", type=int, default=None,
        help="每日换手数量（topk=30 时，n_drop=1 约持 30 天，n_drop=5 约持 6 天）。不指定时按 label_horizon 自动匹配"
    )
    parser.add_argument(
        "--topk", type=int, default=None,
        help="持仓股票数量。默认 30；长期预测可设 15–20 以集中持仓"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Single run (one group of seeds with given wavelet settings)
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "IC",
    "ICIR",
    "Rank IC",
    "Rank ICIR",
    "1day.excess_return_without_cost.annualized_return",
    "1day.excess_return_without_cost.information_ratio",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.information_ratio",
]


def run_group(config: dict, seeds: list, only_backtest: bool,
              use_wavelet_denoise: bool,
              wavelet: str, denoise_level: int,
              threshold_scale: float, threshold_mode: str,
              threshold_method: str = "bayes",
              denoise_blend: float = 0.25,
              denoise_finest_only: bool = True,
              level_dependent_scale: bool = True,
              use_edge_pad: bool = True,
              boundary_smooth_win: int = 1,
              step_len: int = 8,
              label_horizon: int = 5,
              n_drop: int = 1,
              topk: int = 30,
              tag: str = "", log_dir: str = "logs", notes: str = "",
              save_prefix_suffix: Optional[str] = None):
    """
    Train / backtest over `seeds` with a fixed wavelet config.
    Returns dict {metric_name: [values per seed]}.
    """
    all_metrics = {k: [] for k in METRIC_KEYS}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"seed={seed}  wavelet_denoise={use_wavelet_denoise}"
              + (f"  wavelet={wavelet}  level={denoise_level}"
                 f"  scale={threshold_scale}" if use_wavelet_denoise else ""))
        print("=" * 60)

        cfg = copy.deepcopy(config)
        market = cfg.get("market", "csi300")
        cfg["task"]["model"]["kwargs"]["seed"] = seed
        cfg["task"]["model"]["kwargs"]["use_wavelet_denoise"] = use_wavelet_denoise
        # 消融时 baseline 与 denoise 用不同 save_prefix，避免互相覆盖，保留 6 个模型
        if save_prefix_suffix is not None:
            cfg["task"]["model"]["kwargs"]["save_prefix"] = f"{market}_{save_prefix_suffix}"
        if use_wavelet_denoise:
            cfg["task"]["model"]["kwargs"]["wavelet"] = wavelet
            cfg["task"]["model"]["kwargs"]["denoise_level"] = denoise_level
            cfg["task"]["model"]["kwargs"]["threshold_scale"] = threshold_scale
            cfg["task"]["model"]["kwargs"]["threshold_mode"] = threshold_mode
            cfg["task"]["model"]["kwargs"]["threshold_method"] = threshold_method
            cfg["task"]["model"]["kwargs"]["denoise_blend"] = denoise_blend
            cfg["task"]["model"]["kwargs"]["denoise_finest_only"] = denoise_finest_only
            cfg["task"]["model"]["kwargs"]["level_dependent_scale"] = level_dependent_scale
            cfg["task"]["model"]["kwargs"]["use_edge_pad"] = use_edge_pad
            cfg["task"]["model"]["kwargs"]["use_boundary_smooth"] = cfg["task"]["model"]["kwargs"].get("use_boundary_smooth", False)
            cfg["task"]["model"]["kwargs"]["boundary_smooth_win"] = boundary_smooth_win

        model = init_instance_by_config(cfg["task"]["model"])

        if not only_backtest:
            model.fit(dataset=dataset)  # `dataset` is set in __main__
        else:
            save_path = cfg["task"]["model"]["kwargs"].get("save_path", "model/")
            save_prefix = cfg["task"]["model"]["kwargs"]["save_prefix"]
            ckpt = str(Path(save_path) / f"{save_prefix}_{seed}.pkl")
            model.load_model(ckpt)

        exp_name = f"waveformer_{'denoise' if use_wavelet_denoise else 'baseline'}_seed{seed}"
        with R.start(experiment_name=exp_name):
            recorder = R.get_recorder()

            sr = SignalRecord(model, dataset, recorder)
            sr.generate()

            sar = SigAnaRecord(recorder)
            sar.generate()

            par = PortAnaRecord(recorder, cfg["port_analysis_config"], "day")
            par.generate()

            metrics = recorder.list_metrics()
            pp.pprint({k: metrics.get(k) for k in METRIC_KEYS if k in metrics})

            for k in METRIC_KEYS:
                if k in metrics:
                    all_metrics[k].append(metrics[k])

    # --- log this group ---
    try:
        from backtest_logger import BacktestLogger
        logger = BacktestLogger(log_dir=log_dir)
        summary = {k: float(np.mean(v)) for k, v in all_metrics.items() if v}
        logger.log(
            tag=tag or ("denoise" if use_wavelet_denoise else "baseline"),
            config={
                "use_wavelet_denoise": use_wavelet_denoise,
                "wavelet": wavelet if use_wavelet_denoise else None,
                "denoise_level": denoise_level if use_wavelet_denoise else None,
                "threshold_scale": threshold_scale if use_wavelet_denoise else None,
                "threshold_mode": threshold_mode if use_wavelet_denoise else None,
                "threshold_method": threshold_method if use_wavelet_denoise else None,
                "denoise_blend": denoise_blend if use_wavelet_denoise else None,
                "denoise_finest_only": denoise_finest_only if use_wavelet_denoise else None,
                "level_dependent_scale": level_dependent_scale if use_wavelet_denoise else None,
                "step_len": step_len,
                "label_horizon": label_horizon,
                "n_drop": n_drop,
                "topk": topk,
                "seeds": seeds,
                "notes": notes,
            },
            metrics=summary,
        )
    except Exception as e:
        print(f"[main] Warning: logging failed: {e}")

    return all_metrics


# ---------------------------------------------------------------------------
# Ablation comparison table
# ---------------------------------------------------------------------------

def print_ablation_table(baseline: dict, denoise: dict, log_dir: str = ""):
    """打印并保存消融对比表到 log_dir/ablation_YYYYMMDD_HHMMSS.txt"""
    from datetime import datetime
    lines = []
    fmt = "{:<52s} {:>10s} {:>10s} {:>10s}"
    lines.append("")
    lines.append("=" * 78)
    lines.append("消融实验对比表")
    lines.append("=" * 78)
    lines.append(fmt.format("指标", "无去噪", "有去噪(Haar)", "差值 Δ"))
    lines.append("-" * 78)
    for k in METRIC_KEYS:
        b = baseline.get(k, [])
        d = denoise.get(k, [])
        if not b or not d:
            continue
        bm, dm = np.mean(b), np.mean(d)
        delta = dm - bm
        sign = "+" if delta >= 0 else ""
        lines.append(fmt.format(
            k[:52],
            f"{bm:.4f}",
            f"{dm:.4f}",
            f"{sign}{delta:.4f}",
        ))
    lines.append("=" * 78)
    text = "\n".join(lines)
    print(text)

    # 保存到文件
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fname = f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        out_path = log_path / fname
        out_path.write_text(text, encoding="utf-8")
        print(f"[main] 消融对比表已保存 → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    if args.log_dir is None:
        args.log_dir = str(DIRNAME / "logs")  # 固定为脚本所在目录下的 logs，与 cwd 无关

    provider_uri = "~/.qlib/qlib_data/cn_data"
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 注入 step_len 和 label_horizon（长期收益预测，与小波去噪更匹配）
    # 回看窗口应与预测周期匹配：长期预测时 step_len 至少等于 label_horizon
    if args.step_len < args.label_horizon:
        args.step_len = args.label_horizon
        print(f"[main] step_len 已自动调整为 {args.step_len}（与 label_horizon 匹配）")
    config["task"]["dataset"]["kwargs"]["step_len"] = args.step_len
    # 持仓策略与预测周期匹配：短期预测→高换手，长期预测→低换手
    if args.n_drop is None:
        # topk=30: n_drop=1 约持 30 天，n_drop=5 约持 6 天
        if args.label_horizon <= 5:
            args.n_drop = 5   # 短期（4 日收益）→ 约 6 天持仓
        elif args.label_horizon <= 10:
            args.n_drop = 3
        else:
            args.n_drop = 2   # 长期（20 日收益）→ 约 20 天持仓
        print(f"[main] 按 label_horizon={args.label_horizon} 自动设置 n_drop={args.n_drop}")
    config["port_analysis_config"]["strategy"]["kwargs"]["n_drop"] = args.n_drop

    # 持仓数量：不指定时长期用 20、短期用 30
    if args.topk is None:
        args.topk = 20 if args.label_horizon > 10 else 30
        print(f"[main] 按 label_horizon={args.label_horizon} 自动设置 topk={args.topk}")
    config["port_analysis_config"]["strategy"]["kwargs"]["topk"] = args.topk

    hdl = config["task"]["dataset"]["kwargs"]["handler"]
    if isinstance(hdl, dict):
        if "kwargs" not in hdl:
            hdl["kwargs"] = {}
        hdl["kwargs"]["label"] = [f"Ref($close, -{args.label_horizon}) / Ref($close, -1) - 1"]

    # Pre-process and cache the Alpha158 handler（不同 label_horizon 用不同缓存）
    h_conf = config["task"]["dataset"]["kwargs"]["handler"]
    seg = config["task"]["dataset"]["kwargs"]["segments"]
    train_s, test_e = seg["train"][0].strftime("%Y%m%d"), seg["test"][1].strftime("%Y%m%d")
    h_suffix = f"_L{args.label_horizon}" if args.label_horizon != 5 else ""
    h_path = DIRNAME / f"handler_{train_s}_{test_e}{h_suffix}.pkl"
    if not h_path.exists():
        h = init_instance_by_config(h_conf)
        h.to_pickle(h_path, dump_all=True)
        print(f"Preprocessed handler saved to {h_path} (label_horizon={args.label_horizon})")
    config["task"]["dataset"]["kwargs"]["handler"] = f"file://{h_path}"

    # Build dataset once (shared across seeds / groups)
    dataset = init_instance_by_config(config["task"]["dataset"])
    print(f"Dataset: step_len={args.step_len}, label_horizon={args.label_horizon} "
          f"(预测未来 {args.label_horizon-1} 日收益), topk={args.topk}, n_drop={args.n_drop} (约持 {args.topk//args.n_drop} 天)")

    os.makedirs("./model", exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    print(f"[main] 日志目录: {args.log_dir}")

    # ------------------------------------------------------------------
    # Ablation: two groups
    # ------------------------------------------------------------------
    if args.ablation:
        print("\n>>> 消融实验模式：跑「无去噪基线」和「有去噪」两组 <<<")

        baseline_metrics = run_group(
            config=config,
            seeds=args.seeds,
            only_backtest=args.only_backtest,
            use_wavelet_denoise=False,
            wavelet=args.wavelet,
            denoise_level=args.denoise_level,
            threshold_scale=args.threshold_scale,
            threshold_mode=args.threshold_mode,
            threshold_method=getattr(args, "threshold_method", "bayes"),
            denoise_blend=args.denoise_blend,
            denoise_finest_only=not getattr(args, "no_denoise_finest_only", False),
            level_dependent_scale=not getattr(args, "no_level_dependent_scale", False),
            step_len=args.step_len,
            label_horizon=args.label_horizon,
            n_drop=args.n_drop,
            topk=args.topk,
            tag="baseline",
            log_dir=args.log_dir,
            notes="ablation baseline",
            save_prefix_suffix="baseline",
        )

        denoise_metrics = run_group(
            config=config,
            seeds=args.seeds,
            only_backtest=args.only_backtest,
            use_wavelet_denoise=True,
            wavelet=args.wavelet,
            denoise_level=args.denoise_level,
            threshold_scale=args.threshold_scale,
            threshold_mode=args.threshold_mode,
            threshold_method=getattr(args, "threshold_method", "bayes"),
            denoise_blend=args.denoise_blend,
            denoise_finest_only=not getattr(args, "no_denoise_finest_only", False),
            level_dependent_scale=not getattr(args, "no_level_dependent_scale", False),
            step_len=args.step_len,
            label_horizon=args.label_horizon,
            n_drop=args.n_drop,
            topk=args.topk,
            tag=args.tag or f"{args.wavelet}_l{args.denoise_level}_s{args.threshold_scale}_b{args.denoise_blend}",
            log_dir=args.log_dir,
            notes=args.notes or "ablation denoise",
            save_prefix_suffix="denoise",
        )

        print_ablation_table(baseline_metrics, denoise_metrics, log_dir=args.log_dir)

    # ------------------------------------------------------------------
    # Normal run: single group with args.wavelet settings
    # ------------------------------------------------------------------
    else:
        use_denoise = config["task"]["model"]["kwargs"].get("use_wavelet_denoise", False)
        all_metrics = run_group(
            config=config,
            seeds=args.seeds,
            only_backtest=args.only_backtest,
            use_wavelet_denoise=use_denoise,
            wavelet=args.wavelet,
            denoise_level=args.denoise_level,
            threshold_scale=args.threshold_scale,
            threshold_mode=args.threshold_mode,
            threshold_method=getattr(args, "threshold_method", "bayes"),
            denoise_blend=args.denoise_blend,
            denoise_finest_only=not getattr(args, "no_denoise_finest_only", False),
            level_dependent_scale=not getattr(args, "no_level_dependent_scale", False),
            step_len=args.step_len,
            label_horizon=args.label_horizon,
            n_drop=args.n_drop,
            topk=args.topk,
            tag=args.tag,
            log_dir=args.log_dir,
            notes=args.notes,
        )

        print("\n" + "=" * 60)
        print("汇总（mean ± std）")
        print("=" * 60)
        for k, v in all_metrics.items():
            if v:
                print(f"  {k:<52s}: {np.mean(v):.4f} ± {np.std(v):.4f}")

    # Print recent history from log
    try:
        from backtest_logger import BacktestLogger
        BacktestLogger(log_dir=args.log_dir).print_recent(n=10)
    except Exception:
        pass
