 """
BacktestLogger
==============
每次运行自动保存：
  - logs/results.csv        所有实验汇总对照表（可 Excel 打开）
  - logs/{timestamp}_{tag}.json  完整配置 + 指标详情

运行结束后调用 print_recent(n=10) 在终端打印最近 n 条历史对照表。
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# 写入 CSV 的列顺序（固定，方便 Excel 对齐）
_CSV_COLUMNS = [
    "timestamp",
    "tag",
    "use_wavelet_denoise",
    "wavelet",
    "denoise_level",
    "threshold_scale",
    "threshold_mode",
    "denoise_blend",
    "denoise_finest_only",
    "level_dependent_scale",
    "step_len",
    "label_horizon",
    "n_drop",
    "topk",
    "seeds",
    "IC",
    "ICIR",
    "Rank IC",
    "Rank ICIR",
    "ann_return_no_cost",
    "IR_no_cost",
    "ann_return_with_cost",
    "IR_with_cost",
    "notes",
]

# qlib metric key → CSV column 的映射
_METRIC_MAP = {
    "IC":   "IC",
    "ICIR": "ICIR",
    "Rank IC":   "Rank IC",
    "Rank ICIR": "Rank ICIR",
    "1day.excess_return_without_cost.annualized_return": "ann_return_no_cost",
    "1day.excess_return_without_cost.information_ratio": "IR_no_cost",
    "1day.excess_return_with_cost.annualized_return":    "ann_return_with_cost",
    "1day.excess_return_with_cost.information_ratio":    "IR_with_cost",
}


class BacktestLogger:
    """
    Usage
    -----
    logger = BacktestLogger(log_dir="logs")
    logger.log(
        tag="haar_l1_s05",
        config={"use_wavelet_denoise": True, "wavelet": "haar", ...},
        metrics={"IC": 0.042, "ann_return_with_cost": 0.124, ...},
    )
    logger.print_recent(n=10)
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "results.csv"
        self._ensure_csv_header()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        tag: str = "",
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Persist one experiment record.

        Parameters
        ----------
        tag     : short label for this run, e.g. "haar_l1_s05"
        config  : dict of hyperparameters / settings
        metrics : dict of metric_name → value (use qlib key names or CSV column names)
        """
        config = config or {}
        metrics = metrics or {}
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # --- build CSV row ---
        row: Dict[str, Any] = {col: "" for col in _CSV_COLUMNS}
        row["timestamp"] = ts
        row["tag"] = tag
        row["use_wavelet_denoise"] = config.get("use_wavelet_denoise", "")
        row["wavelet"] = config.get("wavelet", "")
        row["denoise_level"] = config.get("denoise_level", "")
        row["threshold_scale"] = config.get("threshold_scale", "")
        row["threshold_mode"] = config.get("threshold_mode", "")
        row["denoise_blend"] = config.get("denoise_blend", "")
        row["denoise_finest_only"] = config.get("denoise_finest_only", "")
        row["level_dependent_scale"] = config.get("level_dependent_scale", "")
        row["step_len"] = config.get("step_len", "")
        row["label_horizon"] = config.get("label_horizon", "")
        row["n_drop"] = config.get("n_drop", "")
        row["topk"] = config.get("topk", "")
        seeds = config.get("seeds", "")
        row["seeds"] = str(seeds) if seeds != "" else ""
        row["notes"] = config.get("notes", "")

        for qlib_key, csv_col in _METRIC_MAP.items():
            if qlib_key in metrics:
                row[csv_col] = _fmt(metrics[qlib_key])
            elif csv_col in metrics:
                row[csv_col] = _fmt(metrics[csv_col])

        self._append_csv(row)

        # --- save detailed JSON ---
        json_name = f"{ts}_{tag}.json" if tag else f"{ts}.json"
        json_path = self.log_dir / json_name
        payload = {
            "timestamp": ts,
            "tag": tag,
            "config": config,
            "metrics": {k: (_fmt(v) if isinstance(v, float) else v)
                        for k, v in metrics.items()},
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"[BacktestLogger] Record saved → {self.csv_path}  /  {json_path.name}")

    def print_recent(self, n: int = 10):
        """Print the last n rows of results.csv as a formatted table."""
        if not self.csv_path.exists():
            print("[BacktestLogger] No log file found.")
            return

        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        rows = reader[-n:]
        if not rows:
            print("[BacktestLogger] Log is empty.")
            return

        # Columns to display in the terminal summary
        display_cols = [
            "timestamp", "tag",
            "use_wavelet_denoise", "wavelet", "denoise_level", "threshold_scale",
            "IC", "ICIR", "ann_return_no_cost", "IR_no_cost",
            "ann_return_with_cost", "IR_with_cost",
        ]

        # Compute column widths
        widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0))
                  for c in display_cols}

        sep = "  "
        header = sep.join(c.ljust(widths[c]) for c in display_cols)
        divider = sep.join("-" * widths[c] for c in display_cols)

        print(f"\n{'='*20} 最近 {len(rows)} 条回测记录 {'='*20}")
        print(header)
        print(divider)
        for r in rows:
            line = sep.join(str(r.get(c, "")).ljust(widths[c]) for c in display_cols)
            print(line)
        print("=" * (len(header) + 2))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_csv_header(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
                writer.writeheader()

    def _append_csv(self, row: Dict[str, Any]):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            writer.writerow(row)


def _fmt(v) -> str:
    """Format a float with 4 decimal places; leave non-float as-is."""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v) if v is not None else ""
