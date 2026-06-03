import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_artifact(run_dir: Path, rel_path: str) -> Path:
    cand_1 = run_dir / "artifacts" / rel_path
    cand_2 = run_dir / rel_path
    if cand_1.exists():
        return cand_1
    if cand_2.exists():
        return cand_2
    raise FileNotFoundError(f"artifact not found: {rel_path} under {run_dir}")


def _drawdown(series: pd.Series) -> pd.Series:
    equity = (1.0 + series.fillna(0.0)).cumprod()
    peak = equity.cummax()
    return equity / peak - 1.0


def _daily_spread(pred: pd.DataFrame, topk: int = 30, next_k: int = 5) -> pd.Series:
    score_col = pred.columns[0]

    def _one_day(df: pd.DataFrame) -> float:
        scores = df[score_col].sort_values(ascending=False)
        if len(scores) < topk + next_k:
            return np.nan
        return float(scores.iloc[:topk].mean() - scores.iloc[topk : topk + next_k].mean())

    return pred.groupby(level="datetime", sort=True).apply(_one_day)


def build_table(run_dir: Path, topk: int = 30, next_k: int = 5) -> pd.DataFrame:
    pred = pd.read_pickle(_resolve_artifact(run_dir, "pred.pkl"))
    report = pd.read_pickle(_resolve_artifact(run_dir, "portfolio_analysis/report_normal_1day.pkl"))
    if not isinstance(report, pd.DataFrame):
        raise TypeError("report_normal_1day.pkl is not DataFrame")

    out = pd.DataFrame(index=report.index.copy())
    out["ret"] = report["return"].astype(float)
    out["bench"] = report["bench"].astype(float)
    out["cost"] = report["cost"].astype(float)
    out["turnover"] = report["turnover"].astype(float)
    out["excess"] = out["ret"] - out["bench"]
    out["excess_cost"] = out["excess"] - out["cost"]
    out["drawdown"] = _drawdown(out["excess_cost"])
    out["spread"] = _daily_spread(pred, topk=topk, next_k=next_k)

    out["spread_ma3"] = out["spread"].rolling(3, min_periods=1).mean()
    out["spread_ma10"] = out["spread"].rolling(10, min_periods=1).mean()
    out["spread_ma20"] = out["spread"].rolling(20, min_periods=1).mean()
    out["spread_std10"] = out["spread"].rolling(10, min_periods=1).std().fillna(0.0)
    out["spread_std20"] = out["spread"].rolling(20, min_periods=1).std().fillna(0.0)
    out["spread_z"] = (out["spread"] - out["spread_ma10"]) / (out["spread_std10"] + 1e-8)
    out["spread_z20"] = (out["spread"] - out["spread_ma20"]) / (out["spread_std20"] + 1e-8)
    out["spread_diff1"] = out["spread"].diff(1).fillna(0.0)
    out["spread_diff5"] = out["spread"].diff(5).fillna(0.0)
    out["spread_slope5"] = out["spread"].rolling(5, min_periods=2).apply(
        lambda x: float(np.polyfit(np.arange(len(x)), x, 1)[0]), raw=False
    ).fillna(0.0)
    out["turnover_ma5"] = out["turnover"].rolling(5, min_periods=1).mean()
    out["turnover_ma20"] = out["turnover"].rolling(20, min_periods=1).mean()
    out["turnover_z20"] = (
        (out["turnover"] - out["turnover_ma20"])
        / (out["turnover"].rolling(20, min_periods=1).std().fillna(0.0) + 1e-8)
    )
    out["dd_ma10"] = out["drawdown"].rolling(10, min_periods=1).mean()
    out["dd_abs"] = out["drawdown"].abs()
    out["excess_cost_ma5"] = out["excess_cost"].rolling(5, min_periods=1).mean()

    # Next-day reward target for timing policy
    out["target_next_excess"] = out["excess"].shift(-1)
    out["target_next_excess_cost"] = out["excess_cost"].shift(-1)
    out = out.dropna(subset=["spread", "target_next_excess", "target_next_excess_cost"]).copy()
    out.index.name = "datetime"
    return out.reset_index()


def parse_args():
    parser = argparse.ArgumentParser(description="Export timing-RL training table from qlib run artifacts.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="logs/timing_rl_table.csv")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--next_k", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_path = Path(args.output_csv).expanduser().resolve()
    table = build_table(run_dir=run_dir, topk=args.topk, next_k=args.next_k)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(f"[timing_rl_dataset] rows={len(table)} saved={out_path}")


if __name__ == "__main__":
    main()
