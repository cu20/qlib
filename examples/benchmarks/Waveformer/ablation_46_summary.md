## 4.6 消融实验（按 csi300_after_norm_no_0 口径修正）
图文件：`ablation_46_main_metrics.png`、`ablation_46_a3_seed_stability.png`。
A2 使用用户指定 checkpoint：`model/csi300_after_norm_no_0.pkl`（only_backtest 产出的 `csi300_after_norm_no_seed0_metrics.json`）。

### 六项核心指标
| Group | IC | ICIR | Rank IC | Rank ICIR | AR | IR |
|---|---:|---:|---:|---:|---:|---:|
| A0 Baseline | 0.0544 | 0.3328 | 0.0616 | 0.3693 | 0.2538 | 1.939 |
| A1 Pre-Norm | 0.0539 | 0.3289 | 0.0612 | 0.3677 | 0.2424 | 1.853 |
| A2 After-Norm(no) | NaN | NaN | NaN | NaN | 0.1166 | 0.752 |
| A3 After+Adaptive | 0.0567 | 0.3611 | 0.0645 | 0.4020 | 0.2451 | 1.914 |

### 结论（修正版）
1. 在你指定的 A2（after_norm_no）口径下，IC/ICIR/Rank IC/Rank ICIR 为 NaN，说明该 checkpoint 在当前评估流程下没有形成可用的预测相关性统计。
2. 仅从无摩擦收益指标看，A2 的 AR=0.1166、IR=0.752，低于 A0/A1/A3。
3. 当前可用结果中，A0 在收益指标上最好（AR=0.2538，IR=1.939）；A0 在当前单次 baseline 下也略优于 A1。
