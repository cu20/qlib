import json
from collections import deque
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden: int = 64,
        n_actions: int = 3,
        aux_mode: str = "none",
        gate_coef: float = 0.8,
        gate_uncertainty_coef: float = 0.30,
    ):
        super().__init__()
        self.aux_mode = str(aux_mode)
        self.gate_coef = float(gate_coef)
        self.gate_uncertainty_coef = float(gate_uncertainty_coef)
        self.n_actions = int(n_actions)
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        if self.aux_mode in {"state_fusion", "logit_gate"}:
            self.aux_head1 = nn.Linear(hidden, 1)
            self.aux_head5 = nn.Linear(hidden, 1)
            self.aux_uncertainty = nn.Linear(hidden, 1)
        if self.aux_mode == "state_fusion":
            self.fusion = nn.Linear(hidden + 3, hidden)
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        if self.aux_mode in {"state_fusion", "logit_gate"}:
            pred1 = self.aux_head1(h)
            pred5 = self.aux_head5(h)
            uncertainty = torch.nn.functional.softplus(self.aux_uncertainty(h))
            if self.aux_mode == "state_fusion":
                h = torch.tanh(self.fusion(torch.cat([h, pred1, pred5, uncertainty], dim=1)))
        else:
            pred1 = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
            pred5 = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
            uncertainty = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
        logits = self.pi(h)
        if self.aux_mode == "logit_gate":
            gate = torch.tanh(1.25 * pred1.squeeze(-1) + 0.85 * pred5.squeeze(-1))
            gate = gate - self.gate_uncertainty_coef * torch.clamp(uncertainty.squeeze(-1), min=0.0, max=2.0)
            logits = logits.clone()
            for i in (2, 3, 4):
                if i < self.n_actions:
                    logits[:, i] = logits[:, i] + self.gate_coef * gate
            if self.n_actions > 0:
                logits[:, 0] = logits[:, 0] - self.gate_coef * gate
        return logits, self.v(h).squeeze(-1)


class TimingRLTopkStrategy(TopkDropoutStrategy):
    """
    RL-timing strategy that learns when to be conservative/aggressive on buy side.
    """

    def __init__(
        self,
        *,
        policy_path: str,
        default_risk_degree: float = 0.8,
        default_topk: int = 30,
        default_n_drop: int = 30,
        min_topk: int = 10,
        max_topk: int = 80,
        min_n_drop: int = 5,
        max_n_drop: int = 80,
        action_mode: str = "argmax",
        sample_temperature: float = 1.0,
        debug_log: bool = False,
        debug_log_warmup_steps: int = 20,
        debug_log_interval: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.policy = self._load_policy(Path(policy_path))
        self.default_risk_degree = float(default_risk_degree)
        self.default_topk = int(default_topk)
        self.default_n_drop = int(default_n_drop)
        self.min_topk = int(min_topk)
        self.max_topk = int(max_topk)
        self.min_n_drop = int(min_n_drop)
        self.max_n_drop = int(max_n_drop)
        self.action_mode = str(action_mode)
        self.sample_temperature = float(sample_temperature)
        self.debug_log = bool(debug_log)
        self.debug_log_warmup_steps = int(debug_log_warmup_steps)
        self.debug_log_interval = int(debug_log_interval)
        self._spread_hist = deque(maxlen=20)
        self._turn_hist = deque(maxlen=20)
        self._dd_hist = deque(maxlen=20)
        self._debug_step = 0
        self._init_actor()

    @staticmethod
    def _load_policy(path: Path) -> Dict:
        if not path.exists():
            raise FileNotFoundError(f"Timing policy not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _init_actor(self):
        if self.policy.get("policy_type") not in {
            "timing_ppo_actor_critic_v1",
            "timing_ppo_actor_critic_v2",
            "timing_ppo_actor_critic_v3",
            "timing_ppo_actor_critic_v4",
        }:
            raise ValueError("Unsupported policy_type for TimingRLTopkStrategy")
        obs_dim = len(self.policy["feature_cols"])
        n_actions = len(self.policy["actions"])
        model_cfg = self.policy.get("model_config", {})
        aux_mode = str(model_cfg.get("aux_mode", "none"))
        if aux_mode == "none" and bool(model_cfg.get("use_aux_state", False)):
            aux_mode = "state_fusion"
        if self.policy.get("policy_type") == "timing_ppo_actor_critic_v3" and aux_mode == "none":
            aux_mode = "state_fusion"
        if self.policy.get("policy_type") == "timing_ppo_actor_critic_v4" and aux_mode == "none":
            aux_mode = "logit_gate"
        hidden = int(model_cfg.get("hidden", 64))
        self.actor = ActorCritic(
            obs_dim=obs_dim,
            hidden=hidden,
            n_actions=n_actions,
            aux_mode=aux_mode,
            gate_coef=float(model_cfg.get("gate_coef", 0.8)),
            gate_uncertainty_coef=float(model_cfg.get("gate_uncertainty_coef", 0.30)),
        )
        state_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in self.policy["state_dict"].items()}
        self.actor.load_state_dict(state_dict)
        self.actor.eval()

    @staticmethod
    def _signal_spread(signal: pd.Series, topk: int) -> float:
        if signal is None or len(signal) < topk + 5:
            return 0.0
        s = signal.sort_values(ascending=False)
        return float(s.iloc[:topk].mean() - s.iloc[topk : topk + 5].mean())

    @staticmethod
    def _safe_mean(buf: deque, fallback: float = 0.0) -> float:
        if not buf:
            return float(fallback)
        return float(np.mean(np.asarray(buf, dtype=float)))

    def _build_features(self, spread: float) -> np.ndarray:
        self._spread_hist.append(spread)
        spread_ma3 = self._safe_mean(deque(list(self._spread_hist)[-3:], maxlen=3), spread)
        spread_ma10 = self._safe_mean(deque(list(self._spread_hist)[-10:], maxlen=10), spread)
        spread_ma20 = self._safe_mean(deque(list(self._spread_hist)[-20:], maxlen=20), spread)
        spread_std10 = float(np.std(np.asarray(list(self._spread_hist)[-10:], dtype=float))) if self._spread_hist else 0.0
        spread_std20 = float(np.std(np.asarray(list(self._spread_hist)[-20:], dtype=float))) if self._spread_hist else 0.0
        spread_z = float((spread - spread_ma10) / (spread_std10 + 1e-8))
        spread_z20 = float((spread - spread_ma20) / (spread_std20 + 1e-8))
        spreads = list(self._spread_hist)
        spread_diff1 = float(spreads[-1] - spreads[-2]) if len(spreads) >= 2 else 0.0
        spread_diff5 = float(spreads[-1] - spreads[-6]) if len(spreads) >= 6 else 0.0
        if len(spreads) >= 2:
            x = np.arange(min(5, len(spreads)), dtype=float)
            y = np.asarray(spreads[-len(x) :], dtype=float)
            spread_slope5 = float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else 0.0
        else:
            spread_slope5 = 0.0

        turnover = self._safe_mean(self._turn_hist, 0.0)
        turn_vals = np.asarray(list(self._turn_hist), dtype=float) if self._turn_hist else np.asarray([0.0], dtype=float)
        turnover_ma5 = float(np.mean(turn_vals[-5:]))
        turnover_ma20 = float(np.mean(turn_vals[-20:]))
        turnover_std20 = float(np.std(turn_vals[-20:]))
        turnover_z20 = float((turnover - turnover_ma20) / (turnover_std20 + 1e-8))

        drawdown = self._safe_mean(self._dd_hist, 0.0)
        dd_vals = np.asarray(list(self._dd_hist), dtype=float) if self._dd_hist else np.asarray([0.0], dtype=float)
        dd_ma10 = float(np.mean(dd_vals[-10:]))
        dd_abs = float(abs(drawdown))
        excess_cost_ma5 = 0.0

        feature_dict = {
            "spread": spread,
            "spread_ma3": spread_ma3,
            "spread_ma10": spread_ma10,
            "spread_ma20": spread_ma20,
            "spread_z": spread_z,
            "spread_z20": spread_z20,
            "spread_diff1": spread_diff1,
            "spread_diff5": spread_diff5,
            "spread_slope5": spread_slope5,
            "turnover": turnover,
            "turnover_ma5": turnover_ma5,
            "turnover_ma20": turnover_ma20,
            "turnover_z20": turnover_z20,
            "drawdown": drawdown,
            "dd_ma10": dd_ma10,
            "dd_abs": dd_abs,
            "excess_cost_ma5": excess_cost_ma5,
        }
        vals = np.asarray([float(feature_dict.get(col, 0.0)) for col in self.policy["feature_cols"]], dtype=float)
        return vals

    def _choose_action(self, features: np.ndarray) -> Dict:
        mu = np.asarray(self.policy["feature_mean"], dtype=float)
        sd = np.asarray(self.policy["feature_std"], dtype=float)
        z = (features - mu) / (sd + 1e-8)
        x = torch.from_numpy(z.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.actor(x)
            if self.action_mode == "sample":
                temp = max(1e-3, self.sample_temperature)
                probs = torch.softmax(logits / temp, dim=1)
                aid = int(torch.multinomial(probs.squeeze(0), num_samples=1).item())
            else:
                aid = int(torch.argmax(logits, dim=1).item())
        for action in self.policy["actions"]:
            if int(action["id"]) == aid:
                return action
        return {"risk_degree": self.default_risk_degree, "topk_mult": 1.0, "n_drop_mult": 1.0}

    def _should_log(self) -> bool:
        if not self.debug_log:
            return False
        if self._debug_step < self.debug_log_warmup_steps:
            return True
        if self.debug_log_interval <= 0:
            return True
        return (self._debug_step % self.debug_log_interval) == 0

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        signal = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(signal, pd.DataFrame):
            signal = signal.iloc[:, 0]

        spread = self._signal_spread(signal, self.topk)
        feats = self._build_features(spread)
        action = self._choose_action(feats)

        topk_old, n_drop_old = int(self.topk), int(self.n_drop)
        risk_old = float(getattr(self, "risk_degree", self.default_risk_degree))
        try:
            new_topk = int(np.clip(round(self.default_topk * float(action.get("topk_mult", 1.0))), self.min_topk, self.max_topk))
            new_n_drop = int(np.clip(round(self.default_n_drop * float(action.get("n_drop_mult", 1.0))), self.min_n_drop, self.max_n_drop))
            new_risk = float(np.clip(float(action.get("risk_degree", self.default_risk_degree)), 0.3, 1.0))
            self.topk = new_topk
            self.n_drop = new_n_drop
            self.risk_degree = new_risk
            if self._should_log():
                print(
                    "[timing_rl] step=%d spread=%.6f action_id=%s topk=%d n_drop=%d risk=%.3f"
                    % (
                        self._debug_step,
                        float(spread),
                        str(action.get("id", "NA")),
                        self.topk,
                        self.n_drop,
                        self.risk_degree,
                    )
                )
            self._debug_step += 1
            return super().generate_trade_decision(execute_result=execute_result)
        finally:
            self.topk = topk_old
            self.n_drop = n_drop_old
            self.risk_degree = risk_old
