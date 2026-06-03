import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


FEATURE_COLS = [
    "spread",
    "spread_ma3",
    "spread_ma10",
    "spread_ma20",
    "spread_z",
    "spread_z20",
    "spread_diff1",
    "spread_diff5",
    "spread_slope5",
]
ACTIONS = [
    {"id": 0, "risk_degree": 0.72, "topk_mult": 0.96, "n_drop_mult": 0.85},
    {"id": 1, "risk_degree": 0.80, "topk_mult": 1.00, "n_drop_mult": 1.00},
    {"id": 2, "risk_degree": 0.90, "topk_mult": 1.04, "n_drop_mult": 1.15},
    {"id": 3, "risk_degree": 0.96, "topk_mult": 1.08, "n_drop_mult": 1.30},
    {"id": 4, "risk_degree": 0.92, "topk_mult": 0.95, "n_drop_mult": 1.25},
]


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
        logits, values, _, _, _ = self.forward_with_aux(x)
        return logits, values

    def forward_with_aux(self, x: torch.Tensor):
        h = self.backbone(x)
        if self.aux_mode in {"state_fusion", "logit_gate"}:
            pred1 = self.aux_head1(h).squeeze(-1)
            pred5 = self.aux_head5(h).squeeze(-1)
            uncertainty = F.softplus(self.aux_uncertainty(h)).squeeze(-1)
            if self.aux_mode == "state_fusion":
                h = torch.tanh(
                    self.fusion(torch.cat([h, pred1.unsqueeze(-1), pred5.unsqueeze(-1), uncertainty.unsqueeze(-1)], dim=1))
                )
        else:
            pred1 = h.new_zeros(h.shape[0])
            pred5 = h.new_zeros(h.shape[0])
            uncertainty = h.new_zeros(h.shape[0])
        logits = self.pi(h)
        if self.aux_mode == "logit_gate":
            gate = torch.tanh(1.25 * pred1 + 0.85 * pred5) - self.gate_uncertainty_coef * torch.clamp(uncertainty, min=0.0, max=2.0)
            logits = logits.clone()
            aggressive_ids = [i for i in (2, 3, 4) if i < self.n_actions]
            for i in aggressive_ids:
                logits[:, i] = logits[:, i] + self.gate_coef * gate
            if self.n_actions > 0:
                logits[:, 0] = logits[:, 0] - self.gate_coef * gate
        return logits, self.v(h).squeeze(-1), pred1, pred5, uncertainty

    def encode(self, x: torch.Tensor):
        return self.backbone(x)


def _normalize_obs(df: pd.DataFrame):
    x = df[FEATURE_COLS].to_numpy(dtype=float)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0) + 1e-8
    z = (x - mu) / sd
    return z.astype(np.float32), mu.astype(float), sd.astype(float)


def _action_turnover_mult(action: dict) -> float:
    risk = float(action["risk_degree"])
    topk_mult = float(action["topk_mult"])
    n_drop_mult = float(action["n_drop_mult"])
    return 1.0 + 1.15 * max(0.0, n_drop_mult - 1.0) + 0.35 * abs(topk_mult - 1.0) + 0.45 * max(0.0, risk - 0.8)


def _reward_per_action(df: pd.DataFrame, lambda_turnover: float, lambda_drawdown: float) -> np.ndarray:
    # Use with-cost alpha as anchor and keep a small gross-return component.
    base_excess = df["target_next_excess"].to_numpy(dtype=float)
    base_excess_cost = df["target_next_excess_cost"].to_numpy(dtype=float)
    spread_z = df["spread_z"].to_numpy(dtype=float)
    spread_diff1 = df["spread_diff1"].to_numpy(dtype=float)
    turnover = df["turnover"].to_numpy(dtype=float)
    drawdown = np.abs(df["drawdown"].to_numpy(dtype=float))
    dd_abs = np.abs(df["dd_abs"].to_numpy(dtype=float))
    high_regime = (spread_z > 0.7).astype(np.float32)
    low_regime = (spread_z < -0.7).astype(np.float32)
    out = np.zeros((len(df), len(ACTIONS)), dtype=np.float32)
    for i, action in enumerate(ACTIONS):
        risk = float(action["risk_degree"])
        topk_mult = float(action["topk_mult"])
        aggressiveness = risk - 0.8
        # Let policy become aggressive only when score spread is strong/improving.
        alpha_boost = 1.0 + 0.55 * aggressiveness * np.tanh(1.35 * spread_z) + 0.12 * np.sign(spread_diff1) * aggressiveness
        blended_alpha = 0.30 * base_excess + 0.70 * base_excess_cost
        turn_mult = _action_turnover_mult(action)
        turn_penalty = lambda_turnover * turnover * turn_mult
        concentration_mult = 1.0 + 0.9 * max(0.0, 1.0 - topk_mult)
        dd_penalty = lambda_drawdown * (0.65 * drawdown + 0.35 * dd_abs) * max(0.5, risk) * concentration_mult
        # Regime-aware bonus/penalty: reward aggression in high-signal days.
        regime_bonus = 0.00045 * high_regime * max(0.0, aggressiveness)
        regime_bonus += 0.00045 * low_regime * max(0.0, -aggressiveness)
        regime_bonus -= 0.00020 * high_regime * max(0.0, -aggressiveness)
        shaped = (blended_alpha * alpha_boost - turn_penalty - dd_penalty + regime_bonus).astype(np.float32)
        out[:, i] = shaped
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Train timing policy with PPO (true RL loop).")
    p.add_argument("--input_csv", type=str, required=True)
    p.add_argument("--output_json", type=str, default="rl_models/timing_policy_v15_auxgate_ppo.json")
    p.add_argument("--lambda_turnover", type=float, default=0.10)
    p.add_argument("--lambda_drawdown", type=float, default=0.30)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--horizon", type=int, default=256)
    p.add_argument("--ppo_epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy_coef", type=float, default=0.03)
    p.add_argument("--entropy_coef_start", type=float, default=0.10)
    p.add_argument("--entropy_coef_end", type=float, default=0.02)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--diversity_coef", type=float, default=0.02)
    p.add_argument("--bc_epochs", type=int, default=28)
    p.add_argument("--bc_batch_size", type=int, default=256)
    p.add_argument("--bc_coef_start", type=float, default=0.18)
    p.add_argument("--bc_coef_end", type=float, default=0.03)
    p.add_argument("--bc_label_smoothing", type=float, default=0.02)
    p.add_argument("--bc_class_balance_pow", type=float, default=0.5)
    p.add_argument("--anchor_kl_coef", type=float, default=0.08)
    p.add_argument("--freeze_backbone_during_ppo", type=int, default=1)
    p.add_argument("--aux_coef", type=float, default=0.20)
    p.add_argument("--rank_coef", type=float, default=0.05)
    p.add_argument("--aux_mode", type=str, default="logit_gate", choices=["none", "state_fusion", "logit_gate"])
    p.add_argument("--gate_coef", type=float, default=0.8)
    p.add_argument("--gate_uncertainty_coef", type=float, default=0.30)
    p.add_argument("--use_aux_state_for_policy", type=int, default=1)
    p.add_argument("--teacher_mode", type=str, default="oracle", choices=["oracle", "policy", "heuristic"])
    p.add_argument("--teacher_policy_path", type=str, default="")
    p.add_argument("--teacher_min_hold", type=int, default=5)
    p.add_argument("--teacher_switch_margin", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _compute_gae(rewards, values, dones, gamma, gae_lambda):
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    last = 0.0
    for t in reversed(range(n)):
        next_v = 0.0 if t == n - 1 else values[t + 1]
        non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_v * non_terminal - values[t]
        last = delta + gamma * gae_lambda * non_terminal * last
        adv[t] = last
    ret = adv + values
    return adv, ret


def _teacher_action_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Heuristic teacher around current strong manual policy.
    This provides a warm start before PPO fine-tuning.
    """
    spread_z = df["spread_z"].to_numpy(dtype=float)
    spread_diff1 = df["spread_diff1"].to_numpy(dtype=float)
    turnover_z20 = df["turnover_z20"].to_numpy(dtype=float) if "turnover_z20" in df.columns else np.zeros(len(df))

    labels = np.full(len(df), 1, dtype=np.int64)  # neutral baseline action
    labels[(spread_z > 0.55) & (spread_diff1 >= 0.0)] = 2
    labels[(spread_z > 1.15) & (spread_diff1 > 0.0)] = 3
    labels[(spread_z > 1.35) & (turnover_z20 < 0.25)] = 4
    labels[(spread_z < -0.60) | ((spread_z < -0.35) & (spread_diff1 < 0.0))] = 0
    return labels


def _policy_action_labels(df: pd.DataFrame, policy_path: str) -> np.ndarray:
    teacher = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    t_cols = teacher["feature_cols"]
    t_mu = np.asarray(teacher["feature_mean"], dtype=np.float32)
    t_sd = np.asarray(teacher["feature_std"], dtype=np.float32)
    t_actions = teacher["actions"]
    t_state = teacher["state_dict"]

    x = df[t_cols].to_numpy(dtype=np.float32)
    x = (x - t_mu) / (t_sd + 1e-8)

    t_obs_dim = int(len(t_cols))
    t_hidden = int(np.asarray(t_state["backbone.0.weight"]).shape[0])
    t_n_actions = int(len(t_actions))
    t_model_cfg = teacher.get("model_config", {})
    t_aux_mode = str(t_model_cfg.get("aux_mode", "none"))
    if t_aux_mode == "none" and bool(t_model_cfg.get("use_aux_state", False)):
        t_aux_mode = "state_fusion"
    if teacher.get("policy_type") == "timing_ppo_actor_critic_v3" and t_aux_mode == "none":
        t_aux_mode = "state_fusion"
    if teacher.get("policy_type") == "timing_ppo_actor_critic_v4" and t_aux_mode == "none":
        t_aux_mode = "logit_gate"
    t_model = ActorCritic(
        obs_dim=t_obs_dim,
        hidden=t_hidden,
        n_actions=t_n_actions,
        aux_mode=t_aux_mode,
        gate_coef=float(t_model_cfg.get("gate_coef", 0.8)),
        gate_uncertainty_coef=float(t_model_cfg.get("gate_uncertainty_coef", 0.30)),
    )
    t_model.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in t_state.items()})
    t_model.eval()

    with torch.no_grad():
        logits, _ = t_model(torch.from_numpy(x))
        teacher_ids = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)

    # Map teacher action space to current action space by nearest risk_degree.
    cur_risks = np.asarray([float(a["risk_degree"]) for a in ACTIONS], dtype=float)
    t_risks = np.asarray([float(a["risk_degree"]) for a in t_actions], dtype=float)
    risk_map = {i: int(np.argmin(np.abs(cur_risks - t_risks[i]))) for i in range(len(t_actions))}
    mapped = np.asarray([risk_map[int(i)] for i in teacher_ids], dtype=np.int64)
    return mapped


def _oracle_teacher_labels(rewards_table: np.ndarray, min_hold: int = 5, switch_margin: float = 5e-5) -> np.ndarray:
    """
    Build an offline "expert" action trajectory from realized one-step rewards,
    with holding and switching friction to avoid noisy day-by-day flips.
    """
    n, n_actions = rewards_table.shape
    out = np.zeros(n, dtype=np.int64)
    cur = int(np.argmax(rewards_table[0]))
    hold = 0
    out[0] = cur
    for t in range(1, n):
        best = int(np.argmax(rewards_table[t]))
        gain = float(rewards_table[t, best] - rewards_table[t, cur])
        if (hold >= int(min_hold)) and (gain > float(switch_margin)):
            cur = best
            hold = 0
        else:
            hold += 1
        out[t] = cur
    return out


def _label_class_weights(labels: np.ndarray, n_classes: int, balance_pow: float) -> np.ndarray:
    cnt = np.bincount(labels.astype(np.int64), minlength=n_classes).astype(np.float32)
    freq = cnt / max(1.0, float(cnt.sum()))
    w = 1.0 / np.power(np.clip(freq, 1e-6, 1.0), float(balance_pow))
    w = w / np.mean(w)
    return w.astype(np.float32)


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float()
    denom = torch.clamp(m.sum(), min=1.0)
    return (((pred - target) ** 2) * m).sum() / denom


def _pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, max_pairs: int = 256) -> torch.Tensor:
    valid = torch.nonzero(mask > 0.5).squeeze(-1)
    if valid.numel() < 2:
        return pred.new_tensor(0.0)
    n = int(valid.numel())
    p = min(max_pairs, n * (n - 1) // 2)
    i = valid[torch.randint(0, n, (p,), device=pred.device)]
    j = valid[torch.randint(0, n, (p,), device=pred.device)]
    non_same = i != j
    if non_same.sum() == 0:
        return pred.new_tensor(0.0)
    i = i[non_same]
    j = j[non_same]
    margin = (pred[i] - pred[j]) * (target[i] - target[j])
    return torch.relu(-margin).mean()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = pd.read_csv(Path(args.input_csv))
    obs_all, mu, sd = _normalize_obs(df)
    target1_all_np = df["target_next_excess_cost"].to_numpy(dtype=float)
    if "excess_cost" in df.columns:
        target5_all_np = df["excess_cost"].shift(-5).to_numpy(dtype=float)
    else:
        target5_all_np = df["target_next_excess_cost"].shift(-4).to_numpy(dtype=float)
    mask1_all_np = np.isfinite(target1_all_np).astype(np.float32)
    mask5_all_np = np.isfinite(target5_all_np).astype(np.float32)
    target1_all_np = np.nan_to_num(target1_all_np, nan=0.0).astype(np.float32)
    target5_all_np = np.nan_to_num(target5_all_np, nan=0.0).astype(np.float32)
    rewards_table = _reward_per_action(
        df=df,
        lambda_turnover=float(args.lambda_turnover),
        lambda_drawdown=float(args.lambda_drawdown),
    )
    teacher_mode = str(getattr(args, "teacher_mode", "oracle")).lower()
    if teacher_mode == "policy":
        if not getattr(args, "teacher_policy_path", ""):
            raise ValueError("teacher_mode=policy requires --teacher_policy_path")
        teacher_labels = _policy_action_labels(df, str(args.teacher_policy_path))
        print(f"[timing_rl_train] teacher_source=policy path={args.teacher_policy_path}")
    elif teacher_mode == "heuristic":
        teacher_labels = _teacher_action_labels(df)
        print("[timing_rl_train] teacher_source=heuristic")
    else:
        teacher_labels = _oracle_teacher_labels(
            rewards_table,
            min_hold=int(getattr(args, "teacher_min_hold", 5)),
            switch_margin=float(getattr(args, "teacher_switch_margin", 5e-5)),
        )
        print(
            f"[timing_rl_train] teacher_source=oracle min_hold={int(getattr(args, 'teacher_min_hold', 5))} "
            f"switch_margin={float(getattr(args, 'teacher_switch_margin', 5e-5)):.6f}"
        )
    n_steps = len(df)

    aux_mode = str(getattr(args, "aux_mode", "logit_gate"))
    if aux_mode not in {"none", "state_fusion", "logit_gate"}:
        aux_mode = "logit_gate"
    model = ActorCritic(
        obs_dim=obs_all.shape[1],
        hidden=64,
        n_actions=len(ACTIONS),
        aux_mode=aux_mode,
        gate_coef=float(getattr(args, "gate_coef", 0.8)),
        gate_uncertainty_coef=float(getattr(args, "gate_uncertainty_coef", 0.30)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    target1_all_t = torch.from_numpy(target1_all_np)
    target5_all_t = torch.from_numpy(target5_all_np)
    mask1_all_t = torch.from_numpy(mask1_all_np)
    mask5_all_t = torch.from_numpy(mask5_all_np)

    # Behavior cloning warm-start (imitation learning)
    obs_bc_t = torch.from_numpy(obs_all.astype(np.float32))
    lbl_bc_t = torch.from_numpy(teacher_labels.astype(np.int64))
    cls_w_np = _label_class_weights(
        teacher_labels, n_classes=len(ACTIONS), balance_pow=float(getattr(args, "bc_class_balance_pow", 0.5))
    )
    cls_w_t = torch.from_numpy(cls_w_np)
    bc_epochs = max(1, int(getattr(args, "bc_epochs", 24)))
    bc_batch = max(32, int(getattr(args, "bc_batch_size", 256)))
    for _ in range(bc_epochs):
        perm = torch.randperm(obs_bc_t.shape[0])
        for s in range(0, obs_bc_t.shape[0], bc_batch):
            idx = perm[s : s + bc_batch]
            logits, _, pred1, pred5, _ = model.forward_with_aux(obs_bc_t[idx])
            gidx = idx
            loss_bc = F.cross_entropy(
                logits,
                lbl_bc_t[idx],
                weight=cls_w_t,
                label_smoothing=float(getattr(args, "bc_label_smoothing", 0.02)),
            )
            loss_aux = _masked_mse(pred1, target1_all_t[gidx], mask1_all_t[gidx]) + _masked_mse(
                pred5, target5_all_t[gidx], mask5_all_t[gidx]
            )
            loss_rank = _pairwise_rank_loss(pred5, target5_all_t[gidx], mask5_all_t[gidx], max_pairs=256)
            loss = loss_bc + float(getattr(args, "aux_coef", 0.20)) * loss_aux + float(getattr(args, "rank_coef", 0.05)) * loss_rank
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    anchor_model = copy.deepcopy(model).eval()
    if int(getattr(args, "freeze_backbone_during_ppo", 1)) == 1:
        for p in model.backbone.parameters():
            p.requires_grad = False

    ep_returns = []
    global_ptr = 0
    for epoch in range(int(args.epochs)):
        obs_buf, idx_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], [], []
        running_return = 0.0

        for _ in range(int(args.horizon)):
            idx = global_ptr % n_steps
            obs = torch.from_numpy(obs_all[idx]).float().unsqueeze(0)
            with torch.no_grad():
                logits, value = model(obs)
                dist = Categorical(logits=logits)
                act = dist.sample()
                logp = dist.log_prob(act)
            action_id = int(act.item())
            reward = float(rewards_table[idx, action_id])
            done = (idx == n_steps - 1)

            obs_buf.append(obs_all[idx])
            idx_buf.append(idx)
            act_buf.append(action_id)
            logp_buf.append(float(logp.item()))
            val_buf.append(float(value.item()))
            rew_buf.append(reward)
            done_buf.append(done)
            running_return += reward
            global_ptr += 1
            if done:
                ep_returns.append(running_return)
                running_return = 0.0

        adv, ret = _compute_gae(
            rewards=np.asarray(rew_buf, dtype=np.float32),
            values=np.asarray(val_buf, dtype=np.float32),
            dones=np.asarray(done_buf, dtype=np.bool_),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.from_numpy(np.asarray(obs_buf, dtype=np.float32))
        idx_t = torch.from_numpy(np.asarray(idx_buf, dtype=np.int64))
        act_t = torch.from_numpy(np.asarray(act_buf, dtype=np.int64))
        old_logp_t = torch.from_numpy(np.asarray(logp_buf, dtype=np.float32))
        adv_t = torch.from_numpy(adv.astype(np.float32))
        ret_t = torch.from_numpy(ret.astype(np.float32))

        n = obs_t.shape[0]
        bc_coef = float(getattr(args, "bc_coef_start", 0.15)) + (
            float(getattr(args, "bc_coef_end", 0.03)) - float(getattr(args, "bc_coef_start", 0.15))
        ) * (epoch / max(1, int(args.epochs) - 1))
        ent_start = float(getattr(args, "entropy_coef_start", float(args.entropy_coef)))
        ent_end = float(getattr(args, "entropy_coef_end", float(args.entropy_coef)))
        ent_coef = ent_end + 0.5 * (ent_start - ent_end) * (
            1.0 + np.cos(np.pi * (epoch / max(1, int(args.epochs) - 1)))
        )
        for _ in range(int(args.ppo_epochs)):
            perm = torch.randperm(n)
            for s in range(0, n, int(args.batch_size)):
                idx = perm[s : s + int(args.batch_size)]
                logits, values, pred1, pred5, _ = model.forward_with_aux(obs_t[idx])
                with torch.no_grad():
                    anchor_logits, _ = anchor_model(obs_t[idx])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act_t[idx])
                entropy = dist.entropy().mean()
                ratio = torch.exp(logp - old_logp_t[idx])
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1.0 - float(args.clip_eps), 1.0 + float(args.clip_eps)) * adv_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, ret_t[idx])
                # Batch action diversity regularizer against single-action collapse
                probs = torch.softmax(logits, dim=1).mean(dim=0)
                uniform = torch.full_like(probs, 1.0 / probs.numel())
                diversity_loss = F.mse_loss(probs, uniform)
                bc_loss = F.cross_entropy(
                    logits,
                    lbl_bc_t[idx],
                    weight=cls_w_t,
                    label_smoothing=float(getattr(args, "bc_label_smoothing", 0.02)),
                )
                gidx = idx_t[idx]
                aux_loss = _masked_mse(pred1, target1_all_t[gidx], mask1_all_t[gidx]) + _masked_mse(
                    pred5, target5_all_t[gidx], mask5_all_t[gidx]
                )
                rank_loss = _pairwise_rank_loss(pred5, target5_all_t[gidx], mask5_all_t[gidx], max_pairs=128)
                kl_anchor = F.kl_div(
                    torch.log_softmax(logits, dim=1),
                    torch.softmax(anchor_logits, dim=1),
                    reduction="batchmean",
                )
                loss = (
                    policy_loss
                    + float(args.value_coef) * value_loss
                    - float(ent_coef) * entropy
                    + float(args.diversity_coef) * diversity_loss
                    + bc_coef * bc_loss
                    + float(getattr(args, "anchor_kl_coef", 0.08)) * kl_anchor
                    + float(getattr(args, "aux_coef", 0.20)) * aux_loss
                    + float(getattr(args, "rank_coef", 0.05)) * rank_loss
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        if (epoch + 1) % 10 == 0:
            recent = ep_returns[-5:] if ep_returns else [0.0]
            print(f"[timing_rl_train] epoch={epoch+1}/{args.epochs} recent_return_mean={float(np.mean(recent)):.6f}")

    # Evaluate greedy policy on full sequence
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(obs_all).float())
        greedy_act = torch.argmax(logits, dim=1).cpu().numpy()
    greedy_reward = float(np.mean([rewards_table[i, int(a)] for i, a in enumerate(greedy_act)]))
    action_counts = {str(i): int((greedy_act == i).sum()) for i in range(len(ACTIONS))}
    action_ratio = {k: (v / max(1, len(greedy_act))) for k, v in action_counts.items()}
    teacher_counts = {str(i): int((teacher_labels == i).sum()) for i in range(len(ACTIONS))}
    teacher_ratio = {k: (v / max(1, len(teacher_labels))) for k, v in teacher_counts.items()}

    state_dict = {k: v.detach().cpu().numpy().astype(float).tolist() for k, v in model.state_dict().items()}
    out = {
        "policy_type": (
            "timing_ppo_actor_critic_v4"
            if aux_mode == "logit_gate"
            else ("timing_ppo_actor_critic_v3" if aux_mode == "state_fusion" else "timing_ppo_actor_critic_v2")
        ),
        "feature_cols": FEATURE_COLS,
        "feature_mean": mu.tolist(),
        "feature_std": sd.tolist(),
        "actions": ACTIONS,
        "state_dict": state_dict,
        "model_config": {
            "aux_mode": aux_mode,
            "hidden": 64,
            "gate_coef": float(getattr(args, "gate_coef", 0.8)),
            "gate_uncertainty_coef": float(getattr(args, "gate_uncertainty_coef", 0.30)),
        },
        "train_meta": {
            "epochs": int(args.epochs),
            "horizon": int(args.horizon),
            "ppo_epochs": int(args.ppo_epochs),
            "batch_size": int(args.batch_size),
            "bc_epochs": int(bc_epochs),
            "teacher_mode": str(getattr(args, "teacher_mode", "oracle")),
            "teacher_policy_path": str(getattr(args, "teacher_policy_path", "")),
            "teacher_action_counts": teacher_counts,
            "teacher_action_ratio": teacher_ratio,
            "greedy_avg_reward": greedy_reward,
            "greedy_action_counts": action_counts,
            "greedy_action_ratio": action_ratio,
        },
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[timing_rl_train] policy saved to {out_path}")
    print(f"[timing_rl_train] greedy_avg_reward={greedy_reward:.6f}")
    print(f"[timing_rl_train] greedy_action_ratio={action_ratio}")


if __name__ == "__main__":
    main()
