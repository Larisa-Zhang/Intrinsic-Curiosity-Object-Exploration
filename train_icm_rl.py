"""
train_icm_rl.py

Flow:
  server.py collects transitions → every rollout_size rows triggers run_rollout()
  run_rollout() → train_one_rollout() → saves updated *.pth to exp_cfg["save_dir"]
  server.py loads those *.pth for inference
"""

import os
import csv
import time
import random
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import pandas as pd
import yaml

from models import (
    EncoderICM, ForwardModel, InverseModel,
    EncoderPolicy, ActorCritic
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================
# Seed
# =========================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================
# Config
# =========================================
def load_config(yaml_path: str = "experiments.yaml") -> dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def get_exp_config(exp_name: str, yaml_path: str = "experiments.yaml") -> dict:
    """Returns exp_cfg dict for the given experiment name."""
    config = load_config(yaml_path)
    for exp in config["experiments"]:
        if exp["name"] == exp_name:
            os.makedirs(exp["save_dir"], exist_ok=True)
            os.makedirs(exp["log_dir"],  exist_ok=True)
            return exp
    raise ValueError(f"Experiment '{exp_name}' not found in {yaml_path}")


def list_experiments(yaml_path: str = "experiments.yaml") -> list[str]:
    config = load_config(yaml_path)
    return [exp["name"] for exp in config["experiments"]]


# =========================================
# Model helpers
# =========================================
def load_or_create(model_fn, name: str, save_dir: str):
    """
    model_fn: zero-argument callable that returns an nn.Module
              e.g. lambda: EncoderICM(latent_dim=128)
    """
    path  = os.path.join(save_dir, f"{name}.pth")
    model = model_fn().to(DEVICE)
    if os.path.exists(path):
        print(f"  🔁 Loading {name} from {path}")
        state = torch.load(path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
    else:
        print(f"  ✨ No checkpoint for {name}, creating fresh")
    return model


def save_model(model, name: str, save_dir: str):
    torch.save(model.state_dict(), os.path.join(save_dir, f"{name}.pth"))


def maybe_snapshot(model, name: str, save_dir: str, total_rows: int | None):
    """Always saves latest; also saves a timestamped snapshot every 1000 rows."""
    save_model(model, name, save_dir)
    try:
        if total_rows is not None and int(total_rows) % 1000 == 0:
            ts    = time.strftime("%Y%m%d_%H%M%S")
            fname = f"{name}_{int(total_rows)}_{ts}.pth"
            torch.save(model.state_dict(), os.path.join(save_dir, fname))
            print(f"  💾 Snapshot saved: {fname}")
    except Exception as e:
        print(f"  ⚠️ Snapshot failed for {name}: {e}")


# =========================================
# Dataset
# =========================================
class CuriosityDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str):
        self.img_dir = img_dir
        df = pd.read_csv(csv_path)
        self.df = df[
            (df["actionId"] != -1)
            & df["s_t_img"].notnull()
            & df["s_t1_img"].notnull()
        ].reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        before = self.transform(
            Image.open(os.path.join(self.img_dir, row["s_t_img"])).convert("RGB"))
        after  = self.transform(
            Image.open(os.path.join(self.img_dir, row["s_t1_img"])).convert("RGB"))
        action = torch.tensor(int(row["actionId"]), dtype=torch.long)
        return before, action, after


# =========================================
# Log CSV
# =========================================
def init_log_csv(log_path: str, action_dim: int):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)  
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "exp_name", "seed", "step",
                "forward_loss", "inverse_loss", "icm_loss",
                "policy_loss", "value_loss", "entropy", "total_loss",
                "intrinsic_reward", "return", "advantage",
                *[f"prob_{i}" for i in range(action_dim)],
                "chosen_action",
                "inverse_acc_each", "inverse_acc_rollout",
                "inv_grad_mean_abs", "inv_grad_max_abs",
            ])


# =========================================
# Core training
# =========================================
def train_one_rollout(
    exp_cfg:    dict,
    seed:       int       = 42,
    total_rows: int | None = None,
):
    exp_name         = exp_cfg["name"]
    action_dim       = exp_cfg["action_dim"]
    latent_dim       = exp_cfg["latent_dim"]
    gamma            = exp_cfg["gamma"]
    rollout_size     = exp_cfg["rollout_size"]
    lr              = float(exp_cfg["lr"])
    beta             = float(exp_cfg["beta"])
    entropy_coef     = float(exp_cfg["entropy_coef"])
    value_loss_coef  = float(exp_cfg["value_loss_coef"])
    pred_lr_scale    = float(exp_cfg["prediction_lr_scale"])
    share_encoder    = exp_cfg["share_encoder"]
    unsup_type       = exp_cfg["unsup_type"]
    save_dir         = exp_cfg["save_dir"]
    log_path         = os.path.join(exp_cfg["log_dir"], "training_log.csv")

    print(f"\n[{exp_name}] rollout | "
          f"lr={lr}  β={beta}  pred_lr_scale={pred_lr_scale}  "
          f"unsup={unsup_type}  share_enc={share_encoder}")
    
    init_log_csv(log_path, action_dim)

    # ── Build models ──────────────────────────────────────────
    encoder_icm   = load_or_create(
        lambda: EncoderICM(latent_dim=latent_dim),
        "encoder_icm", save_dir)
    forward_model = load_or_create(
        lambda: ForwardModel(action_dim=action_dim, latent_dim=latent_dim),
        "forward", save_dir)
    inverse_model = load_or_create(
        lambda: InverseModel(action_dim=action_dim, latent_dim=latent_dim),
        "inverse", save_dir)

    if share_encoder:
        encoder_policy = encoder_icm          # shared weights, no separate file
    else:
        encoder_policy = load_or_create(
            lambda: EncoderPolicy(latent_dim=latent_dim),
            "encoder_policy", save_dir)

    actor_critic = load_or_create(
        lambda: ActorCritic(action_dim=action_dim, latent_dim=latent_dim),
        "actor_critic", save_dir)

    # ── Optimizer: separate lr groups ─────────────────────────
    if share_encoder:
        icm_params = (list(forward_model.parameters()) +
                      list(inverse_model.parameters()))
    else:
        icm_params = (list(encoder_icm.parameters()) +
                      list(forward_model.parameters()) +
                      list(inverse_model.parameters()))

    policy_params = (list(encoder_policy.parameters()) +
                     list(actor_critic.parameters()))

    optimizer = Adam([
        {"params": icm_params,    "lr": lr * pred_lr_scale},
        {"params": policy_params, "lr": lr},
    ])

    # ── Load data ─────────────────────────────────────────────
    dataset = CuriosityDataset(exp_cfg["csv_path"], exp_cfg["image_dir"])
    if len(dataset) < rollout_size:
        print(f"[{exp_name}] ❌ Not enough samples "
              f"({len(dataset)} < {rollout_size})")
        return

    last_n = Subset(dataset, range(len(dataset) - rollout_size, len(dataset)))
    loader = DataLoader(last_n, batch_size=rollout_size, shuffle=False)
    before_batch, action_batch, after_batch = next(iter(loader))

    before_batch = before_batch.to(DEVICE)
    after_batch  = after_batch.to(DEVICE)
    action_batch = action_batch.to(DEVICE)
    T            = rollout_size

    # ── ICM forward ───────────────────────────────────────────
    phi_t      = encoder_icm(before_batch)
    phi_next   = encoder_icm(after_batch)
    a_onehot   = F.one_hot(action_batch, action_dim).float()
    phi_pred   = forward_model(phi_t, a_onehot)
    inv_logits = inverse_model(phi_t, phi_next)

    with torch.no_grad():
        inv_pred            = inv_logits.argmax(dim=1)
        inverse_acc_each    = (inv_pred == action_batch).float()
        inverse_acc_rollout = inverse_acc_each.mean().item()

    # ── Curiosity reward ──────────────────────────────────────
    if unsup_type == "action":
        intrinsic_reward = ((phi_pred - phi_next) ** 2).mean(dim=1).detach()
    elif unsup_type == "state":
        intrinsic_reward = ((before_batch - after_batch) ** 2).mean(dim=[1, 2, 3]).detach()
    else:  # "none"
        intrinsic_reward = torch.zeros(T, device=DEVICE)

    # ── Losses ────────────────────────────────────────────────
    forward_loss_each = ((phi_pred - phi_next) ** 2).mean(dim=1)
    inverse_loss_each = F.cross_entropy(inv_logits, action_batch, reduction="none")
    icm_loss_each     = (beta * forward_loss_each +
                         (1 - beta) * inverse_loss_each)

    # ── Actor-Critic ──────────────────────────────────────────
    phi_policy       = encoder_policy(before_batch)
    logits, values   = actor_critic(phi_policy)
    probs            = F.softmax(logits,     dim=1)
    log_probs        = F.log_softmax(logits, dim=1)
    chosen_log_probs = log_probs[range(T), action_batch]
    entropy_each     = -(probs * log_probs).sum(dim=1)

    # discounted returns
    rew_np = intrinsic_reward.cpu().numpy()
    R, returns_list = 0.0, []
    for r in reversed(rew_np):
        R = r + gamma * R
        returns_list.insert(0, R)
    Returns     = torch.tensor(returns_list, dtype=torch.float32, device=DEVICE)
    values_flat = values.squeeze(-1)
    advantages  = Returns - values_flat

    policy_loss_each = -(chosen_log_probs * advantages.detach())
    value_loss_each  = advantages.pow(2)

    total_loss_each = (
        policy_loss_each
        + value_loss_coef  * value_loss_each
        + icm_loss_each
        - entropy_coef     * entropy_each
    )
    total_loss = total_loss_each.mean()

    # ── Backprop ──────────────────────────────────────────────
    optimizer.zero_grad()
    total_loss.backward()

    with torch.no_grad():
        inv_grads     = [p.grad.abs().mean().item()
                         for p in inverse_model.parameters()
                         if p.grad is not None]
        inv_grads_max = [p.grad.abs().max().item()
                         for p in inverse_model.parameters()
                         if p.grad is not None]
        inv_grad_mean_abs = sum(inv_grads) / len(inv_grads) if inv_grads else 0.0
        inv_grad_max_abs  = max(inv_grads_max)              if inv_grads_max else 0.0

    optimizer.step()

    # ── Console summary ───────────────────────────────────────
    print(f"[{exp_name}] "
          f"fwd={forward_loss_each.mean().item():.4f}  "
          f"inv={inverse_loss_each.mean().item():.4f}  "
          f"pol={policy_loss_each.mean().item():.4f}  "
          f"ent={entropy_each.mean().item():.4f}  "
          f"total={total_loss.item():.4f}  "
          f"inv_acc={inverse_acc_rollout * 100:.1f}%")

    unique, counts = torch.unique(action_batch.cpu(), return_counts=True)
    print(f"[{exp_name}] Action counts: "
          f"{dict(zip(unique.tolist(), counts.tolist()))}")

    # ── Per-sample CSV log ────────────────────────────────────
    init_log_csv(log_path, action_dim)
    ts         = time.time()
    probs_cpu  = probs.detach().cpu()
    Returns_cpu = Returns.detach().cpu()
    adv_cpu    = advantages.detach().cpu()

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        for i in range(T):
            writer.writerow([
                ts, exp_name, seed, i,
                float(forward_loss_each[i].item()),
                float(inverse_loss_each[i].item()),
                float(icm_loss_each[i].item()),
                float(policy_loss_each[i].item()),
                float(value_loss_each[i].item()),
                float(entropy_each[i].item()),
                float(total_loss_each[i].item()),
                float(intrinsic_reward[i].item()),
                float(Returns_cpu[i]),
                float(adv_cpu[i]),
                *[float(probs_cpu[i][j]) for j in range(action_dim)],
                int(action_batch[i].item()),
                float(inverse_acc_each[i].item()),
                inverse_acc_rollout,
                inv_grad_mean_abs,
                inv_grad_max_abs,
            ])

    # ── Save models ───────────────────────────────────────────
    for model, name in [
        (encoder_icm,   "encoder_icm"),
        (forward_model, "forward"),
        (inverse_model, "inverse"),
        (actor_critic,  "actor_critic"),
    ]:
        maybe_snapshot(model, name, save_dir, total_rows)

    if not share_encoder:
        maybe_snapshot(encoder_policy, "encoder_policy", save_dir, total_rows)

    print(f"[{exp_name}] ✅ Rollout complete.")


# =========================================
# Entry points
# =========================================
def run_rollout(
    exp_name:             str       = "exp_1",
    seed:                 int | None = None,
    rollout_size_override: int | None = None,
    total_rows:           int | None = None,
) -> dict:
    """Called by server.py for online training after every rollout."""
    try:
        exp_cfg = get_exp_config(exp_name)
    except ValueError as e:
        return {"status": "error", "error": str(e)}

    if seed is None:
        seed = int(os.environ.get("SEED", "42"))
    set_seed(seed)

    if rollout_size_override is not None:
        exp_cfg = dict(exp_cfg)
        exp_cfg["rollout_size"] = int(rollout_size_override)

    try:
        train_one_rollout(exp_cfg, seed=seed, total_rows=total_rows)
        return {"status": "ok"}
    except Exception as e:
        print(f"[{exp_name}] ❌ Exception:", e)
        print(traceback.format_exc())
        return {"status": "error", "error": str(e)}


def run_all_experiments(seed: int = 42):
    
    """
Called by run_experiments.py for offline parallel training.
    
Online path:
  User explores 3D object → server.py /record route
  → every 50 rows → run_rollout("exp_1")
  → train_one_rollout(exp_cfg) in background thread
  → saves updated .pth to saved_models/exp_1/

Offline path:
  python run_experiments.py
  → run_all_experiments()
  → spawns 4 processes, each calls train_one_rollout(exp_cfg)
  → same saved_models/ directories
    
    
    """
    import torch.multiprocessing as mp

    config = load_config()
    set_seed(seed)

    def _worker(exp_cfg):
        # each subprocess re-sets seed for reproducibility
        set_seed(seed)
        train_one_rollout(exp_cfg, seed=seed)

    mp.set_start_method("spawn", force=True)
    processes = []
    for exp_cfg in config["experiments"]:
        os.makedirs(exp_cfg["save_dir"], exist_ok=True)
        os.makedirs(exp_cfg["log_dir"],  exist_ok=True)
        p = mp.Process(target=_worker, args=(exp_cfg,))
        p.start()
        processes.append(p)
        print(f"🚀 Launched: {exp_cfg['name']}")

    for p in processes:
        p.join()

    print("✅ All experiments complete!")


if __name__ == "__main__":
    run_rollout()