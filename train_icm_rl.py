"""
server.py keeps collecting transitions (screenshot_before, actionId, screenshot_after).
Every 50 rows (in your newer server), it spawns train_icm_rl.py.

train_icm_rl.py:
    1. trains the ICM module (EncoderICM + ForwardModel + InverseModel)
    2. trains the policy (EncoderPolicy + ActorCritic) using intrinsic reward from the ICM prediction error
    3. saves updated weights back to saved_models/*.pth
    4. Then server.py loads those same saved_models/encoder_policy.pth + saved_models/actor_critic.pth to choose the next action.
"""
import time
import csv
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import pandas as pd

from models import (
    EncoderICM, ForwardModel, InverseModel,
    EncoderPolicy, ActorCritic
)

# ===== Reproducibility / Seed =====
import os, random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# NOTE: Do not call set_seed at import time. When this module is imported from
# `server.py` we want a lightweight import that does not perform I/O or set
# global randomness deterministically. `run_rollout()` (below) will set the
# seed when training is invoked.
SEED = None



# ===================
# Config
# ===================
csv_path = "record.csv"
image_folder = r"screenshots"

log_csv = "training_log.csv"   # 日志 CSV
rollout_size = int(os.getenv("rollout_size", "50"))             
# 一次 roll-out 的步数 T (we only have rollout_size samples in each training, hence set batch_size=rollout_size)
# ("rollout_size", "50")), the "50" is a default fallback value.
# Meaning:
# If your environment variable rollout_size is set (e.g., "60"), 
# then os.getenv(...) returns "60" → int(...) becomes 60.
# So, need to manually update "50" when the rollout size changes in server.py.
action_dim = 4
latent_dim = 128 # dimension of encoded state
gamma = 0.99 # discount factor
lr = 1e-4 # learning rate

save_path = "saved_models"
os.makedirs(save_path, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================================
# Dataset
# ========================================
class CuriosityDataset(Dataset):
    def __init__(self, csv_path, img_dir):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_path)
        self.df = self.df[
            (self.df['actionId'] != -1)
            & (self.df['s_t_img'].notnull())
            & (self.df['s_t1_img'].notnull())
        ].reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_before = Image.open(
            os.path.join(self.img_dir, row['s_t_img'])
        ).convert('RGB')

        img_after = Image.open(
            os.path.join(self.img_dir, row['s_t1_img'])
        ).convert('RGB')

        before = self.transform(img_before)
        after = self.transform(img_after)
        action = torch.tensor(int(row['actionId']), dtype=torch.long)

        return before, action, after


# ========================================
# Create/load models
# ========================================
def load_or_create(model_class, name):
    path = os.path.join(save_path, f"{name}.pth")
    model = model_class().to(DEVICE)
    if os.path.exists(path):
        print(f"🔁 Loading {name} from {path}")
        # 如果你的 torch 版本太低，去掉 weights_only=True
        state = torch.load(path, map_location=DEVICE)
        if isinstance(state, dict):
            model.load_state_dict(state)
        else:
            model.load_state_dict(state.state_dict())
    else:
        print(f"✨ No {name} found, creating new")
    return model


def save_model(model, name):
    torch.save(model.state_dict(), os.path.join(save_path, f"{name}.pth"))


# ========================================
# Rollout Training + per-sample Logging
# ========================================
def train_one_rollout(total_rows: int | None = None):
    
    # ========= Load models =========
    encoder_icm = load_or_create(EncoderICM, "encoder_icm")
    forward_model = load_or_create(lambda: ForwardModel(action_dim), "forward")
    inverse_model = load_or_create(lambda: InverseModel(action_dim), "inverse")

    encoder_policy = load_or_create(EncoderPolicy, "encoder_policy")
    actor_critic = load_or_create(lambda: ActorCritic(action_dim), "actor_critic")


    # ========= 初始化 CSV Header =========
    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", 
                "seed",
                "step",

                # per-sample losses
                "forward_loss_each",
                "inverse_loss_each",
                "icm_loss_each",
                "policy_loss_each",
                "value_loss_each",
                "entropy_each",
                "total_loss_each",

                # rewards
                "intrinsic_reward",
                "return",
                "advantage",

                # policy probs
                "prob_up",
                "prob_down",
                "prob_left",
                "prob_right",

                "chosen_action",
                
                # invers accuracy metrics
                # Inverse accuracy measures how often the inverse model correctly predicts 
                # the action that caused the transition,given the encoded states ϕ(st) and ϕ(st+1).
                # It is computed by taking the inverse model’s predicted action (argmax of its logits) 
                # and comparing it to the true action, then averaging the correct predictions (0/1) over a batch or rollout.
                "inverse_acc_each",
                "inverse_acc_rollout",
                
                "inv_grad_mean_abs", 
                "inv_grad_max_abs"
            ])

    # ========= 取最近 50 条样本 =========
    dataset = CuriosityDataset(csv_path, image_folder)
    if len(dataset) < rollout_size:
        print("❌ Not enough samples for rollout.")
        return

    last_50 = Subset(dataset, range(len(dataset) - rollout_size, len(dataset)))
    loader = DataLoader(last_50, batch_size=rollout_size, shuffle=False)

    before_batch, action_batch, after_batch = next(iter(loader))
    before_batch = before_batch.to(DEVICE)   # [T, C, H, W]
    after_batch = after_batch.to(DEVICE)     # [T, C, H, W]
    action_batch = action_batch.to(DEVICE)   # [T]

    T = rollout_size

    # # ========= Load models =========
    # encoder_icm = load_or_create(EncoderICM, "encoder_icm")
    # forward_model = load_or_create(lambda: ForwardModel(action_dim), "forward")
    # inverse_model = load_or_create(lambda: InverseModel(action_dim), "inverse")

    # encoder_policy = load_or_create(EncoderPolicy, "encoder_policy")
    # actor_critic = load_or_create(lambda: ActorCritic(action_dim), "actor_critic")

    # ========= Optimizer =========
    params = (
        list(encoder_icm.parameters()) +
        list(forward_model.parameters()) +
        list(inverse_model.parameters()) +
        list(encoder_policy.parameters()) +
        list(actor_critic.parameters())
    )
    optimizer = Adam(params, lr=lr)

    # ==========================================
    # 1) ICM: Forward + Inverse
    # ==========================================
    # Compute latents using the ICM encoder:
    phi_t = encoder_icm(before_batch)      # [T, D]
    phi_next = encoder_icm(after_batch)    # [T, D]

    # Forward
    a_onehot = F.one_hot(action_batch, num_classes=action_dim).float().to(DEVICE)  # [T, A]
    # Forward model predicts next latent:
    phi_pred = forward_model(phi_t, a_onehot)  # [T, D]

    # Inverse model predicts the action:
    inv_logits = inverse_model(phi_t, phi_next)  # [T, A]
    with torch.no_grad():
        inv_pred = inv_logits.argmax(dim=1)               # [T]
        inverse_acc_each = (inv_pred == action_batch).float()  # [T] each is 0/1
        inverse_acc_rollout = inverse_acc_each.mean().item()   # scalar


    # Intrinsic reward per step is:
    # mean squared error per sample between phi_pred and phi_next
    intrinsic_reward = ((phi_pred - phi_next)**2).mean(dim=1).detach()  # [T]

    # per-sample forward loss
    forward_loss_each = ((phi_pred - phi_next)**2).mean(dim=1)  # [T]
    loss_forward = forward_loss_each.mean()

    # per-sample inverse loss
    inverse_loss_each = F.cross_(inv_logits, action_batch, reduction='none')  # [T]
    loss_inverse = inverse_loss_each.mean()
    
    #icm_loss_each = forward_loss_each + 0.1 * inverse_loss_each
    icm_loss_each = 0.5 * forward_loss_each + 0.5 * inverse_loss_each  # [T]
    loss_icm = icm_loss_each.mean()

    # ==========================================
    # 2) ActorCritic on intrinsic reward
    # ==========================================
    phi_policy = encoder_policy(before_batch)      # [T, D]
    logits, values = actor_critic(phi_policy)      # logits:[T,A], values:[T,1]

    probs = F.softmax(logits, dim=1)              # [T, A]
    log_probs = F.log_softmax(logits, dim=1)      # [T, A]
    chosen_log_probs = log_probs[range(T), action_batch]  # [T]

    # per-sample entropy
    entropy_each = -(probs * log_probs).sum(dim=1)   # [T]
    entropy = entropy_each.mean()

    # Discounted returns from intrinsic_reward
    Returns_list = []
    R = 0.0
    # intrinsic_reward 是 [T]，CPU 上
    rew_np = intrinsic_reward.cpu().numpy()
    for r in reversed(rew_np):
        R = r + gamma * R
        Returns_list.insert(0, R)
    Returns = torch.tensor(Returns_list, dtype=torch.float32, device=DEVICE)  # [T]

    values_flat = values.squeeze(-1)  # [T]
    advantages = Returns - values_flat  # [T]

    # per-sample policy loss & value loss
    policy_loss_each = -(chosen_log_probs * advantages.detach())  # [T]
    policy_loss = policy_loss_each.mean()

    value_loss_each = advantages.pow(2)  # [T]
    value_loss = value_loss_each.mean()

    # ==========================================
    # 3) Combine total loss（batch 用 mean）
    # ==========================================
    total_loss_each = (
        policy_loss_each +
        0.5 * value_loss_each +
        icm_loss_each -
        0.01 * entropy_each
    )  # [T]

    total_loss = total_loss_each.mean()

    # ==========================================
    # Backprop
    # ==========================================
    optimizer.zero_grad()
    total_loss.backward()
    # ===== Gradient stats (after backward, before step) =====
    with torch.no_grad():
        inv_grads = []
        inv_grads_max = []

        for p in inverse_model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                inv_grads.append(g.abs().mean().item())
                inv_grads_max.append(g.abs().max().item())

        inv_grad_mean_abs = (sum(inv_grads) / len(inv_grads)) if inv_grads else 0.0
        inv_grad_max_abs  = max(inv_grads_max) if inv_grads_max else 0.0
    optimizer.step()

    # ========= Debug 输出（batch 级别）=========
    print("\n================ Rollout Debug ================")
    print(f"Forward Loss (mean) : {loss_forward.item():.6f}")
    print(f"Inverse Loss (mean) : {loss_inverse.item():.6f}")
    print(f"ICM Loss (mean)     : {loss_icm.item():.6f}")
    print(f"Policy Loss (mean)  : {policy_loss.item():.6f}")
    print(f"Value Loss (mean)   : {value_loss.item():.6f}")
    print(f"Entropy (mean)      : {entropy.item():.6f}")
    print(f"Total Loss (mean)   : {total_loss.item():.6f}")

    print(f"Curiosity Reward → mean: {intrinsic_reward.mean().item():.6f}, "
          f"max: {intrinsic_reward.max().item():.6f}, "
          f"min: {intrinsic_reward.min().item():.6f}")
    
    print(f"Inverse Acc (rollout) : {inverse_acc_rollout*100:.2f}%")


    # 动作计数
    unique, counts = torch.unique(action_batch.cpu(), return_counts=True)
    action_dict = dict(zip(unique.tolist(), counts.tolist()))
    print(f"Rollout Action Counts: {action_dict}")
    print("================================================\n")

    # ==========================================
    # 写 per-sample log 到 CSV
    # ==========================================
    ts = time.time()
    with open(log_csv, "a", newline="") as f:
        writer = csv.writer(f)
        probs_cpu = probs.detach().cpu()
        Returns_cpu = Returns.detach().cpu()
        adv_cpu = advantages.detach().cpu()
        fl_cpu = forward_loss_each.detach().cpu()
        il_cpu = inverse_loss_each.detach().cpu()
        icm_cpu = icm_loss_each.detach().cpu()
        pl_cpu = policy_loss_each.detach().cpu()
        vl_cpu = value_loss_each.detach().cpu()
        ent_cpu = entropy_each.detach().cpu()
        tl_cpu = total_loss_each.detach().cpu()
        ir_cpu = intrinsic_reward.detach().cpu()
        act_cpu = action_batch.detach().cpu()
        inv_acc_each_cpu = inverse_acc_each.detach().cpu()

        for i in range(T):
            writer.writerow([
                ts, SEED, i,

                float(fl_cpu[i]),
                float(il_cpu[i]),
                float(icm_cpu[i]),
                float(pl_cpu[i]),
                float(vl_cpu[i]),
                float(ent_cpu[i]),
                float(tl_cpu[i]),

                float(ir_cpu[i]),
                float(Returns_cpu[i]),
                float(adv_cpu[i]),

                float(probs_cpu[i][0]),
                float(probs_cpu[i][1]),
                float(probs_cpu[i][2]),
                float(probs_cpu[i][3]),

                int(act_cpu[i]),
                
                float(inv_acc_each_cpu[i]),
                inverse_acc_rollout,
                
                inv_grad_mean_abs,
                inv_grad_max_abs
            ])

    # ==========================================
    # Save models (regular save + optional snapshot when total_rows hits multiples of 1000)
    # ==========================================
    def _save_and_maybe_snapshot(model, name, total_rows_val):
        # regular save
        save_model(model, name)

        # save snapshot with metadata when total_rows is a multiple of 1000
        try:
            if total_rows_val is not None and int(total_rows_val) % 1000 == 0:
                ts = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{int(total_rows_val)}_{ts}.pth"
                torch.save(model.state_dict(), os.path.join(save_path, filename))
                print(f"💾 Snapshot saved: {filename}")
        except Exception as e:
            print(f"⚠️ Failed to write snapshot for {name}:", e)

    _save_and_maybe_snapshot(encoder_icm, "encoder_icm", total_rows)
    _save_and_maybe_snapshot(forward_model, "forward", total_rows)
    _save_and_maybe_snapshot(inverse_model, "inverse", total_rows)
    _save_and_maybe_snapshot(encoder_policy, "encoder_policy", total_rows)
    _save_and_maybe_snapshot(actor_critic, "actor_critic", total_rows)

    print("🎉 Rollout trained & logged.")


# ===========================
# Main
# ===========================
def run_rollout(seed: int | None = None, rollout_size_override: int | None = None, total_rows: int | None = None) -> dict:
    """
    Run one rollout training. This function is safe to import and call from
    `server.py` (it does not run at import time).

    Args:
        seed: optional seed to set for reproducibility. If None, falls back to
              the SEED environment variable or 42.
        rollout_size_override: if provided, overrides module-level rollout_size
                                for this run.

    Returns:
        dict: {"status": "ok"} on success or {"status": "error", "error": str}
    """
    global SEED, rollout_size
    # determine seed
    if seed is None:
        SEED = int(os.environ.get("SEED", "42"))
    else:
        SEED = int(seed)

    # set deterministic behavior now (not at import)
    set_seed(SEED)
    print(f"🌱 SEED = {SEED}")

    # optional rollout_size override
    if rollout_size_override is not None:
        try:
            rollout_size = int(rollout_size_override)
            print(f"🔧 rollout_size overridden to {rollout_size}")
        except Exception:
            print("⚠️ Invalid rollout_size_override, ignoring.")

    try:
        train_one_rollout(total_rows=total_rows)
        return {"status": "ok"}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("❌ Exception during run_rollout:", e)
        print(tb)
        return {"status": "error", "error": str(e), "traceback": tb}


if __name__ == "__main__":
    # keep CLI behaviour for backwards compatibility
    res = run_rollout()
    if res.get("status") != "ok":
        raise SystemExit(1)
