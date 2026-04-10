# server_test.py — Parallel testing mode for the parallel-training architecture
# ENABLE_TRAINING is always False here; no .pth files are updated.
#
# Launch (one experiment at a time, or several on different ports):
#   ACTIVE_EXP=exp_1 PORT=5001 python server_test.py
#   ACTIVE_EXP=exp_2 PORT=5002 python server_test.py

import threading
import csv
import json
import os
import random
import base64
import time

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from PIL import Image
from io import BytesIO
from torchvision import transforms

from models import EncoderICM, ForwardModel, InverseModel, EncoderPolicy, ActorCritic
from train_icm_rl import (
    get_exp_config,
    set_seed,
    DEVICE,
)

# =========================================
# Active experiment
# =========================================
ACTIVE_EXP = os.environ.get("ACTIVE_EXP", "exp_1")
exp_cfg    = get_exp_config(ACTIVE_EXP)
SAVE_DIR   = exp_cfg["save_dir"]

print(f"🔬 Active experiment : {ACTIVE_EXP}")
print(f"📁 Save dir          : {SAVE_DIR}")

# =========================================
# Flask + CORS
# =========================================
app = Flask(__name__)
CORS(app)

# =========================================
# Seed
# =========================================
SEED = int(os.environ.get("SEED", "42"))
set_seed(SEED)
print(f"🌱 SEED = {SEED}")

# =========================================
# Constants
# =========================================
SCREENSHOT_DIR  = exp_cfg["image_dir"]
CSV_FILE        = exp_cfg["csv_path"]   # still log transitions (no training though)
ACTION_DIM      = exp_cfg["action_dim"]
ENABLE_TRAINING = False                 # hard-coded OFF in testing mode

TESTING_LOG_FILE = f"testing_log_{ACTIVE_EXP}.csv"

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# =========================================
# Image preprocessing
# =========================================
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# =========================================
# CSV helpers
# =========================================
csv_lock = threading.Lock()

def init_csv():
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            header = [
                "sessionId", "seed", "frontend_seed",
                "model", "actionId", "greedy_action",
                "s_t_img", "s_t1_img",
                "after_yaw", "after_pitch",
                "delta_yaw", "delta_pitch",
                "init_yaw", "init_pitch",
                "reward",
            ]
            header.extend([f"prob_{i}" for i in range(ACTION_DIM)])
            writer.writerow(header)

init_csv()


def init_testing_csv():
    """Create testing_log_<exp>.csv if it doesn't exist yet."""
    if os.path.exists(TESTING_LOG_FILE):
        return
    with open(TESTING_LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "timestamp", "sessionId", "seed", "frontend_seed",
            "experiment", "model", "actionId",
            "s_t_img", "s_t1_img",
            "forward_loss", "inverse_loss", "entropy",
        ]
        header.extend([f"prob_{i}" for i in range(ACTION_DIM)])
        writer.writerow(header)

init_testing_csv()


def append_csv_row(
    sessionId, seed, frontend_seed, modelName,
    actionId, greedy_action, s_t_img, s_t1_img,
    after, delta, initial, reward, probs=None
):
    row = [
        sessionId, seed, frontend_seed, modelName,
        actionId, greedy_action, s_t_img, s_t1_img,
        after.get("yaw"), after.get("pitch"),
        delta.get("yaw"), delta.get("pitch"),
        initial.get("yaw"), initial.get("pitch"),
        reward,
    ]
    row.extend([float(p) for p in probs] if probs is not None
               else [""] * ACTION_DIM)
    with csv_lock:
        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow(row)


def append_testing_row(
    sessionId, seed, frontend_seed, modelName, actionId,
    s_t_img, s_t1_img,
    forward_loss, inverse_loss, entropy, probs=None
):
    row = [
        time.time(), sessionId, seed, frontend_seed,
        ACTIVE_EXP, modelName, actionId,
        s_t_img, s_t1_img,
        forward_loss, inverse_loss, entropy,
    ]
    row.extend([float(p) for p in probs] if probs is not None
               else [""] * ACTION_DIM)
    with csv_lock:
        with open(TESTING_LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow(row)


# =========================================
# ICM evaluation (forward + inverse loss)
# — loaded once at startup and kept frozen
# =========================================
_enc_icm   = EncoderICM(latent_dim=exp_cfg["latent_dim"]).to(DEVICE)
_fwd_model = ForwardModel(action_dim=ACTION_DIM, latent_dim=exp_cfg["latent_dim"]).to(DEVICE)
_inv_model = InverseModel(action_dim=ACTION_DIM, latent_dim=exp_cfg["latent_dim"]).to(DEVICE)

def _load_icm_weights():
    for model_obj, name in [(_enc_icm, "encoder_icm"),
                             (_fwd_model, "forward"),
                             (_inv_model, "inverse")]:
        path = os.path.join(SAVE_DIR, f"{name}.pth")
        if os.path.exists(path):
            model_obj.load_state_dict(
                torch.load(path, map_location=DEVICE, weights_only=True))
            print(f"  ✅ Loaded {name} from {path}")
        else:
            print(f"  ⚠️  {name}.pth not found — using random weights")
        model_obj.eval()

_load_icm_weights()


def compute_icm_losses(img_before_path: str, img_after_path: str, actionId: int):
    """Returns (forward_loss, inverse_loss) as floats, or ('', '') on failure."""
    try:
        before_t = preprocess(Image.open(img_before_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        after_t  = preprocess(Image.open(img_after_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        a        = torch.tensor([int(actionId)], dtype=torch.long, device=DEVICE)
        a_onehot = F.one_hot(a, num_classes=ACTION_DIM).float()

        with torch.no_grad():
            phi_t      = _enc_icm(before_t)
            phi_next   = _enc_icm(after_t)
            phi_pred   = _fwd_model(phi_t, a_onehot)
            fwd_loss   = float(((phi_pred - phi_next) ** 2).mean().item())
            inv_logits = _inv_model(phi_t, phi_next)
            inv_loss   = float(F.cross_entropy(inv_logits, a).item())

        return fwd_loss, inv_loss
    except Exception as e:
        print(f"⚠️ ICM loss computation failed: {e}")
        return "", ""


# =========================================
# Policy inference
# — weights reloaded each call so we always
#   use the latest checkpoint on disk
# =========================================
def run_policy(img_path: str) -> tuple[int, int, list]:
    """
    Returns (next_action, greedy_action, probs_list).
    Both actions are greedy (no epsilon, no multinomial) in test mode.
    """
    enc   = EncoderPolicy(latent_dim=exp_cfg["latent_dim"]).to(DEVICE)
    actor = ActorCritic(action_dim=ACTION_DIM, latent_dim=exp_cfg["latent_dim"]).to(DEVICE)

    enc_path   = os.path.join(SAVE_DIR, "encoder_policy.pth")
    actor_path = os.path.join(SAVE_DIR, "actor_critic.pth")

    if os.path.exists(enc_path):
        enc.load_state_dict(torch.load(enc_path, map_location=DEVICE, weights_only=True))
    if os.path.exists(actor_path):
        actor.load_state_dict(torch.load(actor_path, map_location=DEVICE, weights_only=True))

    enc.eval()
    actor.eval()

    img    = Image.open(img_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        phi           = enc(tensor)
        logits, _     = actor(phi)
        probs         = F.softmax(logits, dim=1).cpu().numpy()[0]
        greedy_action = int(np.argmax(probs))
        next_action   = greedy_action  # fully greedy — no sampling in test mode

    return next_action, greedy_action, probs.tolist()


def compute_entropy(probs_list: list) -> float:
    probs_t = torch.tensor(probs_list, dtype=torch.float32)
    log_p   = torch.log(probs_t.clamp(min=1e-8))
    return float(-(probs_t * log_p).sum().item())


# =========================================
# Routes
# =========================================
@app.route("/record", methods=["POST", "OPTIONS"])
def record():
    if request.method == "OPTIONS":
        resp = make_response()
        resp.headers["Access-Control-Allow-Origin"]  = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return resp

    data          = request.get_json()
    sessionId     = data.get("sessionId", "")
    frontend_seed = data.get("frontend_seed", "")
    after         = data.get("afterAngles")   or {}
    delta         = data.get("deltaAngles")   or {}
    initial       = data.get("initialAngles") or {}
    modelName     = data["modelName"]
    s_t_img       = data["s_t_img"]
    s_t1_img      = data["s_t1_img"]
    actionId      = data["actionId"]

    print(f"[{ACTIVE_EXP}|TEST] action={actionId}  "
          f"yaw {initial.get('yaw')} → {after.get('yaw')}")

    # ── Save screenshots ───────────────────────────────────────
    CROP_SIZE = 350

    def crop_and_save(data_url: str, filename: str):
        if not data_url.startswith("data:image"):
            return
        img_bytes = base64.b64decode(data_url.split(",", 1)[1])
        img       = Image.open(BytesIO(img_bytes)).convert("RGB")
        w, h      = img.size
        l = (w - CROP_SIZE) // 2
        t = (h - CROP_SIZE) // 2
        img.crop((l, t, l + CROP_SIZE, t + CROP_SIZE)).save(
            os.path.join(SCREENSHOT_DIR, filename), format="PNG")

    if actionId != -1:
        crop_and_save(data.get("imgData1", ""), s_t_img)
        crop_and_save(data.get("imgData2", ""), s_t1_img)

        for _ in range(10):
            if (os.path.exists(os.path.join(SCREENSHOT_DIR, s_t_img)) and
                    os.path.exists(os.path.join(SCREENSHOT_DIR, s_t1_img))):
                break
            time.sleep(0.1)

    # ── Policy inference ───────────────────────────────────────
    next_action   = None
    greedy_action = None
    probs_for_row = None
    entropy       = ""
    fwd_loss      = ""
    inv_loss      = ""

    if actionId != -1:
        # Policy
        try:
            next_action, greedy_action, probs_for_row = run_policy(
                os.path.join(SCREENSHOT_DIR, s_t1_img))
            entropy = compute_entropy(probs_for_row)
            print(f"[{ACTIVE_EXP}|TEST] greedy={greedy_action}  "
                  f"entropy={entropy:.4f}  probs={[f'{p:.3f}' for p in probs_for_row]}")
        except Exception as e:
            print(f"[{ACTIVE_EXP}|TEST] ⚠️ Policy inference failed: {e}")

        # ICM losses — diagnostic only, weights not updated
        fwd_loss, inv_loss = compute_icm_losses(
            os.path.join(SCREENSHOT_DIR, s_t_img),
            os.path.join(SCREENSHOT_DIR, s_t1_img),
            actionId,
        )
        print(f"[{ACTIVE_EXP}|TEST] fwd_loss={fwd_loss}  inv_loss={inv_loss}")

    # ── Log to record.csv (same schema as training server) ────
    append_csv_row(
        sessionId=sessionId, seed=SEED, frontend_seed=frontend_seed,
        modelName=modelName, actionId=actionId, greedy_action=greedy_action,
        s_t_img=s_t_img, s_t1_img=s_t1_img,
        after=after, delta=delta, initial=initial,
        reward=0, probs=probs_for_row,
    )

    # ── Log to testing_log.csv ─────────────────────────────────
    append_testing_row(
        sessionId=sessionId, seed=SEED, frontend_seed=frontend_seed,
        modelName=modelName, actionId=actionId,
        s_t_img=s_t_img, s_t1_img=s_t1_img,
        forward_loss=fwd_loss, inverse_loss=inv_loss,
        entropy=entropy, probs=probs_for_row,
    )

    # No training trigger — ENABLE_TRAINING is always False here
    return jsonify({"status": "ok", "next_action": next_action})


@app.route("/save-progress", methods=["POST", "OPTIONS"])
def save_progress():
    if request.method == "OPTIONS":
        resp = make_response()
        resp.headers["Access-Control-Allow-Origin"]  = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return resp

    try:
        data             = request.get_json()
        processed_models = data.get("processedModels", [])
        progress_file    = f"processed_models_{ACTIVE_EXP}_test.json"
        with open(progress_file, "w") as f:
            json.dump({
                "processedModels": processed_models,
                "experiment":      ACTIVE_EXP,
                "mode":            "testing",
                "lastUpdated":     time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)
        print(f"✅ Progress saved: {len(processed_models)} models")
        return jsonify({"status": "success", "count": len(processed_models)})
    except Exception as e:
        print(f"❌ Save progress error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", "5001"))
    print(f"🧪 Testing mode — ENABLE_TRAINING=False, no weights will be updated")
    app.run(port=PORT)