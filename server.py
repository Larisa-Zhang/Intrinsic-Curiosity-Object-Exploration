
# Launch 4 separate servers:
#     ACTIVE_EXP=exp_1 PORT=5001 python server.py
#     ACTIVE_EXP=exp_2 PORT=5002 python server.py
#     ACTIVE_EXP=exp_3 PORT=5003 python server.py
#     ACTIVE_EXP=exp_4 PORT=5004 python server.py

# Launch 4 separate frontends:
#     VITE_PORT=5173 VITE_BACKEND_PORT=5001 npx vite
#     VITE_PORT=5174 VITE_BACKEND_PORT=5002 npx vite
#     VITE_PORT=5175 VITE_BACKEND_PORT=5003 npx vite
#     VITE_PORT=5176 VITE_BACKEND_PORT=5004 npx vite
import threading
import pandas as pd
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import base64
import os
import random
import csv
import json
from PIL import Image
from io import BytesIO
import torch
import torch.nn.functional as F
from torchvision import transforms
import time
import numpy as np

from models import EncoderICM, ForwardModel, InverseModel, EncoderPolicy, ActorCritic
from train_icm_rl import (
    load_or_create,
    save_model,
    get_exp_config,
    run_rollout,
    set_seed,
    DEVICE,
)
import os
print(f"🔍 RAW ENV: ACTIVE_EXP={os.environ.get('ACTIVE_EXP', 'NOT SET')}  PORT={os.environ.get('PORT', 'NOT SET')}")

#lock until all files are confirmed written before the server starts accepting requests

def init_model_weights(exp_cfg: dict):
    save_dir   = exp_cfg["save_dir"]
    latent_dim = exp_cfg["latent_dim"]
    action_dim = exp_cfg["action_dim"]

    models = {
        "encoder_icm":    lambda: EncoderICM(latent_dim=latent_dim),
        "forward":        lambda: ForwardModel(action_dim=action_dim, latent_dim=latent_dim),
        "inverse":        lambda: InverseModel(action_dim=action_dim, latent_dim=latent_dim),
        "encoder_policy": lambda: EncoderPolicy(latent_dim=latent_dim),
        "actor_critic":   lambda: ActorCritic(action_dim=action_dim, latent_dim=latent_dim),
    }
    for name, fn in models.items():
        path = os.path.join(save_dir, f"{name}.pth")
        if not os.path.exists(path):
            try:
                torch.save(fn().state_dict(), path)
                assert os.path.exists(path)
                print(f"  🆕 Initialised: {name} → {path}")
            except Exception as e:
                print(f"  ❌ Failed to initialise {name}: {e}")
                raise
        else:
            print(f"  ✅ Found existing: {name}")
            
# =========================================
# Active experiment — change this one line
# to switch which experiment the server uses
# =========================================
ACTIVE_EXP = os.environ.get("ACTIVE_EXP", "exp_1")
exp_cfg    = get_exp_config(ACTIVE_EXP)
SAVE_DIR   = exp_cfg["save_dir"]

print("🔧 Initialising model weights...")
init_model_weights(exp_cfg)
print("✅ All weights ready, starting server...")

# =========================================
# Flask app
# =========================================
app = Flask(__name__)
CORS(app)

# =========================================
# Seed
# =========================================
SEED = int(os.environ.get("SEED", "42"))
set_seed(SEED)
print(f"🌱 SEED = {SEED}")
print(f"🔬 Active experiment: {ACTIVE_EXP}")
print(f"📁 Save dir: {SAVE_DIR}")

# =========================================
# Constants from active experiment config
# =========================================
SCREENSHOT_DIR    = exp_cfg["image_dir"]
CSV_FILE          = exp_cfg["csv_path"]
ACTION_DIM        = exp_cfg["action_dim"]
ROLLOUT_SIZE      = exp_cfg["rollout_size"]
TRAIN_ROWS_EVERY  = ROLLOUT_SIZE
ENABLE_TRAINING   = True
LOCK_FILE         = f"training_{ACTIVE_EXP}.lock"  # separate lock per experiment

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
csv_lock = threading.Lock()   # thread-safe CSV writes

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


# =========================================
# Training trigger
# =========================================
def start_training_if_allowed(total_rows: int | None = None):
    if not ENABLE_TRAINING:
        print("🧊 Training disabled.")
        return False

    if os.path.exists(LOCK_FILE):
        try:
            age = time.time() - os.path.getmtime(LOCK_FILE)
            if age > 3600:
                print("⚠️ Stale lock (>1h), removing.")
                os.remove(LOCK_FILE)
            else:
                print("⏳ Training already running, skipping.")
                return False
        except Exception as e:
            print("⏳ Lock check failed:", e)
            return False

    with open(LOCK_FILE, "w") as f:
        f.write("running")

    def _bg_train():
        try:
            print(f"▶️  Background training: {ACTIVE_EXP} "
                  f"(seed={SEED}, total_rows={total_rows})")
            result = run_rollout(
                exp_name=ACTIVE_EXP,
                seed=SEED,
                rollout_size_override=ROLLOUT_SIZE,
                total_rows=total_rows,
            )
            print("🔁 Training result:", result)
        except Exception as e:
            print("❌ Training thread error:", e)
        finally:
            try:
                os.remove(LOCK_FILE)
                print("🔓 Lock removed.")
            except FileNotFoundError:
                pass

    t = threading.Thread(target=_bg_train, daemon=True)
    t.start()
    print(f"✅ Training thread started: {t.name}")
    return True


# =========================================
# Policy inference helper
# =========================================
def run_policy(img_path: str) -> tuple[int, int, list]:
    """
    Load latest policy weights, run inference on img_path.
    Returns (next_action, greedy_action, probs_list).
    """
    enc   = EncoderPolicy(latent_dim=exp_cfg["latent_dim"]).to(DEVICE)
    actor = ActorCritic(
        action_dim=ACTION_DIM,
        latent_dim=exp_cfg["latent_dim"]
    ).to(DEVICE)

    enc_path   = os.path.join(SAVE_DIR, "encoder_policy.pth")
    actor_path = os.path.join(SAVE_DIR, "actor_critic.pth")

    if os.path.exists(enc_path):
        enc.load_state_dict(
            torch.load(enc_path, map_location=DEVICE, weights_only=True))
    if os.path.exists(actor_path):
        actor.load_state_dict(
            torch.load(actor_path, map_location=DEVICE, weights_only=True))

    enc.eval()
    actor.eval()

    img    = Image.open(img_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        phi          = enc(tensor)
        logits, _    = actor(phi)
        probs        = F.softmax(logits, dim=1).cpu().numpy()[0]
        greedy_action = int(np.argmax(probs))
        probs_tensor  = torch.tensor(probs, dtype=torch.float32)
        next_action   = int(torch.multinomial(probs_tensor, 1).item())

    return next_action, greedy_action, probs.tolist()


# =========================================
# Routes
# =========================================
@app.route("/record", methods=["POST", "OPTIONS"])
def record():
    if request.method == "OPTIONS":
        resp = make_response()
        resp.headers["Access-Control-Allow-Origin"]  = "http://localhost:5173"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return resp

    data         = request.get_json()
    sessionId    = data.get("sessionId", "")
    frontend_seed = data.get("frontend_seed", "")
    after        = data.get("afterAngles")   or {}
    delta        = data.get("deltaAngles")   or {}
    initial      = data.get("initialAngles") or {}
    modelName    = data["modelName"]
    s_t_img      = data["s_t_img"]
    s_t1_img     = data["s_t1_img"]
    actionId     = data["actionId"]

    print(f"[{ACTIVE_EXP}] action={actionId}  "
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

        # wait for both files to appear on disk
        for _ in range(10):
            if (os.path.exists(os.path.join(SCREENSHOT_DIR, s_t_img)) and
                    os.path.exists(os.path.join(SCREENSHOT_DIR, s_t1_img))):
                break
            time.sleep(0.1)

    # ── Policy inference ───────────────────────────────────────
    next_action   = None
    greedy_action = None
    probs_for_row = None

    if actionId != -1:
        try:
            next_action, greedy_action, probs_for_row = run_policy(
                os.path.join(SCREENSHOT_DIR, s_t1_img))
            print(f"[{ACTIVE_EXP}] greedy={greedy_action}  "
                  f"sampled={next_action}  probs={probs_for_row}")
        except Exception as e:
            print(f"[{ACTIVE_EXP}] ⚠️ Policy inference failed: {e}")

    # ── Log to CSV ─────────────────────────────────────────────
    append_csv_row(
        sessionId=sessionId,
        seed=SEED,
        frontend_seed=frontend_seed,
        modelName=modelName,
        actionId=actionId,
        greedy_action=greedy_action,
        s_t_img=s_t_img,
        s_t1_img=s_t1_img,
        after=after,
        delta=delta,
        initial=initial,
        reward=0,
        probs=probs_for_row,
    )

    # ── Trigger training every ROLLOUT_SIZE rows ───────────────
    try:
        with csv_lock:
            total_rows = sum(1 for _ in open(CSV_FILE)) - 1  # minus header
        if total_rows % TRAIN_ROWS_EVERY == 0 or total_rows == 1:
            print(f"[{ACTIVE_EXP}] 🚀 {total_rows} rows → triggering training")
            start_training_if_allowed(total_rows=total_rows)
    except Exception as e:
        print(f"[{ACTIVE_EXP}] ⚠️ Training trigger failed: {e}")

    return jsonify({"status": "ok", "next_action": next_action})


@app.route("/save-progress", methods=["POST", "OPTIONS"])
def save_progress():
    if request.method == "OPTIONS":
        resp = make_response()
        resp.headers["Access-Control-Allow-Origin"]  = "http://localhost:5173"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return resp

    try:
        data             = request.get_json()
        processed_models = data.get("processedModels", [])
        progress_file    = "processed_models.json"
        with open(progress_file, "w") as f:
            json.dump({
                "processedModels": processed_models,
                "experiment":      ACTIVE_EXP,
                "lastUpdated":     time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)
        print(f"✅ Progress saved: {len(processed_models)} models")
        return jsonify({"status": "success", "count": len(processed_models)})
    except Exception as e:
        print(f"❌ Save progress error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    PORT       = int(os.environ.get("PORT", "5001"))
    ACTIVE_EXP = os.environ.get("ACTIVE_EXP", "exp_1")
    app.run(port=PORT)