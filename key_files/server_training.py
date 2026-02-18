import subprocess
import threading
import pandas as pd
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import base64
import os, random
import csv
from PIL import Image
from io import BytesIO
import torch
import torch.nn.functional as F
from torchvision import transforms
import time
import cv2
import numpy as np
#(Models.py) also triggers train_icm_rl.py. for training.
# imports and uses models.py every request to load EncoderPolicy and ActorCritic for inference (not “trigger”, but it depends on it).
from models import (
    EncoderICM, ForwardModel, InverseModel,
    EncoderPolicy, ActorCritic
)
from train_icm_rl import load_or_create, save_model, action_dim

app = Flask(__name__) #Creates a Flask server
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#Picks device: cuda if available, else CPU.

# ========= Load models =========
encoder_icm = load_or_create(EncoderICM, "encoder_icm")
forward_model = load_or_create(lambda: ForwardModel(action_dim), "forward")
inverse_model = load_or_create(lambda: InverseModel(action_dim), "inverse")

encoder_policy = load_or_create(EncoderPolicy, "encoder_policy")
actor_critic = load_or_create(lambda: ActorCritic(action_dim), "actor_critic")

# ==========================================
# Save models
# ==========================================
save_model(encoder_icm, "encoder_icm")
save_model(forward_model, "forward")
save_model(inverse_model, "inverse")
save_model(encoder_policy, "encoder_policy")
save_model(actor_critic, "actor_critic")

# ===== Reproducibility / Seed =====
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = int(os.environ.get("SEED", "42"))
set_seed(SEED)
print(f"🌱 SEED = {SEED}")

# Then when you start the server, you can do:
# PowerShell
#   $env:SEED=123; python server.py
# That makes the server’s randomness consistent across runs.

def save_abstract_image(img_bytes, save_path):
    """保存抽象过的图像（灰度 + 边缘检测 + 归一化）(not used currently)"""

    # 从 bytes 加载图片
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    img = np.array(img)  # 转 numpy

    # ---------- 1) 裁剪中心 ----------
    CROP_SIZE =350
    h, w, _ = img.shape
    left = (w - CROP_SIZE) // 2
    top = (h - CROP_SIZE) // 2
    right = left + CROP_SIZE
    bottom = top + CROP_SIZE
    img = img[top:bottom, left:right]

    # ---------- 2) 转灰度 ----------
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ---------- 3) 边缘检测 (Canny) ----------
    edges = cv2.Canny(gray, 50, 150)

    # ---------- 4) 归一化到 0-255 ----------
    edges = edges.astype(np.uint8)

    # ---------- 5) 保存 ----------
    cv2.imwrite(save_path, edges)


# 图像预处理函数（和训练保持一致）
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
]) #This matches what the policy encoder expects (and matches training in train_icm_rl.py).

# ✅ 启用 CORS 支持
CORS(app)

os.makedirs(r'screenshots', exist_ok=True)
CSV_FILE = 'record.csv'
ACTION_DIM = 4  # since ActorCritic(action dimension = 4)

LOCK_FILE = "training.lock"
ENABLE_TRAINING = True   # set False when you want “no .pth updates”, aka disable training and “freeze weights”
rollout_size = 50
TRAIN_ROWS_EVERY = rollout_size  # alias so the meaning is clear


def append_csv_row(sessionId, seed, frontend_seed, modelName, actionId, greedy_action,
                   s_t_img, s_t1_img,
                   after, delta, initial,
                   reward, probs=None):
    """
    Append one row to record.csv, including softmax probs if provided.
    probs should be a 1D list/array of length ACTION_DIM.
    """
    row = [
        sessionId,
        seed,
        frontend_seed,
        modelName,
        actionId,
        greedy_action,
        s_t_img,
        s_t1_img,
        after.get('yaw'), after.get('pitch'),
        delta.get('yaw'), delta.get('pitch'),
        initial.get('yaw'), initial.get('pitch'),
        reward
    ]

    # add prob_0..prob_{ACTION_DIM-1}
    if probs is not None:
        row.extend([float(p) for p in probs])
    else:
        row.extend([''] * ACTION_DIM)  # keep column count consistent

    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row) 
        
# 可选：初始化 CSV
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            'sessionId',
            'seed',
            'frontend_seed',
            'model',
            'actionId',
            'greedy_action',
            's_t_img',
            's_t1_img',
            'after_yaw', 'after_pitch',
            'delta_yaw', 'delta_pitch',
            'init_yaw', 'init_pitch',
            'reward'
        ]
        # add prob columns: prob_0..prob_3
        header.extend([f'prob_{i}' for i in range(ACTION_DIM)])
        writer.writerow(header)
        
def start_training_if_allowed(env, total_rows=None):
    if not ENABLE_TRAINING:
        print("🧊 ENABLE_TRAINING=False → training disabled (no .pth updates).")
        return False

    if os.path.exists(LOCK_FILE):
        # optional: stale lock recovery
        try:
            age = time.time() - os.path.getmtime(LOCK_FILE)
            if age > 3600:
                print("⚠️ Stale lock detected (>1h). Removing lock.")
                os.remove(LOCK_FILE)
            else:
                print("⏳ Training already running (lock exists) → skip.")
                return False
        except Exception as e:
            print("⏳ Lock exists but couldn't check age → skip:", e)
            return False
        
    # create lock
    with open(LOCK_FILE, "w") as f:
        f.write("running")

    # Run training in-process in a background thread (safer and faster than
    # spawning a new Python process). We import train_icm_rl lazily so the
    # module isn't heavy at server startup.
    try:
        import train_icm_rl
    except Exception as e:
        print("⚠️ Failed to import train_icm_rl:", e)
        try:
            os.remove(LOCK_FILE)
        except Exception:
            pass
        return False

    def _bg_train():
        try:
            seed = env.get("SEED") if isinstance(env, dict) else None
            rollout_override = env.get("rollout_size") if isinstance(env, dict) else None
            print(f"▶️ Starting in-process training (seed={seed}, rollout_size={rollout_override})")
            res = train_icm_rl.run_rollout(seed=seed, rollout_size_override=rollout_override, total_rows=total_rows)
            print("🔁 Training finished:", res)
        except Exception as e:
            print("❌ Training thread exception:", e)
        finally:
            try:
                os.remove(LOCK_FILE)
                print("🔓 Training lock removed.")
            except FileNotFoundError:
                pass

    t = threading.Thread(target=_bg_train, daemon=True)
    t.start()
    print(f"✅ Training thread started (name={t.name})")
    return True


@app.route('/record', methods=['POST', 'OPTIONS'])
def record():
    print("Received request with method:", request.method)
    # ✅ 手动处理预检请求（关键）
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = 'http://localhost:5173'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        return response

    # ✅ 正常 POST 请求
    data = request.get_json()
    frontend_seed = data.get("frontend_seed", "")
    reward = 0
    probs_for_row = None  # will be filled once policy runs
    sessionId = data.get('sessionId', '')
    after = (data.get('afterAngles') or {})
    delta = (data.get('deltaAngles') or {})
    initial = (data.get('initialAngles') or {})  # present only for init row
    modelName = data['modelName']
    s_t_img = data['s_t_img']
    s_t1_img = data['s_t1_img']
    actionId = data['actionId']
    print(f"Saving image: {initial.get('yaw')}, {initial.get('pitch')} -> {after.get('yaw')}, {after.get('pitch')}，delta: {delta.get('yaw')}, {delta.get('pitch')}")
    
    
    if actionId != -1:
        if data.get('imgData1', '').startswith('data:image'):
            imgData1 = data['imgData1'].split(',')[1]
            image_bytes = base64.b64decode(imgData1)
            image = Image.open(BytesIO(image_bytes)).convert('RGB')

            # ✅ 裁剪中心
            CROP_SIZE = 350
            w, h = image.size
            left = (w - CROP_SIZE) // 2
            top = (h - CROP_SIZE) // 2
            right = left + CROP_SIZE
            bottom = top + CROP_SIZE
            image = image.crop((left, top, right, bottom))

            image.save(rf'screenshots\{s_t_img}', format='PNG')

        if data.get('imgData2', '').startswith('data:image'):
            imgData2 = data['imgData2'].split(',')[1]
            image_bytes = base64.b64decode(imgData2)
            image = Image.open(BytesIO(image_bytes)).convert('RGB')

            # ✅ 裁剪中心 
            CROP_SIZE = 350
            w, h = image.size
            left = (w - CROP_SIZE) // 2
            top = (h - CROP_SIZE) // 2
            right = left + CROP_SIZE
            bottom = top + CROP_SIZE
            image = image.crop((left, top, right, bottom))

            image.save(rf'screenshots\{s_t1_img}', format='PNG')
            
            for attempt in range(10):
                if (os.path.exists(rf'screenshots\{s_t_img}') and
                    os.path.exists(rf'screenshots\{s_t1_img}')):
                    break
                time.sleep(0.1)
            else:
                print(f"⚠️ Timeout waiting for {s_t_img} or {s_t1_img} to appear.")
                

    """
    Run the policy to choose the NEXT action
    """
    # Load EncoderPolicy + ActorCritic
    encoder_policy = EncoderPolicy().to(device)
    actor_model = ActorCritic(action_dim=4).to(device)

    try:
        encoder_policy.load_state_dict(torch.load(
            "saved_models/encoder_policy.pth", map_location=device, weights_only=True))
        actor_model.load_state_dict(torch.load(
            "saved_models/actor_critic.pth", map_location=device, weights_only=True))

        encoder_policy.eval()
        actor_model.eval()

        next_action = None

        # Load s_t1 image (It loads the AFTER image (s_t1_img) and uses it as the current state for choosing the next action, it stores probs_for_row = probs)
        img2 = Image.open(rf'screenshots\{s_t1_img}').convert('RGB')
        img2 = preprocess(img2).unsqueeze(0).to(device)

        # Encode state with policy encoder
        with torch.no_grad():
            phi_t1 = encoder_policy(img2)
            logits, value = actor_model(phi_t1)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            probs_for_row = probs


        # Epsilon-greedy multinomial sampling
        epsilon = 0.0#随机率调整
        import random as rnd
        if rnd.random() < epsilon:
            next_action = rnd.randint(0, 3)
            print(f"🎲 Random exploration → {next_action}")
        else:
            probs_tensor = torch.tensor(probs, dtype=torch.float32)
            
            # # independent generator for action sampling
            # gen = torch.Generator(device='cpu')
            # gen.manual_seed(int(time.time() * 1000) % 2**31)  # or any per-step seed
            
            greedy_action = int(np.argmax(probs))  # or torch.argmax(torch.tensor(probs)).item()
            print(f"Greedy action (argmax): {greedy_action}")
            next_action = int(torch.multinomial(probs_tensor, 1).item())
            print(f"Sampled action (multinomial): {next_action}")
            #按照它现在的概率分布去随机取
            #e.g. probs = [0.1, 0.2, 0.6, 0.1]，那么动作2被选中的概率就是60%
            #next_action = int(torch.argmax(torch.tensor(probs))) # this line makes the data collection greedy
            #固定取概率最大的,always pick the action with the highest probability (greedy)
            #e.g. probs = [0.1, 0.2, 0.6, 0.1]，那么动作2一定会被选中
            #print(f"🎮 Policy (multinomial) → {next_action}")
            
        print(f"Softmax probs: {probs}")
        ''' print(f"Image: {s_t1_img}")
        print(f"Logits: {logits.cpu().numpy()}")
        print(f"Softmax probs: {probs}")
        print(f"Latent phi_t1 (first 5 dims): {phi_t1[0][:5].cpu().numpy()}")
        print(f"Epsilon: {epsilon}")'''
    except Exception as e:
        print("⚠️ Failed to run ActorCritic policy:", e)
        next_action = None
    
    # 👉 Now log this transition + policy distribution into record.csv
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
        reward=reward,
        probs=probs_for_row  # can be None if policy failed
    )
    print(f"Appended row to {CSV_FILE} for session {sessionId}, action {actionId}")
    
    # === 自动触发训练 ===
    try:
        df = pd.read_csv(CSV_FILE)
        total_rows = len(df)#total rows of data in record.csv
        if total_rows % TRAIN_ROWS_EVERY == 0 or total_rows == 1:
            print(f"🚀 {total_rows} rows reached, attempting training...")
            env = os.environ.copy()
            env["SEED"] = str(SEED)
            env["rollout_size"] = str(rollout_size)

            started = start_training_if_allowed(env, total_rows=total_rows)
            if started:
                threading.Timer(5.0, lambda: print("✅ Training process started.")).start()
    except Exception as e:
        print("⚠️ Failed to trigger training:", e)

    
    return jsonify({'status': 'ok', 'next_action': next_action})

@app.route('/save-progress', methods=['POST', 'OPTIONS'])
def save_progress():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = 'http://localhost:5173'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        return response
    
    try:
        data = request.get_json()
        processed_models = data.get('processedModels', [])
        
        # Save to JSON file
        import json
        progress_file = 'processed_models.json'
        with open(progress_file, 'w') as f:
            json.dump({
                'processedModels': processed_models,
                'lastUpdated': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        
        print(f'✅ Progress saved: {len(processed_models)} models processed')
        return jsonify({'status': 'success', 'count': len(processed_models)})
    
    except Exception as e:
        print(f'❌ Error saving progress: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)