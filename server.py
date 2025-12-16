
import subprocess
import threading
import pandas as pd
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import base64
import os
import csv
from PIL import Image
from io import BytesIO
import torch
import torch.nn.functional as F
from torchvision import transforms

import time
#also triggers train_icm_rl.py. for training.

app = Flask(__name__) #Creates a Flask server
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#Picks device: cuda if available, else CPU.

import cv2
import numpy as np

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

def append_csv_row(sessionId, modelName, actionId,
                   s_t_img, s_t1_img,
                   after, delta, initial,
                   reward, probs=None):
    """
    Append one row to record.csv, including softmax probs if provided.
    probs should be a 1D list/array of length ACTION_DIM.
    """
    row = [
        sessionId,
        modelName,
        actionId,
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
            'model',
            'actionId',
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
    # imports and uses models.py every request to load EncoderPolicy and ActorCritic for inference (not “trigger”, but it depends on it).
    from models import EncoderPolicy, ActorCritic

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
            next_action = int(torch.argmax(torch.tensor(probs)))
            print(f"Actual next action (the greddy action): {next_action}")
            next_action = int(torch.multinomial(probs_tensor, 1).item()) 
            #按照它现在的概率分布去随机取
            #e.g. probs = [0.1, 0.2, 0.6, 0.1]，那么动作2被选中的概率就是60%
            #next_action = int(torch.argmax(torch.tensor(probs)))
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
        modelName=modelName,
        actionId=actionId,
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
        if len(df) % 50 == 0:  # 满 50 行 (triggers at 50, 100, 150… rows)
            print("🚀 50 rows reached, triggering training...")
            subprocess.Popen(["python", "train_icm_rl.py"])#(comment后触发训练不启动)
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