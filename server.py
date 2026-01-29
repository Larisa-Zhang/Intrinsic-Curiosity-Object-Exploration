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
from curiosity_model import Encoder, ForwardModel
import time

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = Encoder().to(device)
forward_model = ForwardModel().to(device)

checkpoint = torch.load('curiosity_model_best.pth', map_location=device)
encoder.load_state_dict(checkpoint['encoder'])
forward_model.load_state_dict(checkpoint['forward_model'])

encoder.eval()
forward_model.eval()

# 图像预处理函数（和训练保持一致）
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
]) #This matches what the policy encoder expects (and matches training in train_icm_rl.py).

# ✅ 启用 CORS 支持
CORS(app)

os.makedirs(r'screenshots', exist_ok=True)
CSV_FILE = 'record.csv'

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
            'init_yaw', 'init_pitch', 'reward'
        ])

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
                
    reward = None  # 默认空
    if actionId != -1:
        try:
            # ✅ 加载裁剪后的图像
            img1 = Image.open(rf'D:\screenshotsnew\{s_t_img}').convert('RGB')
            img2 = Image.open(rf'D:\screenshotsnew\{s_t1_img}').convert('RGB')

            img1 = preprocess(img1).unsqueeze(0).to(device)  # [1, 3, 128, 128]
            img2 = preprocess(img2).unsqueeze(0).to(device)

            state = encoder(img1)         # [1, 128]
            next_state = encoder(img2)    # [1, 128]

            action_tensor = torch.tensor([actionId], dtype=torch.long).to(device)
            action_onehot = F.one_hot(action_tensor, num_classes=4).float()  # [1, 4]

            pred_next = forward_model(state, action_onehot)  # [1, 128]

            reward = F.mse_loss(pred_next, next_state).item()
            reward = float(reward)  # 转为普通 float，方便 JSON 序列化
            reward = reward * 1000
            print(f"✅ Reward computed: {reward:.4f}")
            print("State t      :", state[0][:5].cpu().detach().numpy())
            print("State t+1    :", next_state[0][:5].cpu().detach().numpy())
            print("Predicted t+1:", pred_next[0][:5].cpu().detach().numpy())
            print("MSE          :", F.mse_loss(pred_next, next_state).item())

        except Exception as e:
            print("❌ Failed to compute reward:", e)
            reward = None

    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            sessionId,
            modelName,
            actionId,
            s_t_img,
            s_t1_img,
            after.get('yaw'), after.get('pitch'),
            delta.get('yaw'), delta.get('pitch'),
            initial.get('yaw'), initial.get('pitch'),
            reward
        ])

    from policy_model import PolicyNetwork
    policy = PolicyNetwork().to(device)
    policy.load_state_dict(torch.load('policy_model.pth', map_location=device))
    policy.eval()

    next_action = None
    try:
        img2 = Image.open(rf'D:\screenshotsnew\{s_t1_img}').convert('RGB')
        img2 = preprocess(img2).unsqueeze(0).to(device)
        state_t1 = encoder(img2)
        logits = policy(state_t1)
        probs = F.softmax(logits, dim=1)
        next_action = torch.argmax(probs, dim=1).item()
        import random
        epsilon = 0.0
        if random.random() < epsilon:
            print("🎲 Exploring...")
            next_action = random.randint(0, 3)  # 探索
        next_action = random.randint(0, 3)  # 探索
        print(f"🎮 Policy selected next action: {next_action}")
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