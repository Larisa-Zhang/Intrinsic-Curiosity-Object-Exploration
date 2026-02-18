import torch
from PIL import Image
from torchvision import transforms
from models import EncoderPolicy, ActorCritic  # 如果你的模型文件叫 models.py
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 载入模型
encoder_policy = EncoderPolicy().to(device)
encoder_policy.load_state_dict(
    torch.load("saved_models/encoder_policy.pth", map_location=device, weights_only=True)
)
encoder_policy.eval()

# 预处理（要和你训练用的完全一样）
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 你要测试的两张图片
imgA_path = "screenshots/1763990034914-41049_before.png"   # ← 换成你的路径
imgB_path = "screenshots/1763990034914-41049_after.png"   # ← 换成你的路径

if not os.path.exists(imgA_path):
    print("❌ Image A 不存在:", imgA_path)
if not os.path.exists(imgB_path):
    print("❌ Image B 不存在:", imgB_path)

# 加载图片
imgA = preprocess(Image.open(imgA_path).convert("RGB")).unsqueeze(0).to(device)
imgB = preprocess(Image.open(imgB_path).convert("RGB")).unsqueeze(0).to(device)

# 得到 latent φ(s)
with torch.no_grad():
    phiA = encoder_policy(imgA)   # [1, 128]
    phiB = encoder_policy(imgB)   # [1, 128]

# 计算 L2 距离
dist = torch.norm(phiA - phiB).item()

print("\n==========================")
print("Image A:", imgA_path)
print("Image B:", imgB_path)
print(f"latent distance = {dist}")
print("==========================\n")
