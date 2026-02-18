
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from curiosity_model import Encoder

encoder = Encoder()
checkpoint = torch.load('curiosity_model_best.pth', map_location='cpu')
encoder.load_state_dict(checkpoint['encoder'])
encoder.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

imgs = [
    "screenshots/1760357809361-485006_before.png",  # 第一张旋转角度
    "screenshots/1760357809361-485006_after.png"   # 第二张旋转角度
]

feats = []
for f in imgs:
    x = transform(Image.open(f).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        feat = encoder(x).numpy().flatten()
    feats.append(feat)

diff = np.linalg.norm(feats[0] - feats[1])
print("🔍 特征向量差异 L2 距离:", diff)
