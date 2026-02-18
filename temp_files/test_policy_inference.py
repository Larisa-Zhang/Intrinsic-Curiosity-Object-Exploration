import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models import EncoderPolicy, ActorCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
encoder_policy = EncoderPolicy().to(device)
actor_model = ActorCritic(action_dim=4).to(device)

encoder_policy.load_state_dict(torch.load("saved_models/encoder_policy.pth", map_location=device, weights_only=True))
actor_model.load_state_dict(torch.load("saved_models/actor_critic.pth", map_location=device, weights_only=True))

encoder_policy.eval()
actor_model.eval()

# 预处理
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# ======= 测试：给同一张图 连续推理 20 次 =======
img_path = r"screenshots/1763987398906-777465_after.png"   # 你换成真实的图
img = Image.open(img_path).convert("RGB")


def run_single_inference():
    with torch.no_grad():
        x = preprocess(img).unsqueeze(0).to(device)

        # encode
        phi = encoder_policy(x)

        # actor model → logits & value
        logits, value = actor_model(phi)

        # softmax 概率
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # multinomial 推理（应用阶段可切换成 argmax）
        action_mult = torch.multinomial(torch.tensor(probs), 1).item()

        # argmax 推理
        action_argmax = int(probs.argmax())

    return probs, action_mult, action_argmax


print("\n==============================================")
print(f"Testing inference repeatability for image: {img_path}")
print("==============================================\n")

action_list_mult = []
action_list_argmax = []

for i in range(20):
    probs, action_mult, action_argmax = run_single_inference()

    action_list_mult.append(action_mult)
    action_list_argmax.append(action_argmax)

    print(f"[{i}]")
    print(f"  probs     = {probs}")
    print(f"  multinomial → {action_mult}")
    print(f"  argmax     → {action_argmax}")
    print("---------------------------------------")


print("\n========== Summary ==========")
print(f"Multinomial actions: {action_list_mult}")
print(f"Argmax actions     : {action_list_argmax}")

