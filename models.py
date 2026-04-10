#SMALL-ALEXNET!!!!!!!!!!
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================
# 1. EncoderICM —— 好奇心模块的 Encoder
#    把图片变成 latent 向量，供 Forward/Inverse 使用
# =========================================
class EncoderICM(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        # 128x128
        # conv5 s2 p2 -> 64x64
        # pool3 s2 p1 -> 32x32
        # conv3 s1 p1 -> 32x32
        # pool3 s2 p1 -> 16x16
        # conv3 s1 p1 -> 16x16
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.fc = nn.Linear(256 * 16 * 16, latent_dim)
        self.bn_latent = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        z = self.conv(x)
        z = z.reshape(z.size(0), -1)
        z = self.fc(z)
        z = self.bn_latent(z)
        return z



# =========================================
# 2. Forward Model —— ICM 的“未来预测器”
#    输入（input）：latent φ(s_t) + one-hot(action)
#    输出 （output）：预测 φ(s_{t+1})
# =========================================
class ForwardModel(nn.Module):
    def __init__(self, action_dim=4, latent_dim=128):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, phi_t, a_onehot):
        x = torch.cat([phi_t, a_onehot], dim=1)
        return self.fc(x)  # 预测 φ(s_{t+1})


# =========================================
# 3. Inverse Model —— ICM 的“动作反推器”
#    输入：φ(s_t), φ(s_{t+1})
#    输出：动作 logits（分类）
# =========================================
class InverseModel(nn.Module):
    def __init__(self, action_dim=4, latent_dim=128):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, phi_t, phi_next):
        x = torch.cat([phi_t, phi_next], dim=1)
        return self.fc(x)  # logits


# =========================================
# 4. EncoderPolicy —— 给 Actor–Critic 使用的 Encoder
#    让策略网络看到“更抽象、更整洁”的 latent
#    （通常 Policy 的 encoder 和 ICM 的 encoder 不共享）
# =========================================

class EncoderPolicy(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        # Same pattern but slightly lighter
        self.conv = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.fc = nn.Linear(128 * 16 * 16, latent_dim)
        self.bn_latent = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        z = self.conv(x)
        z = z.reshape(z.size(0), -1)
        z = self.fc(z)
        z = self.bn_latent(z)
        return z




# =========================================
# 5. ActorCritic —— 策略网络 + 价值网络
#    Actor：输出动作 logits
#    Critic：输出状态价值 V(s)
# =========================================
class ActorCritic(nn.Module):
    def __init__(self, action_dim=4, latent_dim=128):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU()
        )

        # Actor（选择动作）
        self.actor = nn.Linear(256, action_dim)

        # Critic（估计未来价值）
        self.critic = nn.Linear(256, 1)

    def forward(self, latent):
        x = self.fc(latent)
        logits = self.actor(x)   # 动作 logits
        value = self.critic(x)   # V(s)
        return logits, value
