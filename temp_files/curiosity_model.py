import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),  # [B, 16, 64, 64]
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), # [B, 32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # [B, 64, 16, 16]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, output_dim)
        )

    def forward(self, x):
        return self.conv(x)

class ForwardModel(nn.Module):
    def __init__(self, state_dim=128, action_dim=4):  # 4 个动作
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )

    def forward(self, state, action_onehot):
        x = torch.cat([state, action_onehot], dim=1)
        return self.fc(x)
    
    
class InverseModel(nn.Module):
    """
    逆模型：给定 (s_t, s_{t+1}) 预测动作 a_t
    """
    def __init__(self, state_dim=128, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)  # 输出 logits，用 CrossEntropyLoss
        )

    def forward(self, state, next_state):
        x = torch.cat([state, next_state], dim=1)  # [B, 2*state_dim]
        return self.net(x)