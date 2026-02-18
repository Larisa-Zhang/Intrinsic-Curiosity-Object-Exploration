import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=128, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
)

    def forward(self, state):
        return self.net(state)  # 最后在外部用 softmax 得到概率分布