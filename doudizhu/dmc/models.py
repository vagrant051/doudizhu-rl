"""
DouZero-ResNet 的 Q 网络。

原版 DouZero 用 LSTM 编码历史出牌序列 z，再用多层 MLP 估计 Q(s, a)。
这里保持完全相同的输入输出接口，只把 LSTM 换成 1D ResNet：

    z: (B, 5, 162)  最近 15 手牌，每 3 手拼成一行（见 env.env._action_seq_list2array）
    x: (B, 373) 或 (B, 484)  当前局面特征 + 候选动作的 4x15 牌矩阵编码
    输出: (B, 1)  每个候选动作的 Q 值

推理时 B = 当前合法动作数；训练时 B = unroll_length * batch_size。
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# 地主 / 农民的静态特征维度（含 54 维动作编码），与 DouZero 论文 Table 4/5 一致。
LANDLORD_X_DIM = 373
FARMER_X_DIM = 484
HISTORY_CHANNELS = 5
HISTORY_LENGTH = 162


class ResidualBlock1D(nn.Module):
    """标准 1D 残差块：Conv-BN-ReLU-Conv-BN，再加 shortcut。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class HistoryResNet(nn.Module):
    """把历史出牌 z 编成固定长度向量，替代 DouZero 的 LSTM。"""

    def __init__(self, out_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(HISTORY_CHANNELS, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock1D(64, 64, stride=1)
        self.layer2 = ResidualBlock1D(64, 128, stride=2)
        self.layer3 = ResidualBlock1D(128, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, out_dim)

    def forward(self, z):
        # z: (B, 5, 162) —— 已经是 Conv1d 需要的 (N, C, L)
        h = self.stem(z)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.pool(h).flatten(1)
        return self.fc(h)


class PositionQNet(nn.Module):
    """单个位置（地主 / 农民）的 Q 网络。"""

    def __init__(self, x_dim):
        super().__init__()
        self.history = HistoryResNet(out_dim=128)
        self.fc1 = nn.Linear(x_dim + 128, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        hist = self.history(z)
        out = torch.cat([hist, x], dim=-1)
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = F.relu(self.fc3(out), inplace=True)
        out = F.relu(self.fc4(out), inplace=True)
        out = F.relu(self.fc5(out), inplace=True)
        values = self.fc6(out)

        if return_value:
            return {"values": values}

        # 推理：在当前所有合法动作的 Q 值里选一个（可 ε-greedy）。
        if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
            action = torch.randint(values.shape[0], (1,))[0]
        else:
            action = torch.argmax(values, dim=0)[0]
        return {"action": action}


def _to_torch_device(device):
    if device == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{device}")


# 评测时按位置名创建网络。
model_dict = {
    "landlord": lambda: PositionQNet(LANDLORD_X_DIM),
    "landlord_up": lambda: PositionQNet(FARMER_X_DIM),
    "landlord_down": lambda: PositionQNet(FARMER_X_DIM),
}


class Model:
    """三个位置各一份网络。Actor 和 Learner 都通过这个包装类访问。"""

    def __init__(self, device=0):
        torch_device = _to_torch_device(device)
        self.models = {
            "landlord": PositionQNet(LANDLORD_X_DIM).to(torch_device),
            "landlord_up": PositionQNet(FARMER_X_DIM).to(torch_device),
            "landlord_down": PositionQNet(FARMER_X_DIM).to(torch_device),
        }

    def forward(self, position, z, x, training=False, flags=None):
        return self.models[position].forward(z, x, training, flags)

    def share_memory(self):
        for model in self.models.values():
            model.share_memory()

    def eval(self):
        for model in self.models.values():
            model.eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models
