"""加载训练好的 Q 网络，按最大 Q 值选合法动作。"""
import os
import torch
import numpy as np

from doudizhu.env.env import get_obs
from doudizhu.dmc.models import model_dict


def _pick_device():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if visible.strip() in ("", "-1"):
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _load_model(position, model_path, device):
    model = model_dict[position]()
    # 权重是 GPU 上存的，先映射到 CPU，再搬到目标设备，避免评测藏 GPU 时 load 失败。
    pretrained = torch.load(model_path, map_location="cpu", weights_only=False)
    # checkpoint 可能是完整 model.tar，也可能是单位置 state_dict。
    if isinstance(pretrained, dict) and "model_state_dict" in pretrained:
        pretrained = pretrained["model_state_dict"][position]
    model.load_state_dict(pretrained)
    model.to(device)
    model.eval()
    return model


class DeepAgent:
    """神经网络智能体：对每个合法动作打分，选 Q 最大的那个。"""

    def __init__(self, position, model_path):
        self.device = _pick_device()
        self.model = _load_model(position, model_path, self.device)

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        obs = get_obs(infoset)
        z_batch = torch.from_numpy(obs["z_batch"]).float().to(self.device)
        x_batch = torch.from_numpy(obs["x_batch"]).float().to(self.device)
        y_pred = self.model.forward(z_batch, x_batch, return_value=True)["values"]
        y_pred = y_pred.detach().cpu().numpy()
        best_action_index = int(np.argmax(y_pred, axis=0)[0])
        return infoset.legal_actions[best_action_index]
