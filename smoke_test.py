"""冒烟测试：环境 reset/step + ResNet 前向，不跑完整训练。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from doudizhu.dmc.models import Model
from doudizhu.env import Env


def main():
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, cuda={torch.cuda.is_available()}")

    env = Env("adp")
    obs = env.reset()
    print("position:", obs["position"])
    print("z_batch:", tuple(obs["z_batch"].shape), "(合法动作数, 5, 162)")
    print("x_batch:", tuple(obs["x_batch"].shape), "(合法动作数, 373 或 484)")
    print("legal_actions:", len(obs["legal_actions"]))

    model = Model(device=device)
    model.eval()
    torch_device = torch.device("cpu" if device == "cpu" else f"cuda:{device}")
    z = torch.from_numpy(obs["z_batch"]).float().to(torch_device)
    x = torch.from_numpy(obs["x_batch"]).float().to(torch_device)
    with torch.no_grad():
        values = model.forward(obs["position"], z, x, training=True)["values"]
        action = model.forward(obs["position"], z, x)["action"]
    print("Q values:", tuple(values.shape), "picked action idx:", int(action))

    next_obs, reward, done, _ = env.step(obs["legal_actions"][int(action)])
    print("step ok, done=", done, "reward=", reward, "next_pos=", None if next_obs is None else next_obs["position"])
    print("smoke test passed")


if __name__ == "__main__":
    main()
