"""把 DouZero Env 包一层：自动 reset，并把观察搬到 Actor 所在设备。"""

import torch


def _format_observation(obs, device):
    """把 numpy 观察搬到 Actor 设备；x_no_action / z 先留在 CPU，写入 buffer 时再拷。"""
    position = obs["position"]
    torch_device = torch.device("cpu" if device == "cpu" else f"cuda:{device}")
    x_batch = torch.from_numpy(obs["x_batch"]).to(torch_device)
    z_batch = torch.from_numpy(obs["z_batch"]).to(torch_device)
    x_no_action = torch.from_numpy(obs["x_no_action"])
    z = torch.from_numpy(obs["z"])
    packed = {
        "x_batch": x_batch,
        "z_batch": z_batch,
        "legal_actions": obs["legal_actions"],
    }
    return position, packed, x_no_action, z


class Environment:
    """Actor 用的环境包装：观察转 tensor，终局自动开下一局。"""

    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.episode_return = None

    def initial(self):
        """开第一局。done=True 只是占位，方便和后续 step 的返回格式对齐。"""
        position, obs, x_no_action, z = _format_observation(self.env.reset(), self.device)
        self.episode_return = torch.zeros(1, 1)
        return position, obs, dict(
            done=torch.ones(1, 1, dtype=torch.bool),
            episode_return=self.episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
        )

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        self.episode_return += reward
        episode_return = self.episode_return

        if done:
            # 终局立刻开下一局，Actor 循环不用自己调 reset
            obs = self.env.reset()
            self.episode_return = torch.zeros(1, 1)

        position, obs, x_no_action, z = _format_observation(obs, self.device)
        return position, obs, dict(
            done=torch.tensor(done).view(1, 1),
            episode_return=episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
        )
