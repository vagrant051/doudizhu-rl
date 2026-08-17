"""DMC 的 Actor 侧：共享内存 buffer、环境采样、把 (s,a,G) 送给 Learner。"""

import logging
import traceback
import typing

import numpy as np
import torch

from .env_utils import Environment
from doudizhu.env import Env
from doudizhu.env.env import _cards2array

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter("[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] %(message)s")
)
log = logging.getLogger("doudizhu")
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def create_env(flags):
    """按 objective（wp / adp / logadp）建一个斗地主环境。"""
    return Env(flags.objective)


def get_batch(free_queue, full_queue, buffers, flags, lock):
    """从 full_queue 取出 batch_size 个 buffer 下标，拼成一个 batch，再把下标还回 free_queue。"""
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers}
    for m in indices:
        free_queue.put(m)
    return batch


def create_optimizers(flags, learner_model):
    """三个座位各一个 RMSProp，互不共享参数。"""
    positions = ["landlord", "landlord_up", "landlord_down"]
    optimizers = {}
    for position in positions:
        optimizers[position] = torch.optim.RMSprop(
            learner_model.parameters(position),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha,
        )
    return optimizers


def create_buffers(flags, device_iterator):
    """
    每个 Actor 设备、每个位置各一组共享内存 buffer。

    x 拆成 obs_x_no_action + obs_action，是为了在 Learner 里再拼起来，
    这样动作编码可以单独存在 54 维牌矩阵里，和 DouZero 一致。
    """
    T = flags.unroll_length
    positions = ["landlord", "landlord_up", "landlord_down"]
    buffers = {}
    for device in device_iterator:
        buffers[device] = {}
        for position in positions:
            x_dim = 319 if position == "landlord" else 430
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                obs_x_no_action=dict(size=(T, x_dim), dtype=torch.int8),
                obs_action=dict(size=(T, 54), dtype=torch.int8),
                obs_z=dict(size=(T, 5, 162), dtype=torch.int8),
            )
            _buffers: Buffers = {key: [] for key in specs}
            torch_device = torch.device("cpu" if device == "cpu" else f"cuda:{device}")
            for _ in range(flags.num_buffers):
                for key in _buffers:
                    _buffers[key].append(torch.empty(**specs[key]).to(torch_device).share_memory_())
            buffers[device][position] = _buffers
    return buffers


def act(i, device, free_queue, full_queue, model, buffers, flags):
    """
    Actor 主循环：self-play 一局，把每一步的 (z, x, a) 存下来，
    终局时用整局回报 G 作为 Monte-Carlo target（这就是 DMC 的核心）。
    """
    positions = ["landlord", "landlord_up", "landlord_down"]
    try:
        T = flags.unroll_length
        log.info("设备 %s 上的 Actor %i 已启动。", str(device), i)

        env = Environment(create_env(flags), device)

        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        obs_x_no_action_buf = {p: [] for p in positions}
        obs_action_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}

        position, obs, env_output = env.initial()

        while True:
            while True:
                obs_x_no_action_buf[position].append(env_output["obs_x_no_action"])
                obs_z_buf[position].append(env_output["obs_z"])
                with torch.no_grad():
                    agent_output = model.forward(position, obs["z_batch"], obs["x_batch"], flags=flags)
                action_idx = int(agent_output["action"].cpu().detach().numpy())
                action = obs["legal_actions"][action_idx]
                obs_action_buf[position].append(_cards2tensor(action))
                size[position] += 1
                position, obs, env_output = env.step(action)
                if env_output["done"]:
                    for p in positions:
                        diff = size[p] - len(target_buf[p])
                        if diff > 0:
                            done_buf[p].extend([False for _ in range(diff - 1)])
                            done_buf[p].append(True)
                            # 环境奖励是从地主视角给的，农民取负。
                            episode_return = (
                                env_output["episode_return"]
                                if p == "landlord"
                                else -env_output["episode_return"]
                            )
                            episode_return_buf[p].extend([0.0 for _ in range(diff - 1)])
                            episode_return_buf[p].append(episode_return)
                            target_buf[p].extend([episode_return for _ in range(diff)])
                    break

            for p in positions:
                while size[p] > T:
                    index = free_queue[p].get()
                    if index is None:
                        break
                    for t in range(T):
                        buffers[p]["done"][index][t, ...] = done_buf[p][t]
                        buffers[p]["episode_return"][index][t, ...] = episode_return_buf[p][t]
                        buffers[p]["target"][index][t, ...] = target_buf[p][t]
                        buffers[p]["obs_x_no_action"][index][t, ...] = obs_x_no_action_buf[p][t]
                        buffers[p]["obs_action"][index][t, ...] = obs_action_buf[p][t]
                        buffers[p]["obs_z"][index][t, ...] = obs_z_buf[p][t]
                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
                    obs_action_buf[p] = obs_action_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    size[p] -= T

    except KeyboardInterrupt:
        pass
    except Exception:
        log.error("Actor 进程 %i 出错", i)
        traceback.print_exc()
        raise


def _cards2tensor(list_cards):
    """一手牌 -> 54 维 tensor，写入 buffer 的 obs_action。"""
    return torch.from_numpy(_cards2array(list_cards))
