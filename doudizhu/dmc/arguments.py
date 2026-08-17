"""训练超参。Windows 上默认 CPU Actor + GPU Learner，避免 CUDA 多进程共享内存问题。"""

import argparse
import sys

parser = argparse.ArgumentParser(description="DouZero-ResNet 斗地主 DMC 训练")

parser.add_argument("--xpid", default="douzero-resnet", help="实验名，用于 checkpoint 目录")
parser.add_argument("--save_interval", default=30, type=int, help="保存间隔（分钟）")
parser.add_argument(
    "--objective",
    default="adp",
    type=str,
    choices=["adp", "wp", "logadp"],
    help="奖励：wp=胜负±1，adp=炸弹翻倍分差，logadp=炸弹数+1",
)

parser.add_argument(
    "--actor_device_cpu",
    action="store_true",
    help="Actor 用 CPU。Windows 建议打开（本仓库 train.py 会在 Windows 上自动打开）",
)
parser.add_argument(
    "--gpu_actors",
    action="store_true",
    help="强制 Actor 也走 GPU（仅 Linux 可靠；Windows 的 CUDA 多进程通常会报错）",
)
parser.add_argument("--gpu_devices", default="0", type=str, help="可见 GPU，例如 0 或 0,1")
parser.add_argument("--num_actor_devices", default=1, type=int, help="用于模拟的设备数")
parser.add_argument(
    "--num_actors",
    default=4,
    type=int,
    help="每个模拟设备上的 Actor 进程数。笔记本 4060 建议 2~6",
)
parser.add_argument(
    "--training_device",
    default="0",
    type=str,
    help="Learner 设备：GPU 编号或 cpu",
)
parser.add_argument("--load_model", action="store_true", help="从 checkpoint 继续训")
parser.add_argument("--disable_checkpoint", action="store_true", help="不保存 checkpoint")
parser.add_argument("--savedir", default="checkpoints", help="checkpoint 根目录")

parser.add_argument("--total_frames", default=100000000000, type=int, help="训练总步数（极大则相当于一直训）")
parser.add_argument("--exp_epsilon", default=0.01, type=float, help="ε-greedy 探索率")
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--unroll_length", default=100, type=int, help="每个 buffer 片段的时间步")
parser.add_argument("--num_buffers", default=50, type=int)
parser.add_argument("--num_threads", default=4, type=int, help="Learner 线程数")
parser.add_argument("--max_grad_norm", default=40.0, type=float)

parser.add_argument("--learning_rate", default=0.0001, type=float)
parser.add_argument("--alpha", default=0.99, type=float, help="RMSProp 平滑系数")
parser.add_argument("--momentum", default=0, type=float)
parser.add_argument("--epsilon", default=1e-5, type=float, help="RMSProp 数值稳定项")


def apply_device_defaults(flags):
    """Windows 上 CUDA tensor 无法在进程间 share_memory，默认改成 CPU Actor。"""
    on_windows = sys.platform.startswith("win")
    if on_windows and not flags.gpu_actors:
        flags.actor_device_cpu = True
    return flags
