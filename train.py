"""
入口脚本。Windows 默认 CPU Actor + GPU Learner。
笔记本 4060（8GB）建议：--num_actors 4 --batch_size 32
"""

import os
import sys

# 保证从项目根目录运行、以及 Windows spawn 子进程里都能 import doudizhu
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

from doudizhu.dmc import apply_device_defaults, parser, train


if __name__ == "__main__":
    flags = parser.parse_args()
    flags = apply_device_defaults(flags)
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices
    train(flags)
