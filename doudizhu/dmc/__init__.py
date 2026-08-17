"""Deep Monte Carlo 训练：Actor 采样 + Learner 更新三个座位的 Q 网络。"""
from .arguments import apply_device_defaults, parser
from .dmc import train
