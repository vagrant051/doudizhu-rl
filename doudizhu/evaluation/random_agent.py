"""随机智能体：在合法动作里均匀抽一个，常用作评测下限。"""
import random


class RandomAgent:
    def __init__(self):
        self.name = "Random"

    def act(self, infoset):
        return random.choice(infoset.legal_actions)
