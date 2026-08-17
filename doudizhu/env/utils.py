"""
出牌类型编号、顺子最短长度等规则常量。
牌用整数编码：3-14 是 3~A，17 是 2，20 小王，30 大王。
"""
import itertools

# 顺子 / 连对 / 飞机 的最短长度
MIN_SINGLE_CARDS = 5  # 顺子至少 5 张，例如 34567
MIN_PAIRS = 3         # 连对至少 3 对，例如 334455
MIN_TRIPLES = 2       # 飞机至少 2 组三张，例如 333444

# 出牌类型编号（move_detector / move_generator / move_selector 共用）
TYPE_0_PASS = 0           # 不出
TYPE_1_SINGLE = 1         # 单张
TYPE_2_PAIR = 2           # 对子
TYPE_3_TRIPLE = 3         # 三张
TYPE_4_BOMB = 4           # 炸弹（四张相同）
TYPE_5_KING_BOMB = 5      # 王炸（大小王）
TYPE_6_3_1 = 6            # 三带一
TYPE_7_3_2 = 7            # 三带二（带一对）
TYPE_8_SERIAL_SINGLE = 8  # 顺子
TYPE_9_SERIAL_PAIR = 9    # 连对
TYPE_10_SERIAL_TRIPLE = 10  # 飞机（不带翅膀）
TYPE_11_SERIAL_3_1 = 11   # 飞机带单
TYPE_12_SERIAL_3_2 = 12   # 飞机带对
TYPE_13_4_2 = 13          # 四带二（带两张单牌）
TYPE_14_4_22 = 14         # 四带两对
TYPE_15_WRONG = 15        # 非法牌型

# 叫地主阶段动作（本仓库训练从「已经确定地主」开始，这几个常量基本不用）
PASS = 0
CALL = 1
RAISE = 2

# 从 cards 里选 num 张的所有组合
def select(cards, num):
    return [list(i) for i in itertools.combinations(cards, num)]
