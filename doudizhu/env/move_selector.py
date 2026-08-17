"""
跟牌过滤器：从「同类型」的候选里，只留下能压过上家的。
moves 和 rival_move 必须是同一牌型。
"""
import collections

def common_handle(moves, rival_move):
    """最简单的比大小：排序后第一张更大即可（单张 / 对子 / 三张 / 顺子 / 炸弹等）。"""
    new_moves = list()
    for move in moves:
        if move[0] > rival_move[0]:
            new_moves.append(move)
    return new_moves

def filter_type_1_single(moves, rival_move):
    """压单张。"""
    return common_handle(moves, rival_move)


def filter_type_2_pair(moves, rival_move):
    """压对子。"""
    return common_handle(moves, rival_move)


def filter_type_3_triple(moves, rival_move):
    """压三张。"""
    return common_handle(moves, rival_move)


def filter_type_4_bomb(moves, rival_move):
    """压炸弹（更大的四张，或王炸走另一条路径）。"""
    return common_handle(moves, rival_move)

# 王炸最大，不需要过滤器

def filter_type_6_3_1(moves, rival_move):
    """三带一：比那三张的点数，带的单牌不参与比较。"""
    rival_move.sort()
    rival_rank = rival_move[1]
    new_moves = list()
    for move in moves:
        move.sort()
        my_rank = move[1]
        if my_rank > rival_rank:
            new_moves.append(move)
    return new_moves

def filter_type_7_3_2(moves, rival_move):
    """三带二：同样只比三张的点数。"""
    rival_move.sort()
    rival_rank = rival_move[2]
    new_moves = list()
    for move in moves:
        move.sort()
        my_rank = move[2]
        if my_rank > rival_rank:
            new_moves.append(move)
    return new_moves

def filter_type_8_serial_single(moves, rival_move):
    """压顺子（长度已在生成时对齐）。"""
    return common_handle(moves, rival_move)

def filter_type_9_serial_pair(moves, rival_move):
    """压连对。"""
    return common_handle(moves, rival_move)

def filter_type_10_serial_triple(moves, rival_move):
    """压飞机（不带翅膀）。"""
    return common_handle(moves, rival_move)

def filter_type_11_serial_3_1(moves, rival_move):
    """飞机带单：比连续三张里最大的那组。"""
    rival = collections.Counter(rival_move)
    rival_rank = max([k for k, v in rival.items() if v == 3])
    new_moves = list()
    for move in moves:
        mymove = collections.Counter(move)
        my_rank = max([k for k, v in mymove.items() if v == 3])
        if my_rank > rival_rank:
            new_moves.append(move)
    return new_moves

def filter_type_12_serial_3_2(moves, rival_move):
    """飞机带对：同样比三张部分。"""
    rival = collections.Counter(rival_move)
    rival_rank = max([k for k, v in rival.items() if v == 3])
    new_moves = list()
    for move in moves:
        mymove = collections.Counter(move)
        my_rank = max([k for k, v in mymove.items() if v == 3])
        if my_rank > rival_rank:
            new_moves.append(move)
    return new_moves

def filter_type_13_4_2(moves, rival_move):
    """四带二：比那四张的点数。"""
    rival_move.sort()
    rival_rank = rival_move[2]
    new_moves = list()
    for move in moves:
        move.sort()
        my_rank = move[2]
        if my_rank > rival_rank:
            new_moves.append(move)
    return new_moves

def filter_type_14_4_22(moves, rival_move):
    """四带两对：比四张的点数。"""
    rival = collections.Counter(rival_move)
    rival_rank = my_rank = 0
    for k, v in rival.items():
        if v == 4:
            rival_rank = k
    new_moves = list()
    for move in moves:
        mymove = collections.Counter(move)
        for k, v in mymove.items():
            if v == 4:
                my_rank = k
        if my_rank > rival_rank:
            new_moves.append(move)
    return new_moves
