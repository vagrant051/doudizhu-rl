"""根据当前手牌，枚举所有可能打出的牌型组合（不管能不能压过上家）。"""
from doudizhu.env.utils import MIN_SINGLE_CARDS, MIN_PAIRS, MIN_TRIPLES, select
import collections
import itertools

class MovesGener(object):
    """
    出牌生成器：把手牌拆成单张 / 对子 / 三张 / 炸弹等，
    再组合成斗地主里合法的各种牌型。
    """
    def __init__(self, cards_list):
        self.cards_list = cards_list
        self.cards_dict = collections.defaultdict(int)

        for i in self.cards_list:
            self.cards_dict[i] += 1

        self.single_card_moves = []
        self.gen_type_1_single()
        self.pair_moves = []
        self.gen_type_2_pair()
        self.triple_cards_moves = []
        self.gen_type_3_triple()
        self.bomb_moves = []
        self.gen_type_4_bomb()
        self.final_bomb_moves = []
        self.gen_type_5_king_bomb()

    def _gen_serial_moves(self, cards, min_serial, repeat=1, repeat_num=0):
        """
        生成连续牌型。

        cards: 已经去重、可组成序列的点数
        min_serial: 最短长度（顺子 5、连对 3、飞机 2）
        repeat: 每个点数重复几次（单顺 1、连对 2、飞机 3）
        repeat_num: 若 >0，只生成固定长度；若为 0，生成所有合法长度
        """
        if repeat_num < min_serial:  # 指定长度太短则改成「不限长度」
            repeat_num = 0

        single_cards = sorted(list(set(cards)))
        seq_records = list()
        moves = list()

        start = i = 0
        longest = 1
        while i < len(single_cards):
            if i + 1 < len(single_cards) and single_cards[i + 1] - single_cards[i] == 1:
                longest += 1
                i += 1
            else:
                seq_records.append((start, longest))
                i += 1
                start = i
                longest = 1

        for seq in seq_records:
            if seq[1] < min_serial:
                continue
            start, longest = seq[0], seq[1]
            longest_list = single_cards[start: start + longest]

            if repeat_num == 0:  # 不限长度：从最短一直滑到最长
                steps = min_serial
                while steps <= longest:
                    index = 0
                    while steps + index <= longest:
                        target_moves = sorted(longest_list[index: index + steps] * repeat)
                        moves.append(target_moves)
                        index += 1
                    steps += 1

            else:  # 固定长度（跟牌时必须和上家一样长）
                if longest < repeat_num:
                    continue
                index = 0
                while index + repeat_num <= longest:
                    target_moves = sorted(longest_list[index: index + repeat_num] * repeat)
                    moves.append(target_moves)
                    index += 1

        return moves

    def gen_type_1_single(self):
        """单张。"""
        self.single_card_moves = []
        for i in set(self.cards_list):
            self.single_card_moves.append([i])
        return self.single_card_moves

    def gen_type_2_pair(self):
        """对子。"""
        self.pair_moves = []
        for k, v in self.cards_dict.items():
            if v >= 2:
                self.pair_moves.append([k, k])
        return self.pair_moves

    def gen_type_3_triple(self):
        """三张。"""
        self.triple_cards_moves = []
        for k, v in self.cards_dict.items():
            if v >= 3:
                self.triple_cards_moves.append([k, k, k])
        return self.triple_cards_moves

    def gen_type_4_bomb(self):
        """炸弹（四张相同）。"""
        self.bomb_moves = []
        for k, v in self.cards_dict.items():
            if v == 4:
                self.bomb_moves.append([k, k, k, k])
        return self.bomb_moves

    def gen_type_5_king_bomb(self):
        """王炸：小王 20 + 大王 30。"""
        self.final_bomb_moves = []
        if 20 in self.cards_list and 30 in self.cards_list:
            self.final_bomb_moves.append([20, 30])
        return self.final_bomb_moves

    def gen_type_6_3_1(self):
        """三带一：三张 + 一张不同点数的单牌。"""
        result = []
        for t in self.single_card_moves:
            for i in self.triple_cards_moves:
                if t[0] != i[0]:
                    result.append(t+i)
        return result

    def gen_type_7_3_2(self):
        """三带二：三张 + 一对。"""
        result = list()
        for t in self.pair_moves:
            for i in self.triple_cards_moves:
                if t[0] != i[0]:
                    result.append(t+i)
        return result

    def gen_type_8_serial_single(self, repeat_num=0):
        """顺子，至少 5 张连续单牌。"""
        return self._gen_serial_moves(self.cards_list, MIN_SINGLE_CARDS, repeat=1, repeat_num=repeat_num)

    def gen_type_9_serial_pair(self, repeat_num=0):
        """连对，至少 3 对连续。"""
        single_pairs = list()
        for k, v in self.cards_dict.items():
            if v >= 2:
                single_pairs.append(k)

        return self._gen_serial_moves(single_pairs, MIN_PAIRS, repeat=2, repeat_num=repeat_num)

    def gen_type_10_serial_triple(self, repeat_num=0):
        """飞机不带翅膀，至少 2 组连续三张。"""
        single_triples = list()
        for k, v in self.cards_dict.items():
            if v >= 3:
                single_triples.append(k)

        return self._gen_serial_moves(single_triples, MIN_TRIPLES, repeat=3, repeat_num=repeat_num)

    def gen_type_11_serial_3_1(self, repeat_num=0):
        """飞机带单：N 组连续三张 + N 张单牌。"""
        serial_3_moves = self.gen_type_10_serial_triple(repeat_num=repeat_num)
        serial_3_1_moves = list()

        for s3 in serial_3_moves:  # 例如 [3,3,3,4,4,4]
            s3_set = set(s3)
            new_cards = [i for i in self.cards_list if i not in s3_set]

            # 从剩下的牌里选「飞机长度」张作为翅膀
            subcards = select(new_cards, len(s3_set))

            for i in subcards:
                serial_3_1_moves.append(s3 + i)

        return list(k for k, _ in itertools.groupby(serial_3_1_moves))

    def gen_type_12_serial_3_2(self, repeat_num=0):
        """飞机带对：N 组连续三张 + N 对。"""
        serial_3_moves = self.gen_type_10_serial_triple(repeat_num=repeat_num)
        serial_3_2_moves = list()
        pair_set = sorted([k for k, v in self.cards_dict.items() if v >= 2])

        for s3 in serial_3_moves:
            s3_set = set(s3)
            pair_candidates = [i for i in pair_set if i not in s3_set]

            # 从剩余对子里选「飞机长度」对作为翅膀
            subcards = select(pair_candidates, len(s3_set))
            for i in subcards:
                serial_3_2_moves.append(sorted(s3 + i * 2))

        return serial_3_2_moves

    def gen_type_13_4_2(self):
        """四带二：四张 + 两张单牌。"""
        four_cards = list()
        for k, v in self.cards_dict.items():
            if v == 4:
                four_cards.append(k)

        result = list()
        for fc in four_cards:
            cards_list = [k for k in self.cards_list if k != fc]
            subcards = select(cards_list, 2)
            for i in subcards:
                result.append([fc]*4 + i)
        return list(k for k, _ in itertools.groupby(result))

    def gen_type_14_4_22(self):
        """四带两对。"""
        four_cards = list()
        for k, v in self.cards_dict.items():
            if v == 4:
                four_cards.append(k)

        result = list()
        for fc in four_cards:
            cards_list = [k for k, v in self.cards_dict.items() if k != fc and v>=2]
            subcards = select(cards_list, 2)
            for i in subcards:
                result.append([fc] * 4 + [i[0], i[0], i[1], i[1]])
        return result

    def gen_moves(self):
        """领出时：把手牌能组成的所有合法牌型都列出来。"""
        moves = []
        moves.extend(self.gen_type_1_single())
        moves.extend(self.gen_type_2_pair())
        moves.extend(self.gen_type_3_triple())
        moves.extend(self.gen_type_4_bomb())
        moves.extend(self.gen_type_5_king_bomb())
        moves.extend(self.gen_type_6_3_1())
        moves.extend(self.gen_type_7_3_2())
        moves.extend(self.gen_type_8_serial_single())
        moves.extend(self.gen_type_9_serial_pair())
        moves.extend(self.gen_type_10_serial_triple())
        moves.extend(self.gen_type_11_serial_3_1())
        moves.extend(self.gen_type_12_serial_3_2())
        moves.extend(self.gen_type_13_4_2())
        moves.extend(self.gen_type_14_4_22())
        return moves
