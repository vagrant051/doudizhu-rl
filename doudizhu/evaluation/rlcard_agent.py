"""
RLCard 规则 Bot 的适配器。

RLCard 用字符表示牌（T=10, B=小王, R=大王），本仓库用整数。
这里做两边编码转换，并沿用 RLCard 的启发式：
  - 领出：尽量打出手牌里含最小牌的组合
  - 跟牌：出同类型里最小能压过的；农民对农民可以让牌（pass）
出错时退回随机出牌。
"""
import random

from rlcard.games.doudizhu.utils import CARD_TYPE

EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                    8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q',
                    13: 'K', 14: 'A', 17: '2', 20: 'B', 30: 'R'}
RealCard2EnvCard = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                    '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12,
                    'K': 13, 'A': 14, '2': 17, 'B': 20, 'R': 30}

INDEX = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4,
         '8': 5, '9': 6, 'T': 7, 'J': 8, 'Q': 9,
         'K': 10, 'A': 11, '2': 12, 'B': 13, 'R': 14}

class RLCardAgent(object):

    def __init__(self, position):
        self.name = 'RLCard'
        self.position = position

    def act(self, infoset):
        try:
            # 手牌：整数 -> RLCard 字符串
            hand_cards = infoset.player_hand_cards
            for i, c in enumerate(hand_cards):
                hand_cards[i] = EnvCard2RealCard[c]
            hand_cards = ''.join(hand_cards)

            # 上一手
            last_move = infoset.last_move.copy()
            for i, c in enumerate(last_move):
                last_move[i] = EnvCard2RealCard[c]
            last_move = ''.join(last_move)

            # 最近两手
            last_two_cards = infoset.last_two_moves
            for i in range(2):
                for j, c in enumerate(last_two_cards[i]):
                    last_two_cards[i][j] = EnvCard2RealCard[c]
                last_two_cards[i] = ''.join(last_two_cards[i])

            last_pid = infoset.last_pid

            action = None
            # 两家都 pass：轮到自己领出，打出含最小牌的组合
            if last_two_cards[0] == '' and last_two_cards[1] == '':
                chosen_action = None
                comb = combine_cards(hand_cards)
                min_card = hand_cards[0]
                for _, acs in comb.items():
                    for ac in acs:
                        if min_card in ac:
                            chosen_action = ac
                            action = [char for char in chosen_action]
                            for i, c in enumerate(action):
                                action[i] = RealCard2EnvCard[c]
            # 跟牌：同类型里选最小能压过的
            else:
                the_type = CARD_TYPE[0][last_move][0][0]
                chosen_action = ''
                rank = 1000
                for ac in infoset.legal_actions:
                    _ac = ac.copy()
                    for i, c in enumerate(_ac):
                        _ac[i] = EnvCard2RealCard[c]
                    _ac = ''.join(_ac)
                    if _ac != '' and the_type == CARD_TYPE[0][_ac][0][0]:
                        if int(CARD_TYPE[0][_ac][0][1]) < rank:
                            rank = int(CARD_TYPE[0][_ac][0][1])
                            chosen_action = _ac
                if chosen_action != '':
                    action = [char for char in chosen_action]
                    for i, c in enumerate(action):
                        action[i] = RealCard2EnvCard[c]
                elif last_pid != 'landlord' and self.position != 'landlord':
                    # 农民对农民：队友出的牌可以让过
                    action = []

            if action is None:
                action = random.choice(infoset.legal_actions)
        except:
            action = random.choice(infoset.legal_actions)

        assert action in infoset.legal_actions

        return action
        
def card_str2list(hand):
    """例如 '3345' -> 各点数张数的长度为 15 的列表。"""
    hand_list = [0 for _ in range(15)]
    for card in hand:
        hand_list[INDEX[card]] += 1
    return hand_list

def list2card_str(hand_list):
    """上面的逆变换。"""
    card_str = ''
    cards = [card for card in INDEX]
    for index, count in enumerate(hand_list):
        card_str += cards[index] * count
    return card_str

def pick_chain(hand_list, count):
    """从手牌计数里抽出顺子（count=1）或连对（count=2）。"""
    chains = []
    str_card = [card for card in INDEX]
    hand_list = [str(card) for card in hand_list]
    hand = ''.join(hand_list[:12])
    chain_list = hand.split('0')
    add = 0
    for index, chain in enumerate(chain_list):
        if len(chain) > 0:
            if len(chain) >= 5:
                start = index + add
                min_count = int(min(chain)) // count
                if min_count != 0:
                    str_chain = ''
                    for num in range(len(chain)):
                        str_chain += str_card[start+num]
                        hand_list[start+num] = int(hand_list[start+num]) - int(min(chain))
                    for _ in range(min_count):
                        chains.append(str_chain)
            add += len(chain)
    hand_list = [int(card) for card in hand_list]
    return (chains, hand_list)

def combine_cards(hand):
    """
    把一手牌贪心拆成：王炸、炸弹、三张、飞机、顺子、连对、对子、单张。
    领出时用这个结果挑「含最小牌」的组合先打出去。
    """
    comb = {'rocket': [], 'bomb': [], 'trio': [], 'trio_chain': [],
            'solo_chain': [], 'pair_chain': [], 'pair': [], 'solo': []}
    # 1. 王炸
    if hand[-2:] == 'BR':
        comb['rocket'].append('BR')
        hand = hand[:-2]
    # 2. 炸弹
    hand_cp = hand
    for index in range(len(hand_cp) - 3):
        if hand_cp[index] == hand_cp[index+3]:
            bomb = hand_cp[index: index+4]
            comb['bomb'].append(bomb)
            hand = hand.replace(bomb, '')
    # 3. 三张和飞机
    hand_cp = hand
    for index in range(len(hand_cp) - 2):
        if hand_cp[index] == hand_cp[index+2]:
            trio = hand_cp[index: index+3]
            if len(comb['trio']) > 0 and INDEX[trio[-1]] < 12 and (INDEX[trio[-1]]-1) == INDEX[comb['trio'][-1][-1]]:
                comb['trio'][-1] += trio
            else:
                comb['trio'].append(trio)
            hand = hand.replace(trio, '')
    only_trio = []
    only_trio_chain = []
    for trio in comb['trio']:
        if len(trio) == 3:
            only_trio.append(trio)
        else:
            only_trio_chain.append(trio)
    comb['trio'] = only_trio
    comb['trio_chain'] = only_trio_chain
    # 4. 顺子
    hand_list = card_str2list(hand)
    chains, hand_list = pick_chain(hand_list, 1)
    comb['solo_chain'] = chains
    # 5. 连对
    chains, hand_list = pick_chain(hand_list, 2)
    comb['pair_chain'] = chains
    hand = list2card_str(hand_list)
    # 6. 对子和单张
    index = 0
    while index < len(hand) - 1:
        if hand[index] == hand[index+1]:
            comb['pair'].append(hand[index] + hand[index+1])
            index += 2
        else:
            comb['solo'].append(hand[index])
            index += 1
    if index == (len(hand) - 1):
        comb['solo'].append(hand[index])
    return comb
