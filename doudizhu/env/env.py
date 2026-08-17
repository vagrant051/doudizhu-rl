"""
斗地主环境封装（特征编码 + Gym 风格接口）。

来源：DouZero (ICML 2021, Apache-2.0)
https://github.com/kwai/DouZero

这里只复用「规则 + 出牌编码」，不复用原论文的 LSTM 网络。
观察特征 z / x 的含义见 get_obs() 的注释，以及论文 Table 4/5。
"""
from collections import Counter
import numpy as np

from doudizhu.env.game import GameEnv

# 牌面点数 -> 4x13 矩阵的列号。
# 3,4,5,6,7,8,9,10,J(11),Q(12),K(13),A(14),2(17)
# 大小王不进这个矩阵，单独用 2 维表示。
Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

# 某个点数有几张牌，用长度为 4 的 0/1 向量表示（一副牌最多 4 张同点数）。
# 例如 3 张 -> [1, 1, 1, 0]
NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

# 一副牌 54 张：3~A 各 4 张，2 有 4 张（编码 17），小王 20，大王 30。
deck = []
for i in range(3, 15):
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])
deck.extend([20, 30])

class Env:
    """
    三人斗地主的多智能体封装，接口接近 Gym：reset() / step(action)。

    objective 决定终局奖励怎么算：
      - wp：只看输赢，赢 +1 / 输 -1
      - adp：炸弹翻倍，赢 +2^炸弹数 / 输 -2^炸弹数
      - logadp：赢 +(炸弹数+1) / 输 -(炸弹数+1)

    内部用三个 DummyAgent 占位。真正的策略在环境外面：
    你调用 step(action) 时，环境先把 action 塞给当前座位的 DummyAgent，
    再让规则引擎走一步。这样可以把「玩家」和「环境」分开。
    """
    def __init__(self, objective):
        self.objective = objective

        # 三个座位：landlord 地主，landlord_up 地主上家，landlord_down 地主下家
        self.players = {}
        for position in ['landlord', 'landlord_up', 'landlord_down']:
            self.players[position] = DummyAgent(position)

        # 内部规则引擎（发牌、出牌合法性、胜负）
        self._env = GameEnv(self.players)

        self.infoset = None

    def reset(self):
        """
        重新洗牌开一局。一局结束后通常会再调一次。

        发牌：地主 20 张（含 3 张底牌），两个农民各 17 张。
        three_landlord_cards 是那 3 张底牌（从地主手牌里取 [17:20]）。
        """
        self._env.reset()

        # 随机洗牌后按座位切分
        _deck = deck.copy()
        np.random.shuffle(_deck)
        card_play_data = {'landlord': _deck[:20],
                          'landlord_up': _deck[20:37],
                          'landlord_down': _deck[37:54],
                          'three_landlord_cards': _deck[17:20],
                          }
        for key in card_play_data:
            card_play_data[key].sort()

        self._env.card_play_init(card_play_data)
        self.infoset = self._game_infoset

        return get_obs(self.infoset)

    def step(self, action):
        """
        执行一步出牌。

        参数:
            action: 整数列表，表示打出的牌（空列表 [] 表示 pass / 不出）。
                    必须在当前 infoset.legal_actions 里。

        返回:
            obs: 下一个要出牌的人看到的观察；终局时为 None
            reward: 终局才给非零奖励，过程中为 0（奖励始终从地主视角给出）
            done: 本局是否结束
            info: 预留空字典
        """
        assert action in self.infoset.legal_actions
        self.players[self._acting_player_position].set_action(action)
        self._env.step()
        self.infoset = self._game_infoset
        done = False
        reward = 0.0
        if self._game_over:
            done = True
            reward = self._get_reward()
            obs = None
        else:
            obs = get_obs(self.infoset)
        return obs, reward, done, {}

    def _get_reward(self):
        """
        一局结束时调用。地主赢为正，农民赢为负（始终从地主视角）。
        炸弹会影响 adp / logadp 的分值。
        """
        winner = self._game_winner
        bomb_num = self._game_bomb_num
        if winner == 'landlord':
            if self.objective == 'adp':
                return 2.0 ** bomb_num
            elif self.objective == 'logadp':
                return bomb_num + 1.0
            else:
                return 1.0
        else:
            if self.objective == 'adp':
                return -2.0 ** bomb_num
            elif self.objective == 'logadp':
                return -bomb_num - 1.0
            else:
                return -1.0

    @property
    def _game_infoset(self):
        """
        infoset（信息集）= 当前局面的全部信息：三家手牌、历史出牌等。
        这是「上帝视角」的完美信息；真正喂给网络的是 get_obs() 抽出来的
        不完美信息（看不到对手具体手牌，只看剩余牌的并集）。
        """
        return self._env.game_infoset

    @property
    def _game_bomb_num(self):
        """目前已经打出的炸弹数（含王炸）。既当网络特征，也用来算 ADP。"""
        return self._env.get_bomb_num()

    @property
    def _game_winner(self):
        """胜者字符串：'landlord' 或 'farmer'。"""
        return self._env.get_winner()

    @property
    def _acting_player_position(self):
        """当前该谁出牌：landlord / landlord_down / landlord_up。"""
        return self._env.acting_player_position

    @property
    def _game_over(self):
        """本局是否已经结束。"""
        return self._env.game_over

class DummyAgent(object):
    """
    占位智能体：环境先 set_action() 告诉它要出什么，
    规则引擎再调用 act() 真正执行。用来把环境和策略解耦。
    """
    def __init__(self, position):
        self.position = position
        self.action = None

    def act(self, infoset):
        """直接返回环境事先设好的动作。"""
        assert self.action in infoset.legal_actions
        return self.action

    def set_action(self, action):
        """环境用这个函数把动作塞给当前座位。"""
        self.action = action

def get_obs(infoset):
    """
    从不完美信息视角构造观察。三个座位的特征维度不同，所以分三支。

    返回字典 obs，训练时会用到这些字段：

    position: 'landlord' / 'landlord_down' / 'landlord_up'

    x_batch: 静态局面特征 + 候选动作编码，batch 维 = 合法动作数。
             地主 373 维，农民 484 维（含 54 维动作）。
    z_batch: 历史出牌特征，形状 (合法动作数, 5, 162)。
             最近 15 手，每 3 手拼成一行。

    legal_actions: 当前所有合法出牌（含 [] 表示不出）

    x_no_action: 不含「候选动作」的静态特征，没有 batch 维。
                 地主 319 维，农民 430 维。
    z: 和 z_batch 一样，但没有 batch 维，形状 (5, 162)
    """
    if infoset.player_position == 'landlord':
        return _get_obs_landlord(infoset)
    elif infoset.player_position == 'landlord_up':
        return _get_obs_landlord_up(infoset)
    elif infoset.player_position == 'landlord_down':
        return _get_obs_landlord_down(infoset)
    else:
        raise ValueError('')

def _get_one_hot_array(num_left_cards, max_num_cards):
    """把「还剩几张牌」编成 one-hot。例如剩 3 张、最多 17 -> 第 2 位为 1。"""
    one_hot = np.zeros(max_num_cards)
    one_hot[num_left_cards - 1] = 1

    return one_hot

def _cards2array(list_cards):
    """
    把一手牌（整数列表）编成 54 维向量：
      13 个点数 × 4 张（按列 flatten，Fortran 顺序）+ 小王 + 大王。
    空牌（pass）则全 0。
    """
    if len(list_cards) == 0:
        return np.zeros(54, dtype=np.int8)

    matrix = np.zeros([4, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(list_cards)
    for card, num_times in counter.items():
        if card < 20:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
        elif card == 20:
            jokers[0] = 1
        elif card == 30:
            jokers[1] = 1
    return np.concatenate((matrix.flatten('F'), jokers))

def _action_seq_list2array(action_seq_list):
    """
    编码历史出牌：固定看最近 15 手（不够则前面补空）。
    斗地主三人一轮，所以每连续 3 手拼成一行 162 维（3×54）。
    最终得到 5×162 矩阵，原来喂给 LSTM，本仓库改成喂给 1D ResNet。
    """
    action_seq_array = np.zeros((len(action_seq_list), 54))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards)
    action_seq_array = action_seq_array.reshape(5, 162)
    return action_seq_array

def _process_action_seq(sequence, length=15):
    """截取最近 15 手历史；不足 15 手时在前面用空列表补齐。"""
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence

def _get_one_hot_bomb(bomb_num):
    """炸弹数的 one-hot，长度 15（一局炸弹不会太多）。"""
    one_hot = np.zeros(15)
    one_hot[bomb_num] = 1
    return one_hot

def _get_obs_landlord(infoset):
    """
    地主视角特征（DouZero 论文 Table 4）。
    各块维度：手牌54 + 别人手牌并集54 + 上一手54
              + 上家已出54 + 下家已出54
              + 上家剩牌one-hot17 + 下家剩牌one-hot17
              + 炸弹数15 + 候选动作54
              = 373（不含动作则为 319）
    """
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    landlord_up_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)
    landlord_up_num_cards_left_batch = np.repeat(
        landlord_up_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)
    landlord_down_num_cards_left_batch = np.repeat(
        landlord_down_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_up_played_cards = _cards2array(
        infoset.played_cards['landlord_up'])
    landlord_up_played_cards_batch = np.repeat(
        landlord_up_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_played_cards = _cards2array(
        infoset.played_cards['landlord_down'])
    landlord_down_played_cards_batch = np.repeat(
        landlord_down_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         last_action_batch,
                         landlord_up_played_cards_batch,
                         landlord_down_played_cards_batch,
                         landlord_up_num_cards_left_batch,
                         landlord_down_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             last_action,
                             landlord_up_played_cards,
                             landlord_down_played_cards,
                             landlord_up_num_cards_left,
                             landlord_down_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
            'position': 'landlord',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
          }
    return obs

def _get_obs_landlord_up(infoset):
    """
    地主上家（农民）视角特征（论文 Table 5）。
    比地主多了「地主上一手 / 队友上一手」等，总维度 484（不含动作 430）。
    """
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    last_landlord_action = _cards2array(
        infoset.last_move_dict['landlord'])
    last_landlord_action_batch = np.repeat(
        last_landlord_action[np.newaxis, :],
        num_legal_actions, axis=0)
    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    last_teammate_action = _cards2array(
        infoset.last_move_dict['landlord_down'])
    last_teammate_action_batch = np.repeat(
        last_teammate_action[np.newaxis, :],
        num_legal_actions, axis=0)
    teammate_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)
    teammate_num_cards_left_batch = np.repeat(
        teammate_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    teammate_played_cards = _cards2array(
        infoset.played_cards['landlord_down'])
    teammate_played_cards_batch = np.repeat(
        teammate_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         landlord_played_cards_batch,
                         teammate_played_cards_batch,
                         last_action_batch,
                         last_landlord_action_batch,
                         last_teammate_action_batch,
                         landlord_num_cards_left_batch,
                         teammate_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             landlord_played_cards,
                             teammate_played_cards,
                             last_action,
                             last_landlord_action,
                             last_teammate_action,
                             landlord_num_cards_left,
                             teammate_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
            'position': 'landlord_up',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
          }
    return obs

def _get_obs_landlord_down(infoset):
    """
    地主下家（农民）视角特征，结构和上家相同（论文 Table 5），
    只是「队友」换成了 landlord_up。
    """
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    last_landlord_action = _cards2array(
        infoset.last_move_dict['landlord'])
    last_landlord_action_batch = np.repeat(
        last_landlord_action[np.newaxis, :],
        num_legal_actions, axis=0)
    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    last_teammate_action = _cards2array(
        infoset.last_move_dict['landlord_up'])
    last_teammate_action_batch = np.repeat(
        last_teammate_action[np.newaxis, :],
        num_legal_actions, axis=0)
    teammate_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)
    teammate_num_cards_left_batch = np.repeat(
        teammate_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    teammate_played_cards = _cards2array(
        infoset.played_cards['landlord_up'])
    teammate_played_cards_batch = np.repeat(
        teammate_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         landlord_played_cards_batch,
                         teammate_played_cards_batch,
                         last_action_batch,
                         last_landlord_action_batch,
                         last_teammate_action_batch,
                         landlord_num_cards_left_batch,
                         teammate_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             landlord_played_cards,
                             teammate_played_cards,
                             last_action,
                             last_landlord_action,
                             last_teammate_action,
                             landlord_num_cards_left,
                             teammate_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
            'position': 'landlord_down',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
          }
    return obs
