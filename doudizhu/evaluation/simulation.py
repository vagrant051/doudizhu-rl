"""
评测：用预先生成的固定发牌数据，让三个座位的智能体对打，统计 WP / ADP。

WP  = 胜率；ADP = 考虑炸弹翻倍后的平均分差。
智能体可以是：
  - 权重文件路径：加载 DeepAgent（本仓库的 ResNet Q 网络）
  - "random"：随机出合法牌
  - "rlcard"：RLCard 规则 Bot
"""
import multiprocessing as mp
import pickle

from doudizhu.env.game import GameEnv


def load_card_play_models(card_play_model_path_dict):
    """按座位名字加载智能体。"""
    players = {}
    for position in ["landlord", "landlord_up", "landlord_down"]:
        name = card_play_model_path_dict[position]
        if name == "rlcard":
            from .rlcard_agent import RLCardAgent

            players[position] = RLCardAgent(position)
        elif name == "random":
            from .random_agent import RandomAgent

            players[position] = RandomAgent()
        else:
            from .deep_agent import DeepAgent

            players[position] = DeepAgent(position, name)
    return players


def mp_simulate(card_play_data_list, card_play_model_path_dict, q):
    """一个 worker：打完分到的所有牌局，把胜场和分数放进队列。"""
    players = load_card_play_models(card_play_model_path_dict)
    env = GameEnv(players)
    for card_play_data in card_play_data_list:
        env.card_play_init(card_play_data)
        while not env.game_over:
            env.step()
        env.reset()
    q.put(
        (
            env.num_wins["landlord"],
            env.num_wins["farmer"],
            env.num_scores["landlord"],
            env.num_scores["farmer"],
        )
    )


def data_allocation_per_worker(card_play_data_list, num_workers):
    """把牌局列表均分给各个进程。"""
    buckets = [[] for _ in range(num_workers)]
    for idx, data in enumerate(card_play_data_list):
        buckets[idx % num_workers].append(data)
    return buckets


def evaluate(landlord, landlord_up, landlord_down, eval_data, num_workers):
    """
    评测入口。

    landlord / landlord_up / landlord_down:
        权重路径，或 "random" / "rlcard"
    eval_data:
        pickle 文件，里面是若干局固定发牌
    """
    with open(eval_data, "rb") as f:
        card_play_data_list = pickle.load(f)

    card_play_data_list_each_worker = data_allocation_per_worker(card_play_data_list, num_workers)
    del card_play_data_list

    card_play_model_path_dict = {
        "landlord": landlord,
        "landlord_up": landlord_up,
        "landlord_down": landlord_down,
    }

    num_landlord_wins = 0
    num_farmer_wins = 0
    num_landlord_scores = 0
    num_farmer_scores = 0

    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    processes = []
    for worker_data in card_play_data_list_each_worker:
        p = ctx.Process(target=mp_simulate, args=(worker_data, card_play_model_path_dict, q))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for _ in range(num_workers):
        result = q.get()
        num_landlord_wins += result[0]
        num_farmer_wins += result[1]
        num_landlord_scores += result[2]
        num_farmer_scores += result[3]

    num_total_wins = num_landlord_wins + num_farmer_wins
    wp_landlord = num_landlord_wins / num_total_wins
    wp_farmer = num_farmer_wins / num_total_wins
    adp_landlord = num_landlord_scores / num_total_wins
    # 农民这边乘 2，是为了和地主底分对齐（地主底分 2、农民底分 1）
    adp_farmer = 2 * num_farmer_scores / num_total_wins
    print("WP results:")
    print(f"landlord : Farmers - {wp_landlord} : {wp_farmer}")
    print("ADP results:")
    print(f"landlord : Farmers - {adp_landlord} : {adp_farmer}")
    return {
        "wp_landlord": wp_landlord,
        "wp_farmer": wp_farmer,
        "adp_landlord": adp_landlord,
        "adp_farmer": adp_farmer,
        "num_games": num_total_wins,
    }
