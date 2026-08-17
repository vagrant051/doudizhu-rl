import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

deck = []
for i in range(3, 15):
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])
deck.extend([20, 30])


def generate():
    _deck = deck.copy()
    np.random.shuffle(_deck)
    card_play_data = {
        "landlord": _deck[:20],
        "landlord_up": _deck[20:37],
        "landlord_down": _deck[37:54],
        "three_landlord_cards": _deck[17:20],
    }
    for key in card_play_data:
        card_play_data[key].sort()
    return card_play_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成固定评测牌局")
    parser.add_argument("--output", default="eval_data", type=str)
    parser.add_argument("--num_games", default=1000, type=int)
    flags = parser.parse_args()
    output_pickle = flags.output + ".pkl"
    data = [generate() for _ in range(flags.num_games)]
    with open(output_pickle, "wb") as g:
        pickle.dump(data, g, pickle.HIGHEST_PROTOCOL)
    print(f"saved {len(data)} games to {output_pickle}")
