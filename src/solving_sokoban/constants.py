import numpy as np


REP_NP = {
    "empty": np.array([0, 0, 0, 0]),
    "player": np.array([1, 0, 0, 0]),
    "box": np.array([0, 1, 0, 0]),
    "storage": np.array([0, 0, 1, 0]),
    "wall": np.array([0, 0, 0, 1]),
}


REP_TUP = {
    "empty": (0, 0, 0, 0),
    "player": (1, 0, 0, 0),
    "box": (0, 1, 0, 0),
    "storage": (0, 0, 1, 0),
    "wall": (0, 0, 0, 1),
    "player_storage": (1, 0, 1, 0),
    "filled_storage": (0, 1, 1, 0),
}


LEGAL_SQUARES = {
    (0, 0, 0, 0): " ",  # Empty square
    (1, 0, 0, 0): "p",  # Player
    (0, 1, 0, 0): "B",  # Box
    (0, 0, 1, 0): "S",  # Storage
    (0, 0, 0, 1): "#",  # Wall
    (1, 0, 1, 0): "P",  # Player, Storage
    (0, 1, 1, 0): "$",  # Box, Storage
}


REWARDS = {"move" : -1,
           "fill_storage": 10,
           "unfill_storage": 10,
           "manhattan": True}
