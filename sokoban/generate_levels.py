#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import random

from collections import defaultdict
from game_engine import State


def generate_end_states(inner_width, inner_height, n_boxes):
    import numpy as np
    walls = set([(0, i) for i in range(inner_height)] +
                [(inner_width - 1, i) for i in range(inner_height)] +
                [(i, 0) for i in range(inner_width)] +
                [(i, inner_height - 1) for i in range(inner_width)])
    boxes = []
    while len(boxes) < n_boxes:
        box = (np.random.randint(1, inner_width-1), np.random.randint(1,
            inner_height-1))
        if box not in walls and box not in boxes:
            boxes.append(box)
    storages = boxes[:]

    random_box = boxes[np.random.choice(len(boxes))]
    next_to = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    out = []
    for inc in next_to:
       player = (random_box[0] + inc[0], random_box[1] + inc[1])
       if 1 <= player[0] < inner_width - 1 and 1 <= player[1] < inner_height - 1:
          if player not in boxes:
              out.append([player, boxes, storages, walls])
    return out

size = 8, 8
n_levels = 100
n_moves = 50
inner_height, inner_width = 8, 8
state_value_pairs = defaultdict(list)

for _ in range(n_levels):
    n_boxes = random.choice([1, 2])
    print("level: ", _, "boxes: ", n_boxes) 
    levels = generate_end_states(inner_width, inner_height, n_boxes)
    states = [State(*[size[0], size[1]] + level) for level in levels]
    states_seen = {s.__str__() for s in states}
    state_value_pairs[0] = [state for state in states]
    for move in range(1, n_moves + 1):
        prev_states_list = []
        for state in states:
            prev_states_list.extend(state.prev_states())
        print(move, len(states), len(states_seen))
        states = set()
        for state in prev_states_list:
            if state.__str__() in states_seen:
                continue
            states.add(state)
            if state.filled_storages[0] == state.filled_storages[1]:
                state_value_pairs[0].append(state)
            else:
                state_value_pairs[-move].append(state)
            states_seen.add(state.__str__())

x_vals = []
for L in state_value_pairs.values():
    for el in L:
        x_vals.append(el.state)

x_vals = np.array(x_vals)
y_vals = []
for el in state_value_pairs.items():
    for _ in el[1]:
        y_vals.append(np.array([el[0]]))
y_vals = np.array(y_vals)

with open("data/test_levels_x.txt", "wb") as f:
    pickle.dump(x_vals, f)

with open("data/test_levels_y.txt", "wb") as f:
    pickle.dump(y_vals, f)
