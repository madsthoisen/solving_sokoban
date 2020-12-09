#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import random

from collections import defaultdict
from game_engine import State

from tensorflow.keras import datasets, layers, models


size = 20, 15
with open("data/test_levels_x.txt", "rb") as f:
    x_vals = pickle.load(f)

with open("data/test_levels_y.txt", "rb") as f:
    y_vals = pickle.load(f)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
          input_shape=(size[1], size[0], 4)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')                
history = model.fit(x_vals, y_vals, epochs=100)

# 
# s = random.choice(state_value_pairs[-2])
# pred = model.predict(np.array([s.state]))
# print(s)
# print(pred)
# 
# 
# s = random.choice(state_value_pairs[-4])
# pred = model.predict(np.array([s.state]))
# print(s)
# print(pred)
# 
# 
# s = random.choice(state_value_pairs[-6])
# pred = model.predict(np.array([s.state]))
# print(s)
# print(pred)
# 
# level_test = {'width': 10,
#               'height': 10,
#               'player': (1, 1),
#               'boxes': [(3, 1)],
#                    'storages': [(2, 1)],
#                    'walls': set([(0, i) for i in range(10)] +
#                                 [(10, i) for i in range(10)] +
#                                 [(i, 0) for i in range(10)] +
#                                 [(i, 10) for i in range(10)])}
# 
# state = State(*[size[0], size[1]] + list(level_test.values())[2:])
# print(state)
# pred = model.predict(np.array([state.state]))
# print(pred)
# 
# model.save('/Users/madsthoisen/Google Drev/Privat/Kodning/SolvingSokoban_ver2/sokoban')
# 
