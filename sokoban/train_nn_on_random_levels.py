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

print(x_vals[:5])
print(x_vals.shape) #(345567, 8, 8, 4)

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu',
          input_shape=(8, 8, 4)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')                
history = model.fit(x_vals, y_vals, epochs=50)


def pr(st):
    types = {(0, 0, 0, 0): ' ',  # Empty square
             (1, 0, 0, 0): 'p',  # Player
             (0, 1, 0, 0): 'B',  # Box
             (0, 0, 1, 0): 'S',  # Storage
             (0, 0, 0, 1): '#',  # Wall
             (1, 0, 1, 0): 'P',  # Player, Storage
             (0, 1, 1, 0): '$'   # Box, Storage
             }

    out = ''
    for y in range(len(st)):
        for x in range(len(st[0])):
            out += types[tuple(st[y][x])]
        out += '\n'
    return out


for _ in range(5):
    s = random.choice(x_vals)
    pred = model.predict(np.array([s]))
    print(pr(s))
    print(pred)


model.save('model_trained')

