import numpy as np
import os
import random
import time

from collections import deque
from copy import deepcopy
from itertools import count
from math import log

from solving_sokoban.game_engine import State
from solving_sokoban.deep_q_learning import agent, train
from solving_sokoban.classes import *
from solving_sokoban.levels import levels


def print_status(epoch, completers, epsilon, start_state, state, total_training_rewards):
    os.system("clear")
    status = (f"Epoch no: {epoch}\n"
              f"{sum(completers)} epochs completed of last 25\n"
              f"{[int(x) for x in completers]}\n"
              "########################\n"
              f"epsilon: {epsilon}\n"
              f"Start state:\n{start_state}\n"
              f"Final state:\n{state}\n"
              f"round reward: {total_training_rewards}\n")
    print(status)


def solve(level, n_epochs, moves_per_epoch):
    os.system("clear")

    start_state = levels[level]

    train_rate = 20
    copy_rate = 100

    epsilon = 1
    min_epsilon = 0.05
    decay = min_epsilon**(1/n_epochs)
    
    n_actions = 4
    state_shape = (start_state.height, start_state.width, n_actions)
    model = agent(state_shape, n_actions)
   
    target_model = agent(state_shape, n_actions)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=50_000)
    completers = deque(maxlen=25)

    X = []  # states
    y = []  # actions

    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # r, l, d, u

    count = 0
    for epoch in range(n_epochs):
        state = deepcopy(start_state)
        total_training_rewards = 0

        for i in range(moves_per_epoch):
            count += 1
            observation = deepcopy(state.state)
            if np.random.rand() <= epsilon:
                action_idx = random.choice([0, 1, 2, 3])
            else:
                observation_reshaped = state.state.reshape([1] + list(state.state.shape))
                predicted = model.predict(observation_reshaped)
                action_idx = np.argmax(predicted)

            action = moves[action_idx]
            reward = state.next_state(action)
            completed = (state.n_storages == state.filled_storages)

            replay_memory.append([observation, action_idx, reward, state.state, completed])
            total_training_rewards += reward

            if count % train_rate == 0:
                train(state, replay_memory, model, target_model, completed, state_shape)

            if count % copy_rate == 0:
                target_model.set_weights(model.get_weights())

            if completed:
                break

        completers.append(completed)
        print_status(epoch, completers, epsilon, start_state, state, total_training_rewards)

        epsilon *= decay

#solve(level="level_easiest", n_epochs=5000, moves_per_epoch=4)
solve("level_easy", 5000, 15)
