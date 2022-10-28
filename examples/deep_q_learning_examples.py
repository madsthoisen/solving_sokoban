import numpy as np
import os
import random
import time

from collections import deque
from copy import deepcopy
from itertools import count

from solving_sokoban.game_engine import State
from solving_sokoban.deep_q_learning import agent, train
from solving_sokoban.classes import *
from solving_sokoban.levels import levels


def solve(level, solver, n_epochs, moves_per_epoch):
    os.system("clear")

    start_state = levels[level]

    epsilon = 1
    max_epsilon = 1
    min_epsilon = 0.01
    decay = 0.01
     
    state_shape = (start_state.height, start_state.width, 4)
    n_actions = 4
    model = agent(state_shape, n_actions)
    
    # Target Model (updated every 100 steps)
    target_model = agent(state_shape, n_actions)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=50_000)
    completers = deque(maxlen=50)

    target_update_counter = 0
    
    # X = states, y = actions
    X = []
    y = []

    steps_to_update_target_model = 0
    
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for episode in range(n_epochs):
        state = deepcopy(start_state)
        observation = state.state
        total_training_rewards = 0
        done = False
        for i in count():
            steps_to_update_target_model += 1
            random_number = np.random.rand()
            if random_number <= epsilon:
                action_idx = random.sample([0, 1, 2, 3], 1)[0]
                action = moves[action_idx]
            else:
                observation_reshaped = state.state.reshape([1] + list(state.state.shape))
                predicted = model.predict(observation_reshaped)
                action_idx = np.argmax(predicted)
                action = moves[action_idx]
            reward = state.next_state(action)
            if state.n_storages == state.filled_storages:
                done = True

            

            replay_memory.append([observation, action_idx, reward, state.state, done])
            
            if i == moves_per_epoch:
                done = True
            if steps_to_update_target_model % 4 == 0 or done:
                train(state, replay_memory, model, target_model, done, state_shape)

            total_training_rewards += reward

            if done:
                if state.n_storages == state.filled_storages:
                    completers.append(1)
                else:
                    completers.append(0)
                os.system("clear")
                print(f"Epoch no: {episode}")
                print(f"{sum(completers)} epochs completed of last 50")
                print(completers)
                print(model.summary())
                print("#############################")
                print(start_state)
                print()
                print(state)

                if steps_to_update_target_model >= 50:
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

solve("level_easiest", "q_learning", 5000, 5)
#solve("level_easy", "q_learning", 1000, 15)
