#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 22:39:53 2020

@author: madsthoisen
"""
import copy
import heapq
import random

import numpy as np

from collections import deque
from game_engine import State
from itertools import permutations
from tensorflow.keras import datasets, layers, models

class ForwardSolvers():
    def q_learning(start_state, n_epochs, moves_per_epoch, epsilon, alpha, gamma):
        Q = {}
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        completed = [0]*25
        comp = False
        vanish_rate = (epsilon[1] / epsilon[0]) ** (1 / n_epochs)
        for epoch in range(n_epochs):

            completed = [1] + completed[:-1] if comp else [0] + completed[:-1]
            comp = False
            epsilon_temp = epsilon[0] * vanish_rate**epoch

            print('Completed last 25 epochs: %s' % (sum(completed)))
            print('Exploration rate %s' % epsilon_temp)

            state = copy.deepcopy(start_state)
            for i in range(moves_per_epoch):
                if random.randint(0, 100) < epsilon_temp:
                    move = random.choice(moves)
                else:
                    rep_moves = [state.rep() + move for move in moves]
                    q = [Q[rep] if rep in Q.keys() else 0 for rep in rep_moves]
                    q = np.array(q)
                    move = moves[np.random.choice(np.flatnonzero(q == max(q)))]

                old_state = copy.deepcopy(state)
                reward = state.next_state(move)

                q_max = 0
                for next_move in moves:
                    rep = state.rep() + next_move
                    if rep in Q.keys():
                        q_max = max(q_max, Q[rep])

                rep = old_state.rep() + move
                if rep in Q.keys():
                    Q[rep] = Q[rep] + alpha * (reward + gamma * q_max - Q[rep])
                else:
                    Q[rep] = 0 + alpha * (reward + gamma * q_max - 0)

                if state.filled_storages[0] == state.filled_storages[1]:
                    comp = True
                    break

            print(state)

    def deep_q_learning(start_state, n_epochs, moves_per_epoch, epsilon, gamma):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                    input_shape=(start_state.height, start_state.width, 4)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        # model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(4, activation='linear'))

        model.compile(optimizer='adam', loss='mean_squared_error')                

        history = model.fit(np.array([start_state.state]), 
                            np.array([[-moves_per_epoch]*4]), epochs=1)
        
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        completed = [0]*25
        comp = False
        vanish_rate = (epsilon[1] / epsilon[0]) ** (1 / n_epochs)
        for epoch in range(n_epochs):

            completed = [1] + completed[:-1] if comp else [0] + completed[:-1]
            comp = False
            epsilon_temp = epsilon[0] * vanish_rate**epoch

            print('Completed last 25 epochs: %s' % (sum(completed)))
            print('Exploration rate %s' % epsilon_temp)

            # print(model.predict(np.array([start_state.state])))

            state = copy.deepcopy(start_state)
            #epoch_history = {'states': [], 'q': []}
            
            acc_reward = 0
            for i in range(moves_per_epoch):
                
                # print('epoch: %s, move: %s, epsilon: %s' % (epoch, i, epsilon_temp))
                # print(acc_reward)
                # print(state)
                q = model.predict(np.array([state.state]))

                if random.randint(0, 100) < epsilon_temp*100:
                    action = random.randint(0, 3)
                else:
                    action = np.random.choice(np.flatnonzero(q[0] == max(q[0])))
                move = moves[action]
                old_state = copy.deepcopy(state)
                reward = state.next_state(move)
                acc_reward += reward
                new_q = model.predict(np.array([state.state]))
                # print(max(new_q[0]))
                q[0][action] = reward + gamma * max(new_q[0])
                # if reward > -1:
                #     print(reward)
                #     print(state)
                #epoch_history['states'].append()
                #epoch_history['q'].append(q[0])

                history = model.fit(np.array([state.state]), np.array([q[0]]), epochs=1, verbose = 0)
                #history = model.fit(np.array(epoch_history['states']), np.array(epoch_history['q']), epochs=1)
                epoch_history = {'states': [], 'q': []}
                # print(state)
                if state.filled_storages[0] == state.filled_storages[1]:
                    comp = True
                    break
        return


level_easiest = {'width': 8,
              'height': 8,
              'player': (1, 1),
              'boxes': [(2, 3)],
              'storages': [(2, 6)],
              'walls': set([(0, i) for i in range(8)] +
                           [(7, i) for i in range(8)] +
                           [(i, 0) for i in range(8)] +
                           [(i, 7) for i in range(8)])}


start_state = State(*list(level_easiest.values()))
print(start_state.__str__())
ForwardSolvers.deep_q_learning(start_state, 1000, 15, (.25, .05), .95)





level_easy = {'width': 8,
              'height': 8,
              'player': (1, 1),
              'boxes': [(2, 2)],
              'storages': [(2, 3)],
              'walls': set([(0, i) for i in range(8)] +
                           [(7, i) for i in range(8)] +
                           [(i, 0) for i in range(8)] +
                           [(i, 7) for i in range(8)])}


# start_state = State(*list(level_easy.values()))
# print(start_state.__str__())
# ForwardSolvers.deep_q_learning(start_state, 100, 5, (.2, .05), .95)
#ForwardSolvers.q_learning(start_state, 10000, 100, (.1, .05), .9, .9)

    
level_ok = {'width': 6,
            'height': 6,
            'player': (1, 1),
            'boxes': [(2, 2), (3, 3), (2, 4)],
            'storages': [(2, 2), (3, 4), (4, 3)],
            'walls': set([(0, i) for i in range(6)] +
                         [(5, i) for i in range(6)] +
                         [(i, 0) for i in range(6)] +
                         [(i, 5) for i in range(6)])}


# start_state = State(*list(level_ok.values()))
# print(start_state.__str__())
#ForwardSolvers.q_learning(start_state, 1000, 100, (0.50, 0.05), 1, 1)
# ForwardSolvers.deep_q_learning(start_state, 10000, 100, (.8, .1), 1)

level_blah = {'width': 8,
              'height': 9,
              'player': (2, 2),
              'boxes': [(3, 2), (4, 3), (4, 4), (1, 6), (3, 6), (4, 6), (5, 6)],
              'storages': [(1, 2), (5, 3), (1, 4), (4, 5), (3, 6), (6, 6), (4, 7)],
              'walls': [(2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
                        (0, 1), (1, 1), (2, 1), (6, 1),
                        (0, 2), (6, 2),
                        (0, 3), (1, 3), (2, 3), (6, 3),
                        (0, 4), (2, 4), (3, 4), (6, 4),
                        (0, 5), (2, 5), (6, 5), (7, 5),
                        (0, 6), (7, 6),
                        (0, 7), (7, 7),
                        (0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8)]}


# start_state = State(*list(level_blah.values()))
# print(start_state.__str__())
#ForwardSolvers.q_learning(start_state, 10000, 300, (.8, .1), 1, 1)
# ForwardSolvers.deep_q_learning(start_state, 10000, 100, (.8, .1), 1)


# print('\n|-------|\n')
# original_level_1 = {'width': 19,
#                     'height': 11,
#                     'player': (11, 8),
#                     'boxes': [(2, 7), (5, 2), (5, 4), (5, 7), (7, 3), (7, 4)],
#                     'storages': [(16, 6), (16, 7), (16, 8), (17, 6), (17, 7), (17, 8)],
#                     'walls': [(0, 5), (0, 6), (0, 7), (0, 8),
#                               (1, 5), (1, 8),
#                               (2, 3), (2, 4), (2, 5), (2, 8),
#                               (3, 3), (3, 8),
#                               (4, 0), (4, 1), (4, 2), (4, 3), (4, 5), (4, 6),
#                               (4, 8), (4, 9), (4, 10),
#                               (5, 0), (5, 10),
#                               (6, 0), (6, 5), (6, 6), (6, 8), (6, 10),
#                               (7, 0), (7, 5), (7, 6), (7, 8), (7, 10),
#                               (8, 0), (8, 1), (8, 2), (8, 3), (8, 8), (8, 10),
#                               (9, 3), (9, 4), (9, 5), (9, 6), (9, 10),
#                               (10, 6), (10, 8), (10, 9), (10, 10),
#                               (11, 6), (11, 9),
#                               (12, 6), (12, 8), (12, 9),
#                               (13, 5), (13, 6), (13, 8), (13, 9),
#                               (14, 5), (14, 9),
#                               (15, 5), (15, 9),
#                               (16, 5), (16, 9),
#                               (17, 5), (17, 9),
#                               (18, 5), (18, 6), (18, 7), (18, 8), (18, 9)]}

# start_state = State(*list(original_level_1.values()))
