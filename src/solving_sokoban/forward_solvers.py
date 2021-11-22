import copy
import heapq
import random

import numpy as np

from collections import defaultdict, deque
from itertools import permutations
from tensorflow.keras import datasets, layers, models

from .game_engine import State


class ForwardSolvers:
    def q_learning(start_state, n_epochs, moves_per_epoch, epsilon, alpha, gamma):
        Q = defaultdict(int)
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        completed = [0] * 25
        comp = False
        vanish_rate = (epsilon[1] / epsilon[0]) ** (1 / n_epochs)
        for epoch in range(n_epochs):

            completed = [int(comp)] + completed[:-1]
            comp = False
            epsilon_temp = epsilon[0] * vanish_rate ** epoch
            if epoch in list(range(n_epochs // 10, n_epochs, n_epochs // 10)):
                print(f"Epoch no {epoch}")
                print("Completed last 25 epochs: %s" % (sum(completed)))
                print("Exploration rate %s" % epsilon_temp)

            state = copy.deepcopy(start_state)
            for i in range(moves_per_epoch):
                if random.randint(0, 100) < epsilon_temp:
                    move = random.choice(moves)
                else:
                    rep_moves = [state.rep() + (move,) for move in moves]
                    q = np.array([Q[rep] if rep in Q else 0 for rep in rep_moves])
                    move = moves[np.random.choice(np.flatnonzero(q == max(q)))]

                old_state = copy.deepcopy(state)
                reward = state.next_state(move)

                q_max = 0
                for next_move in moves:
                    rep = state.rep() + (next_move,)
                    if rep in Q.keys():
                        q_max = max(q_max, Q[rep])

                rep = old_state.rep() + (move,)
                Q[rep] = Q[rep] + alpha * (reward + gamma * q_max - Q[rep])

                if state.filled_storages == state.n_storages:
                    comp = True
                    break
        return Q
