import numpy as np
import os
import time

from copy import deepcopy
from itertools import count

from solving_sokoban.game_engine import State
from solving_sokoban.forward_solvers import ForwardSolvers
from solving_sokoban.classes import *
from solving_sokoban.levels import levels


def solve_and_visualise(level, solver, n_epochs):
    os.system("clear")
    solver_dic = {"q_learning": ForwardSolvers.q_learning}

    start_state = levels[level]
    print(start_state)
    print("Solving level...")
    policy = solver_dic[solver](start_state=start_state, n_epochs=n_epochs, moves_per_epoch=20, epsilon=(0.5, 0.05), alpha=0.95, gamma=0.95)
    state = deepcopy(start_state)
    
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for i in count():
        rep_moves = [state.rep() + (move,) for move in moves]
        q = np.array([policy[rep] if rep in policy else 0 for rep in rep_moves])
        move = moves[np.random.choice(np.flatnonzero(q == max(q)))]
        os.system("clear")
        print(start_state)
        print(f"Solved with: {solver}")
        print(f"Move no: {i}")
        state.next_state(move)
        print(state)
        if state.filled_storages == state.n_storages or i == 30:
            break
        time.sleep(1)


#solve_and_visualise("level_easiest", "q_learning", 25)
#solve_and_visualise("level_easy", "q_learning", 50)
solve_and_visualise("level_with_3_boxes", "q_learning", 10_000)
#solve_and_visualise("hard_human_level", "q_learning", 25_000)
#solve_and_visualise("original_level_1", "q_learning")
