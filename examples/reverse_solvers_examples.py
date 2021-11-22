import time
import os

from copy import deepcopy

from solving_sokoban.game_engine import State
from solving_sokoban.reverse_solvers import ReverseSolvers
from solving_sokoban.classes import *
from solving_sokoban.levels import levels


def solve_and_visualise(level, solver):
    os.system("clear")
    solver_dic = {"bfs": ReverseSolvers.bfs, "a_star": ReverseSolvers.a_star}

    start_state = levels[level]
    print(start_state)
    print("Solving level...")
    result, policy, states_seen = solver_dic[solver](start_state)

    state = deepcopy(start_state)
    for i, move in enumerate(policy):
        os.system("clear")
        print(start_state)
        print(f"Solved with: {solver}")
        print(f"States seen: {states_seen}")
        print(f"Policy: {policy}")
        print(f"Move no: {i + 1}")
        state.next_state(move)
        print(state)
        time.sleep(1)


solve_and_visualise("level_easiest", "bfs")
input("Press Enter to continue")
solve_and_visualise("level_easy", "bfs")
input("Press Enter to continue")
solve_and_visualise("level_with_3_boxes", "bfs")
input("Press Enter to continue")
solve_and_visualise("hard_human_level", "a_star")
input("Press Enter to continue")
solve_and_visualise("original_level_1", "a_star")
