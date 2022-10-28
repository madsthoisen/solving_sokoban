import heapq

from itertools import permutations

from collections import deque
from typing import Tuple

from .classes import *
from .game_engine import State


def manh(boxes_1, boxes_2):
    add = 0
    for box_1 in boxes_1:
        dist = 0
        for box_2 in boxes_2:
            dist += abs(box_1.x - box_2.x) + abs(box_1.y - box_2.y)
        add += dist
    return add


class ReverseSolvers:
    def bfs(start_state: State) -> Tuple[bool, int, int]:
        states_seen = set()
        queue = deque()
        for state in start_state.possible_final_states():
            queue.appendleft(([], state))
        while True:
            moves, state = queue.pop()
            if state.__str__() in states_seen:
                continue
            states_seen.add(state.__str__())
            for move, state_prev in state.prev_states():
                if state_prev.__str__() == start_state.__str__():
                    return True, (moves + [move])[::-1], len(states_seen)
                queue.appendleft((moves + [move], state_prev))

    def a_star(start_state: State) -> Tuple[bool, int, int]:
        states_seen = set()
        priority_queue = []
        moves = {}
        for state in start_state.possible_final_states():
            heapq.heappush(priority_queue, (0, state, []))
            moves[state] = (0, [])
        while True:
            priority, state, policy = heapq.heappop(priority_queue)
            if state.__str__() in states_seen:
                continue
            states_seen.add(state.__str__())
            for move, state_prev in state.prev_states():
                moves[state_prev] = (moves[state][0] + 1, moves[state][1] + [move])
                if state_prev.__str__() == start_state.__str__():
                    return True, moves[state_prev][1][::-1], len(states_seen)
                heuristic = manh(start_state.boxes, state_prev.boxes)
                heapq.heappush(
                    priority_queue,
                    (heuristic + moves[state_prev][0], state_prev, policy + [move]),
                )
