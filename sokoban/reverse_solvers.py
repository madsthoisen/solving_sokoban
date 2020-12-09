#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 22:39:53 2020

@author: madsthoisen
"""
import heapq

from itertools import permutations

from game_engine import State
from collections import deque


def manh(boxes_1, boxes_2):
    add = 0
    for box_1 in boxes_1:
        dist = 1e6
        for box_2 in boxes_2:
            dist = min(dist,
                       abs(box_1[0] - box_2[0]) + abs(box_1[1] - box_2[1]))
        add += dist
    return add

def manh(boxes_1, boxes_2):
    add = 0
    for box_1 in boxes_1:
        dist = 0
        for box_2 in boxes_2:
            dist += (abs(box_1[0] - box_2[0]) + abs(box_1[1] - box_2[1]))
        add += dist
    return add


class ReverseSolvers():
    def bfs(start_state):
        states_seen = set()
        queue = deque()
        for state in start_state.possible_final_states():
            queue.appendleft((0, state))
        while True:
            move, state = queue.pop()
            if state.__str__() in states_seen:
                continue
            states_seen.add(state.__str__())
            for state_prev in state.prev_states():
                if state_prev.__str__() == start_state.__str__():
                    return True, move + 1, len(states_seen)
                queue.appendleft((move + 1, state_prev))

    def a_star(start_state):
        boxes_start_state = start_state.get_boxes()
        states_seen = set()
        priority_queue = []
        moves = {}
        for state in start_state.possible_final_states():
            heapq.heappush(priority_queue, (0, state))
            moves[state] = 0
        while True:
            priority, state = heapq.heappop(priority_queue)
            if state.__str__() in states_seen:
                continue
            states_seen.add(state.__str__())
            for state_prev in state.prev_states():
                moves[state_prev] = moves[state] + 1
                if state_prev.__str__() == start_state.__str__():
                    return True, moves[state_prev], len(states_seen)
                heuristic = manh(boxes_start_state, state_prev.boxes)
                heapq.heappush(priority_queue, (heuristic + moves[state_prev], state_prev))


level_ok = {'width': 6,
            'height': 6,
            'player': (1, 1),
            'boxes': [(2, 2), (3, 3), (2, 4)],
            'storages': [(2, 2), (3, 4), (4, 3)],
            'walls': set([(0, i) for i in range(6)] +
                         [(5, i) for i in range(6)] +
                         [(i, 0) for i in range(6)] +
                         [(i, 5) for i in range(6)])}


start_state = State(*list(level_ok.values()))
print(start_state)
result, move, states_seen = ReverseSolvers.bfs(start_state)
print(result, move, states_seen)

print('\n|-------|\n')

start_state = State(*list(level_ok.values()))
print(start_state)
result, move, states_seen = ReverseSolvers.a_star(start_state)
print(result, move, states_seen)

print('\n|-------|\n')
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
     
start_state = State(*list(level_blah.values()))
print(start_state)
result, move, states_seen = ReverseSolvers.a_star(start_state)
print(result, move, states_seen)


print('\n|-------|\n')
original_level_1 = {'width': 19,
                    'height': 11,
                    'player': (11, 8),
                    'boxes': [(2, 7), (5, 2), (5, 4), (5, 7), (7, 3), (7, 4)],
                    'storages': [(16, 6), (16, 7), (16, 8), (17, 6), (17, 7), (17, 8)],
                    'walls': [(0, 5), (0, 6), (0, 7), (0, 8),
                              (1, 5), (1, 8),
                              (2, 3), (2, 4), (2, 5), (2, 8),
                              (3, 3), (3, 8),
                              (4, 0), (4, 1), (4, 2), (4, 3), (4, 5), (4, 6), 
                              (4, 8), (4, 9), (4, 10), 
                              (5, 0), (5, 10), 
                              (6, 0), (6, 5), (6, 6), (6, 8), (6, 10),
                              (7, 0), (7, 5), (7, 6), (7, 8), (7, 10),
                              (8, 0), (8, 1), (8, 2), (8, 3), (8, 8), (8, 10),
                              (9, 3), (9, 4), (9, 5), (9, 6), (9, 10),
                              (10, 6), (10, 8), (10, 9), (10, 10),
                              (11, 6), (11, 9),
                              (12, 6), (12, 8), (12, 9),
                              (13, 5), (13, 6), (13, 8), (13, 9),
                              (14, 5), (14, 9),
                              (15, 5), (15, 9),
                              (16, 5), (16, 9),
                              (17, 5), (17, 9),
                              (18, 5), (18, 6), (18, 7), (18, 8), (18, 9)]}
     
start_state = State(*list(original_level_1.values()))
print(start_state)
result, move, states_seen = ReverseSolvers.a_star(start_state)
print(result, move, states_seen)
