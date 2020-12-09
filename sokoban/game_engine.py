#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:28:34 2020
@author: madsthoisen
"""
import numpy as np

from copy import deepcopy


def manh(a, b):
    return int(np.linalg.norm(np.array(a) - np.array(b), 1))


def min_manh(boxes, storages):
    return sum(min(list(map(lambda s: manh(b, s), storages))) for b in boxes)


class State():
    def __init__(self, width, height, player, boxes, storages, walls):
        self.height, self.width = height, width

        self.state = np.zeros((height, width, 4), dtype=int)

        self.state[player[1]][player[0]] += np.array([1, 0, 0, 0])
        self.player = player

        for box in boxes:
            self.state[box[1]][box[0]] += np.array([0, 1, 0, 0])
        for storage in storages:
            self.state[storage[1]][storage[0]] += np.array([0, 0, 1, 0])
        for wall in walls:
            self.state[wall[1]][wall[0]] += np.array([0, 0, 0, 1])

        legal_squares = {(0, 0, 0, 0),  # Empty square
                         (1, 0, 0, 0),  # Player
                         (0, 1, 0, 0),  # Box
                         (0, 0, 1, 0),  # Storage
                         (0, 0, 0, 1),  # Wall
                         (1, 0, 1, 0),  # Player, Storage
                         (0, 1, 1, 0)   # Box, Storage
                         }

        self.filled_storages = [0, len(storages)]

        self.boxes = boxes
        self.storages = storages

        for row in self.state:
            for square in row:
                if tuple(square) == (0, 1, 1, 0):
                    self.filled_storages[0] += 1
                assert tuple(square) in legal_squares, \
                    'Illegal square created: %s' % square

        assert len(boxes) == len(storages), '''%d boxes and %d storages (must
            be equal)''' % (len(boxes), len(storages))

        self.manhattan = min_manh(boxes, storages)

    def __str__(self):
        types = {(0, 0, 0, 0): ' ',  # Empty square
                 (1, 0, 0, 0): 'p',  # Player
                 (0, 1, 0, 0): 'B',  # Box
                 (0, 0, 1, 0): 'S',  # Storage
                 (0, 0, 0, 1): '#',  # Wall
                 (1, 0, 1, 0): 'P',  # Player, Storage
                 (0, 1, 1, 0): '$'   # Box, Storage
                 }

        out = ''
        for y in range(self.height):
            for x in range(self.width):
                out += types[tuple(self.state[y][x])]
            out += '\n'
        return out

    def __lt__(self, other):
        return False

    def rep(self):
        return self.player + tuple(self.boxes)

    def next_state(self, move):
        reward = -1
        reward_end = 1000
        reward_fill = 100

        legal_moves = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        assert move in legal_moves, \
            '%s is an illegal move. Valid moves: %s' % (move, legal_moves)

        # Temporary player position
        pos = (self.player[0] + move[0], self.player[1] + move[1])
        # Position next to temporary player position in direction of move
        pos_adj = (self.player[0] + 2*move[0], self.player[1] + 2*move[1])

        if min(pos) < 0 or pos[0] >= self.width or pos[1] >= self.height:
            return self

        state_pos = tuple(self.state[pos[1]][pos[0]])

        p = np.array([1, 0, 0, 0])  # Player square
        b = np.array([0, 1, 0, 0])  # Box square

        if state_pos == (0, 0, 0, 0):  # Moving onto empty square
            self.state[pos[1]][pos[0]] = p
            self.state[self.player[1]][self.player[0]] -= p
            self.player = pos

        if state_pos[1] == 1:
            if (min(pos_adj) < 0 or
                    pos_adj[0] >= self.width or pos_adj[1] >= self.height):
                return self

            state_pos_adj = tuple(self.state[pos_adj[1]][pos_adj[0]])

            if state_pos_adj in {(0, 0, 0, 0), (0, 0, 1, 0)}:
                self.state[self.player[1]][self.player[0]] -= p

                self.state[pos[1]][pos[0]] += p
                self.state[pos[1]][pos[0]] -= b
                self.boxes.remove(pos)

                self.state[pos_adj[1]][pos_adj[0]] += b
                self.boxes.append(pos_adj)

                self.player = pos

                if state_pos_adj == (0, 0, 1, 0):
                    self.filled_storages[0] += 1
                    reward += reward_fill

                if state_pos == (0, 1, 1, 0):
                    self.filled_storages[0] -= 1
                    reward -= reward_fill

        if self.filled_storages[0] == self.filled_storages[1]:
            reward += reward_end

        old_manhattan = self.manhattan

        self.manhattan = min_manh(self.boxes, self.storages)

        reward += (old_manhattan - self.manhattan)
        return reward

    def prev_states(self):
        previous_states = set()
        legal_moves = {(1, 0), (-1, 0), (0, 1), (0, -1)}

        for move in legal_moves:
            temp_state = deepcopy(self)
            # Temporary player position
            pos = (self.player[0] - move[0], self.player[1] - move[1])
            # Position next to current player in direction of movement
            pos_opp = (self.player[0] + move[0], self.player[1] + move[1])

            if min(pos) < 0 or pos[0] >= self.width or pos[1] >= self.height:
                previous_states.add(deepcopy(self))
                continue

            p = np.array([1, 0, 0, 0])  # Player square
            b = np.array([0, 1, 0, 0])  # Box square

            state_pos = tuple(self.state[pos[1]][pos[0]])
            if state_pos in {(0, 0, 0, 0), (0, 0, 1, 0)}:
                temp_state.player = pos
                temp_state.state[pos[1]][pos[0]] += p
                temp_state.state[self.player[1]][self.player[0]] -= p
                previous_states.add(deepcopy(temp_state))
                state_pos_opp = tuple(self.state[pos_opp[1]][pos_opp[0]])
                if state_pos_opp[1] == 1:
                    temp_state.state[pos_opp[1]][pos_opp[0]] -= b
                    temp_state.boxes.remove(pos_opp)
                    if temp_state.state[pos_opp[1]][pos_opp[0]][2] == 1:
                        temp_state.filled_storages[0] -= 1
                    temp_state.state[self.player[1]][self.player[0]] += b
                    temp_state.boxes.append(self.player)
                    if temp_state.state[self.player[1]][self.player[0]][2] == 1:
                        temp_state.filled_storages[0] += 1
                    previous_states.add(deepcopy(temp_state))
        return previous_states

    def possible_final_states(self):
        players = set()
        filled_state_no_player = deepcopy(self)
        filled_state_no_player.boxes = filled_state_no_player.storages
        for h in range(self.height):
            for w in range(self.width):
                square = tuple(self.state[h][w])
                if square in {(1, 0, 0, 0), (0, 1, 0, 0)}:
                    filled_state_no_player.state[h][w] = (0, 0, 0, 0)
                elif square[2] == 1:
                    if square[1] == 0:
                        filled_state_no_player.state[h][w] = (0, 1, 1, 0)
                    for s in {(1, 0), (-1, 0), (0, 1), (0, -1)}:
                        no_player = {(0, 0, 0, 1), (0, 1, 1, 0)}
                        if 0 <= h + 2*s[0] < self.height and \
                                0 <= w + 2*s[1] < self.width:
                            adj = tuple(self.state[h + s[0]][w + s[1]])
                            adj_2 = tuple(self.state[h + 2*s[0]][w + 2*s[1]])
                            if adj not in no_player and adj_2 not in no_player:
                                players.add((w + s[1], h + s[0]))

        final_states = set()
        for player in players:
            temp_state = deepcopy(filled_state_no_player)
            temp_state.state[player[1]][player[0]] = np.array([1, 0, 0, 0])
            temp_state.player = player
            temp_state.filled_storages[0] = temp_state.filled_storages[1]
            final_states.add(temp_state)
        return final_states

    def get_boxes(self):
        return set([(c, r) for c in range(self.width)
                    for r in range(self.height) if self.state[r][c][1] == 1])
