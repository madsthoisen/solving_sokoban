#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:51:06 2020

@author: madsthoisen
"""
import pytest

from game_engine import State


# Define level and test that it prints, creates storages, and completes
level_ok = {'width': 6,
            'height': 6,
            'player': (1, 1),
            'boxes': [(2, 2), (3, 3), (2, 4)],
            'storages': [(2, 2), (3, 4), (4, 3)],
            'walls': set([(0, i) for i in range(6)] +
                         [(5, i) for i in range(6)] +
                         [(i, 0) for i in range(6)] +
                         [(i, 5) for i in range(6)])}

level_ok_moves = [(0, 1), (0, 1), (1, 0), (1, 0),
                  (-1, 0), (-1, 0), (0, 1), (1, 0)]


def test_level_ok_prints_right():
    state = State(*list(level_ok.values()))
    out = state.__str__()
    lev = '######\n#p   #\n# $  #\n#  BS#\n# BS #\n######\n'

    assert lev == out


def test_level_ok_counting_storages():
    state = State(*list(level_ok.values()))

    assert state.filled_storages == [1, 3]


def test_level_ok_completes_level():
    state = State(*list(level_ok.values()))
    for move in level_ok_moves:
        state.next_state(move)
    assert state.filled_storages == [3, 3]


# Define level and test that it fails
level_box_on_player = {'width': 6,
                       'height': 6,
                       'player': (1, 1),
                       'boxes': [(1, 1)],
                       'storages': [],
                       'walls': []}


def test_level_box_on_player_fails():
    with pytest.raises(AssertionError):
        State(*list(level_box_on_player.values()))


# Define level and test that it fails
level_box_on_wall = {'width': 6,
                     'height': 6,
                     'player': (2, 2),
                     'boxes': [(1, 1)],
                     'storages': [],
                     'walls': [(1, 1)]}


def test_level_box_on_wall_fails():
    with pytest.raises(AssertionError):
        State(*list(level_box_on_wall.values()))


# Define level and test that it fails
level_player_on_wall = {'width': 6,
                        'height': 6,
                        'player': (1, 1),
                        'boxes': [],
                        'storages': [],
                        'walls': [(1, 1)]}


def test_level_player_on_wall_fails():
    with pytest.raises(AssertionError):
        State(*list(level_player_on_wall.values()))


# Define level and test that it fails
level_storage_box_mismatch = {'width': 6,
                              'height': 6,
                              'player': (1, 1),
                              'boxes': [(2, 2), (3, 3)],
                              'storages': [(2, 2), (3, 4), (4, 3)],
                              'walls': []}


def test_level_storage_box_mismatch():
    with pytest.raises(AssertionError):
        State(*list(level_storage_box_mismatch.values()))


# Define level and moves so player potentially moves out of screen
level_move_out_of_screen = {'width': 6,
                            'height': 4,
                            'player': (1, 1),
                            'boxes': [(2, 3)],
                            'storages': [(3, 3)],
                            'walls': []}

move_out_left = [(-1, 0), (-1, 0)]
move_out_right = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]
move_out_up = [(0, -1), (0, -1)]
move_out_down = [(0, 1), (0, 1), (0, 1)]


def test_move_out_left():
    state = State(*list(level_move_out_of_screen.values()))
    for move in move_out_left:
        state.next_state(move)
    assert state.player == (0, 1)


def test_move_out_right():
    state = State(*list(level_move_out_of_screen.values()))
    for move in move_out_right:
        state.next_state(move)
    assert state.player == (5, 1)


def test_move_out_up():
    state = State(*list(level_move_out_of_screen.values()))
    for move in move_out_up:
        state.next_state(move)
    assert state.player == (1, 0)


def test_move_out_down():
    state = State(*list(level_move_out_of_screen.values()))
    for move in move_out_down:
        state.next_state(move)
    assert state.player == (1, 3)


# Define level and moves so player potentially moves into walls
level_move_into_walls = {'width': 4,
                         'height': 6,
                         'player': (1, 1),
                         'boxes': [(2, 2)],
                         'storages': [(2, 3)],
                         'walls': set([(0, i) for i in range(6)] +
                                      [(3, i) for i in range(6)] +
                                      [(i, 0) for i in range(4)] +
                                      [(i, 5) for i in range(4)])}

move_into_wall_left = [(-1, 0), (-1, 0), (-1, 0)]
move_into_wall_right = [(1, 0), (1, 0), (1, 0), (1, 0)]
move_into_wall_up = [(0, -1), (0, -1), (0, -1), (0, -1)]
move_into_wall_down = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]


def test_move_into_wall_left():
    state = State(*list(level_move_into_walls.values()))
    for move in move_into_wall_left:
        state.next_state(move)
    assert state.player == (1, 1)


def test_move_into_wall_right():
    state = State(*list(level_move_into_walls.values()))
    for move in move_into_wall_right:
        state.next_state(move)
    assert state.player == (2, 1)


def test_move_into_wall_up():
    state = State(*list(level_move_into_walls.values()))
    for move in move_into_wall_up:
        state.next_state(move)
    assert state.player == (1, 1)


def test_move_into_wall_down():
    state = State(*list(level_move_into_walls.values()))
    for move in move_into_wall_down:
        state.next_state(move)
    assert state.player == (1, 4)


# Define level and moves so player pushes two boxes in a row
level_two_boxes = {'width': 8,
                   'height': 8,
                   'player': (1, 1),
                   'boxes': [(2, 2), (3, 2), (2, 4), (2, 5)],
                   'storages': [(3, 6), (4, 6), (5, 6), (6, 6)],
                   'walls': set([(0, i) for i in range(8)] +
                                [(7, i) for i in range(8)] +
                                [(i, 0) for i in range(8)] +
                                [(i, 7) for i in range(8)])}


push_two_boxes_from_left = [(0, 1), (1, 0), (1, 0)]
push_two_boxes_from_right = [(1, 0), (1, 0), (1, 0), (0, 1), (-1, 0), (-1, 0)]
push_two_boxes_from_up = [(0, 1), (0, 1), (1, 0), (0, 1), (0, 1)]
push_two_boxes_from_down = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (1, 0),
                            (0, -1), (0, -1)]


def test_push_two_boxes_from_left():
    state = State(*list(level_two_boxes.values()))
    for move in push_two_boxes_from_left:
        state.next_state(move)
    assert state.player == (1, 2)


def test_push_two_boxes_from_right():
    state = State(*list(level_two_boxes.values()))
    for move in push_two_boxes_from_right:
        state.next_state(move)
    assert state.player == (4, 2)


def test_push_two_boxes_from_up():
    state = State(*list(level_two_boxes.values()))
    for move in push_two_boxes_from_up:
        state.next_state(move)
    assert state.player == (2, 3)


def test_push_two_boxes_from_down():
    state = State(*list(level_two_boxes.values()))
    for move in push_two_boxes_from_down:
        state.next_state(move)
    assert state.player == (2, 6)


# Define level and test the reverse game engine, no boxes
level_reverse_no_boxes = {'width': 5,
                          'height': 5,
                          'player': (2, 2),
                          'boxes': [],
                          'storages': [],
                          'walls': set([(0, i) for i in range(5)] +
                                       [(4, i) for i in range(5)] +
                                       [(i, 0) for i in range(5)] +
                                       [(i, 4) for i in range(5)])}


def test_reverse_no_boxes():
    state = State(*list(level_reverse_no_boxes.values()))
    previous_states = state.prev_states()
    previous_players = {state.player for state in previous_states}
    assert previous_players == {(2, 1), (1, 2), (3, 2), (2, 3)}


# Define level and test the reverse game engine, one box next to
level_reverse_box = {'width': 5,
                     'height': 5,
                     'player': (2, 2),
                     'boxes': [(3, 2)],
                     'storages': [(3, 3)],
                     'walls': set([(0, i) for i in range(5)] +
                                  [(4, i) for i in range(5)] +
                                  [(i, 0) for i in range(5)] +
                                  [(i, 4) for i in range(5)])}


def test_reverse_box():
    state = State(*list(level_reverse_box.values()))
    previous_states = state.prev_states()
    previous_states = {state.__str__() for state in previous_states}
    assert previous_states == {'#####\n# p #\n#  B#\n#  S#\n#####\n',
                               '#####\n#   #\n#p B#\n#  S#\n#####\n',
                               '#####\n#   #\n#pB #\n#  S#\n#####\n',
                               '#####\n#   #\n#  B#\n# pS#\n#####\n'}


# Define level and test the reverse game engine, one box on storagenext to
level_reverse_box_on_storage = {'width': 5,
                                'height': 5,
                                'player': (2, 2),
                                'boxes': [(3, 2)],
                                'storages': [(3, 2)],
                                'walls': set([(0, i) for i in range(5)] +
                                             [(4, i) for i in range(5)] +
                                             [(i, 0) for i in range(5)] +
                                             [(i, 4) for i in range(5)])}


def test_reverse_box_on_storage():
    state = State(*list(level_reverse_box_on_storage.values()))
    previous_states = state.prev_states()
    previous_states = {state.__str__() for state in previous_states}
    assert previous_states == {'#####\n#   #\n#  $#\n# p #\n#####\n',
                               '#####\n#   #\n#p $#\n#   #\n#####\n',
                               '#####\n# p #\n#  $#\n#   #\n#####\n',
                               '#####\n#   #\n#pBS#\n#   #\n#####\n'}
