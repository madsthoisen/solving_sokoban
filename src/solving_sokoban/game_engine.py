from __future__ import annotations

import numpy as np

from copy import deepcopy
from typing import List, Set, Tuple

from .constants import LEGAL_SQUARES, REP_NP, REP_TUP, REWARDS
from .classes import Box, Player, Storage, Wall
from .exceptions import IllegalStateException




class State:
    def __init__(
        self,
        width: int,
        height: int,
        player: Player,
        boxes: Set[Box],
        storages: List[Storage],
        walls: List[Wall],
    ):

        self.width = width
        self.height = height
        self.player = player
        self.boxes = boxes
        self.storages = storages
        self.n_storages = len(storages)
        self.walls = walls

        self.state = np.zeros((height, width, 4), dtype=int)

        self.state[player.y][player.x] += REP_NP["player"]

        for box in boxes:
            self.state[box.y][box.x] += REP_NP["box"]
        for storage in storages:
            self.state[storage.y][storage.x] += REP_NP["storage"]
        for wall in walls:
            self.state[wall.y][wall.x] += REP_NP["wall"]
        self.filled_storages = 0

        if len(self.boxes) != self.n_storages:
            raise IllegalStateException(
                f"State contains {len(boxes)} boxes and {len(storages)} storages"
            )

        for row in self.state:
            for square in row:
                if tuple(square) == (0, 1, 1, 0):
                    self.filled_storages += 1
                if tuple(square) not in LEGAL_SQUARES:
                    raise IllegalStateException(f"Illegal square created: {square}")

    def __str__(self):
        out = ""
        for y in range(self.height):
            for x in range(self.width):
                out += LEGAL_SQUARES[tuple(self.state[y][x])]
            out += "\n"
        return out

    def __lt__(self, other):
        return False

    def rep(self):
        return tuple([(self.player.x, self.player.y)] + [(box.x, box.y) for box in self.boxes])

    def manhattan(self) -> int:
        def manh(b: Box, s: Storage) -> int:
            return int(np.linalg.norm(np.array([b.x, b.y]) - np.array([s.x, s.y]), 1))
        return sum(min(list(map(lambda s: manh(b, s), self.storages))) for b in self.boxes)


    def next_state(self, move: Tuple[int, int]) -> None:
        old_manhattan = self.manhattan()
        x, y = move
        reward = REWARDS["move"]
        legal_moves = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        assert move in legal_moves, "%s is an illegal move. Valid moves: %s" % (
            move,
            legal_moves,
        )

        # Temporary player position
        player_tmp = Player(self.player.x + x, self.player.y + y)

        if not (0 <= player_tmp.x < self.width) or not (
            0 <= player_tmp.y < self.height
        ):
            return reward

        tmp_square = tuple(self.state[player_tmp.y][player_tmp.x])

        if tmp_square in {REP_TUP["empty"], REP_TUP["storage"]}:
            self.state[player_tmp.y][player_tmp.x] += REP_NP["player"]
            self.state[self.player.y][self.player.x] -= REP_NP["player"]
            self.player = player_tmp

        if tmp_square[1] == 1:  ## box
            # Position next to temporary player position in direction of move
            player_tmp_adj = Player(self.player.x + 2 * x, self.player.y + 2 * y)
            tmp_square_adj = tuple(self.state[player_tmp_adj.y][player_tmp_adj.x])

            if not (0 <= player_tmp_adj.x < self.width) or not (
                0 <= player_tmp_adj.y < self.height
            ):
                return reward

            if tmp_square_adj in {REP_TUP["empty"], REP_TUP["storage"]}:
                self.state[self.player.y][self.player.x] -= REP_NP["player"]

                self.state[player_tmp.y][player_tmp.x] += REP_NP["player"]
                self.state[player_tmp.y][player_tmp.x] -= REP_NP["box"]
                self.boxes.remove(Box(player_tmp.x, player_tmp.y))

                self.state[player_tmp_adj.y][player_tmp_adj.x] += REP_NP["box"]
                self.boxes.append(Box(player_tmp_adj.x, player_tmp_adj.y))

                self.player = player_tmp

                if tmp_square_adj == REP_TUP["storage"]:
                    self.filled_storages += 1
                    reward += REWARDS["fill_storage"]

                if tmp_square == REP_TUP["filled_storage"]:
                    self.filled_storages -= 1
                    reward += REWARDS["unfill_storage"]
        
        return reward + old_manhattan - self.manhattan()



    def prev_states(self) -> Set[State]:
        previous_states = set()
        legal_moves = {(1, 0), (-1, 0), (0, 1), (0, -1)}

        for x, y in legal_moves:
            state_tmp = deepcopy(self)
            player_tmp = Player(self.player.x - x, self.player.y - y)

            if not (0 <= player_tmp.x < state_tmp.width) or not (
                0 <= player_tmp.y < state_tmp.height
            ):
                previous_states.add(((x, y), state_tmp))
                continue

            tmp_square = tuple(self.state[player_tmp.y][player_tmp.x])
            if tmp_square in {REP_TUP["empty"], REP_TUP["storage"]}:
                state_tmp.player = player_tmp
                state_tmp.state[player_tmp.y][player_tmp.x] += REP_NP["player"]
                state_tmp.state[self.player.y][self.player.x] -= REP_NP["player"]

                previous_states.add(((x, y), deepcopy(state_tmp)))
                # square in front of player in opposite direction of (reverse) movement
                square_in_front = tuple(
                    self.state[self.player.y + y][self.player.x + x]
                )
                if square_in_front[1] == 1:  # box
                    state_tmp.state[self.player.y + y][self.player.x + x] -= REP_NP[
                        "box"
                    ]
                    state_tmp.boxes.remove(Box(self.player.x + x, self.player.y + y))
                    if (
                        state_tmp.state[self.player.y + y][self.player.x + x][2] == 1
                    ):  # storage
                        state_tmp.filled_storages -= 1
                    state_tmp.state[self.player.y][self.player.x] += REP_NP["box"]
                    state_tmp.boxes.append(Box(self.player.x, self.player.y))
                    if state_tmp.state[self.player.y][self.player.x][2] == 1:  # storage
                        state_tmp.filled_storages += 1
                    previous_states.add(((x, y), state_tmp))

        return previous_states

    def possible_final_states(self) -> Set[State]:
        players = []
        filled_state_no_player = deepcopy(self)
        filled_state_no_player.boxes = [
            Box(s.x, s.y) for s in filled_state_no_player.storages
        ]
        for h in range(self.height):
            for w in range(self.width):
                square = tuple(self.state[h][w])
                if square in {REP_TUP["player"], REP_TUP["box"]}:
                    filled_state_no_player.state[h][w] = REP_TUP["empty"]
                elif square[2] == 1:
                    if square[1] == 0:
                        filled_state_no_player.state[h][w] = (0, 1, 1, 0)
                    for s in {(1, 0), (-1, 0), (0, 1), (0, -1)}:
                        no_player = {(0, 0, 0, 1), (0, 1, 1, 0)}
                        if (
                            0 <= h + 2 * s[0] < self.height
                            and 0 <= w + 2 * s[1] < self.width
                        ):
                            adj = tuple(self.state[h + s[0]][w + s[1]])
                            adj_2 = tuple(self.state[h + 2 * s[0]][w + 2 * s[1]])
                            if adj not in no_player and adj_2 not in no_player:
                                players.append(Player(w + s[1], h + s[0]))

        final_states = set()
        for player in players:

            temp_state = deepcopy(filled_state_no_player)
            temp_state.state[player.y][player.x] = np.array([1, 0, 0, 0])
            temp_state.player = player
            temp_state.filled_storages = temp_state.n_storages
            final_states.add(temp_state)
        return final_states
