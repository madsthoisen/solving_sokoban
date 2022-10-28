import pytest

import sys

from ..src.solving_sokoban.classes import Box, Player, Storage, Wall
from ..src.solving_sokoban.exceptions import IllegalStateException
from ..src.solving_sokoban.game_engine import State


@pytest.fixture
def state_ok() -> State:
    width = 6
    height = 6
    player = Player(1, 1)
    boxes = [Box(2, 2), Box(3, 3), Box(2, 4)]
    storages = [Storage(2, 2), Storage(3, 4), Storage(4, 3)]
    walls = [
        Wall(i, j) for i in range(6) for j in range(6) if i in {0, 5} or j in {0, 5}
    ]

    return State(width, height, player, boxes, storages, walls)


@pytest.fixture
def state_ok_moves():
    return [(0, 1), (0, 1), (1, 0), (1, 0), (-1, 0), (-1, 0), (0, 1), (1, 0)]


class TestStateOk:
    """
    Test that state prints, creates storages, and completes correct 
    right series of moves
    """

    def test_state_ok_prints_right(self, state_ok):
        out = state_ok.__str__()
        lev = "######\n#p   #\n# $  #\n#  BS#\n# BS #\n######\n"
        assert lev == out

    def test_state_ok_counting_storages(self, state_ok):
        assert state_ok.filled_storages == 1
        assert state_ok.n_storages == 3

    def test_state_ok_completes(self, state_ok, state_ok_moves):
        for move in state_ok_moves:
            state_ok.next_state(move)
        assert state_ok.filled_storages == 3
        assert state_ok.n_storages == 3


class TestStateFail:
    """
    Test that State instantiation fails on illegal states
    """

    def test_box_on_player(self):
        with pytest.raises(IllegalStateException):
            State(
                width=6,
                height=6,
                player=Player(1, 1),
                boxes=[Box(1, 1)],
                storages=[Storage(2, 2)],
                walls=[],
            )

    def test_box_on_wall(self):
        with pytest.raises(IllegalStateException):
            State(
                width=6,
                height=6,
                player=Player(1, 1),
                boxes=[Box(2, 2)],
                storages=[Storage(3, 3)],
                walls=[Wall(2, 2)],
            )

    def test_player_on_wall(self):
        with pytest.raises(IllegalStateException):
            State(
                width=6,
                height=6,
                player=Player(1, 1),
                boxes=[Box(2, 2)],
                storages=[Storage(3, 3)],
                walls=[Wall(1, 1)],
            )

    def test_storage_box_mismatch(self):
        with pytest.raises(IllegalStateException):
            State(
                width=6,
                height=6,
                player=Player(1, 1),
                boxes=[Box(2, 2), Box(3, 3)],
                storages=[Storage(2, 2)],
                walls=[Wall(1, 1)],
            )


@pytest.fixture
def state_no_walls() -> State:
    return State(
        width=6,
        height=4,
        player=Player(1, 1),
        boxes=[Box(2, 3)],
        storages=[Storage(2, 2)],
        walls=[],
    )


@pytest.fixture
def move_out_left():
    return [(-1, 0)] * 2


@pytest.fixture
def move_out_right():
    return [(1, 0)] * 5


@pytest.fixture
def move_out_up():
    return [(0, -1)] * 2


@pytest.fixture
def move_out_down():
    return [(0, 1)] * 3


class TestMoveOffScreen:
    """
    Test that player does not move when attempting to move off screen
    """

    def test_move_out_left(self, state_no_walls, move_out_left):
        for move in move_out_left:
            state_no_walls.next_state(move)
        assert state_no_walls.player == Player(0, 1)

    def test_move_out_right(self, state_no_walls, move_out_right):
        for move in move_out_right:
            state_no_walls.next_state(move)
        assert state_no_walls.player == Player(5, 1)

    def test_move_out_up(self, state_no_walls, move_out_up):
        for move in move_out_up:
            state_no_walls.next_state(move)
        assert state_no_walls.player == Player(1, 0)

    def test_move_out_down(self, state_no_walls, move_out_down):
        for move in move_out_down:
            state_no_walls.next_state(move)
        assert state_no_walls.player == Player(1, 3)


@pytest.fixture
def state_with_walls() -> State:
    return State(
        width=4,
        height=6,
        player=Player(1, 1),
        boxes=[Box(2, 2)],
        storages=[Storage(2, 3)],
        walls=[
            Wall(i, j) for i in range(4) for j in range(6) if i in {0, 3} or j in {0, 5}
        ],
    )


@pytest.fixture
def move_into_wall_left():
    return [(-1, 0)] * 3


@pytest.fixture
def move_into_wall_right():
    return [(1, 0)] * 4


@pytest.fixture
def move_into_wall_up():
    return [(0, -1)] * 4


@pytest.fixture
def move_into_wall_down():
    return [(0, 1)] * 6


class TestMoveIntoWall:
    """
    Test that player does not move when attempting to move into wall
    """

    def test_move_out_left(self, state_with_walls, move_into_wall_left):
        for move in move_into_wall_left:
            state_with_walls.next_state(move)
        assert state_with_walls.player == Player(1, 1)

    def test_move_out_right(self, state_with_walls, move_into_wall_right):
        for move in move_into_wall_right:
            state_with_walls.next_state(move)
        assert state_with_walls.player == Player(2, 1)

    def test_move_out_up(self, state_with_walls, move_into_wall_up):
        for move in move_into_wall_up:
            state_with_walls.next_state(move)
        assert state_with_walls.player == Player(1, 1)

    def test_move_out_down(self, state_with_walls, move_into_wall_down):
        for move in move_into_wall_down:
            state_with_walls.next_state(move)
        assert state_with_walls.player == Player(1, 4)


@pytest.fixture
def state_with_four_boxes() -> State:
    return State(
        width=8,
        height=8,
        player=Player(1, 1),
        boxes=[Box(2, 2), Box(3, 2), Box(2, 4), Box(2, 5)],
        storages=[Storage(3, 6), Storage(4, 6), Storage(5, 6), Storage(6, 6)],
        walls=[
            Wall(i, j) for i in range(8) for j in range(8) if i in {0, 7} or j in {0, 7}
        ],
    )


@pytest.fixture
def push_two_boxes_from_left():
    return [(0, 1), (1, 0), (1, 0)]


@pytest.fixture
def push_two_boxes_from_right():
    return [(1, 0), (1, 0), (1, 0), (0, 1), (-1, 0), (-1, 0)]


@pytest.fixture
def push_two_boxes_from_up():
    return [(0, 1), (0, 1), (1, 0), (0, 1), (0, 1)]


@pytest.fixture
def push_two_boxes_from_down():
    return [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (1, 0), (0, -1), (0, -1)]


class TestPushTwoBoxes:
    """
    Test that player does not move when attempting to push two boxes
    """

    def test_push_boxes_from_left(
        self, state_with_four_boxes, push_two_boxes_from_left
    ):
        for move in push_two_boxes_from_left:
            state_with_four_boxes.next_state(move)
        assert state_with_four_boxes.player == Player(1, 2)

    def test_push_boxes_from_right(
        self, state_with_four_boxes, push_two_boxes_from_right
    ):
        for move in push_two_boxes_from_right:
            state_with_four_boxes.next_state(move)
        assert state_with_four_boxes.player == Player(4, 2)

    def test_push_boxes_from_up(self, state_with_four_boxes, push_two_boxes_from_up):
        for move in push_two_boxes_from_up:
            state_with_four_boxes.next_state(move)
        assert state_with_four_boxes.player == Player(2, 3)

    def test_push_boxes_from_down(
        self, state_with_four_boxes, push_two_boxes_from_down
    ):
        for move in push_two_boxes_from_down:
            state_with_four_boxes.next_state(move)
        assert state_with_four_boxes.player == Player(2, 6)


@pytest.fixture
def state_with_no_boxes() -> State:
    return State(
        width=5,
        height=5,
        player=Player(2, 2),
        boxes=[],
        storages=[],
        walls=[
            Wall(i, j) for i in range(5) for j in range(5) if i in {0, 4} or j in {0, 4}
        ],
    )


@pytest.fixture
def state_with_box() -> State:
    return State(
        width=5,
        height=5,
        player=Player(2, 2),
        boxes=[Box(3, 2)],
        storages=[Storage(3, 3)],
        walls=[
            Wall(i, j) for i in range(5) for j in range(5) if i in {0, 4} or j in {0, 4}
        ],
    )


@pytest.fixture
def state_with_box_on_storage() -> State:
    return State(
        width=5,
        height=5,
        player=Player(2, 2),
        boxes=[Box(3, 2)],
        storages=[Storage(3, 2)],
        walls=[
            Wall(i, j) for i in range(5) for j in range(5) if i in {0, 4} or j in {0, 4}
        ],
    )


class TestReverseGameEngine:
    """
    Test the reverse game engine
    """

    def test_reverse_game_engine_no_box(self, state_with_no_boxes):
        previous_states = state_with_no_boxes.prev_states()
        previous_players = [state.player for move, state in previous_states]
        assert sorted(previous_players) == [
            Player(1, 2),
            Player(2, 1),
            Player(2, 3),
            Player(3, 2),
        ]

    def test_reverse_game_engine_with_box(self, state_with_box):
        previous_states = state_with_box.prev_states()
        previous_states = {state.__str__() for move, state in previous_states}
        assert previous_states == {
            "#####\n# p #\n#  B#\n#  S#\n#####\n",
            "#####\n#   #\n#p B#\n#  S#\n#####\n",
            "#####\n#   #\n#pB #\n#  S#\n#####\n",
            "#####\n#   #\n#  B#\n# pS#\n#####\n",
        }

    def test_reverse_game_engine_box_on_storage(self, state_with_box_on_storage):
        previous_states = state_with_box_on_storage.prev_states()
        previous_states = {state.__str__() for move, state in previous_states}
        assert previous_states == {
            "#####\n#   #\n#  $#\n# p #\n#####\n",
            "#####\n#   #\n#p $#\n#   #\n#####\n",
            "#####\n# p #\n#  $#\n#   #\n#####\n",
            "#####\n#   #\n#pBS#\n#   #\n#####\n",
        }
