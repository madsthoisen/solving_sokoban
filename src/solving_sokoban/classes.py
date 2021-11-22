from dataclasses import dataclass
from abc import ABC


class Object(ABC):
    x: int
    y: int

    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)


@dataclass
class Box(Object):
    x: int
    y: int


@dataclass
class Player(Object):
    x: int
    y: int


@dataclass
class Storage(Object):
    x: int
    y: int


@dataclass
class Wall(Object):
    x: int
    y: int
