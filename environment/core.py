import numpy as np
from enum import Enum
from util import get_unit_vector, Rect
from dataclasses import dataclass

OFFENDER_SPEED = 0.1
DEFENDER_SPEED = 0.08
SHOOT_SPEED = 0.5

PLAYER_SIZE = 16
BALL_SIZE = 12


class Team(int, Enum):
    OFFEND = 0
    DEFEND = 1


# todo: remove BluelockDrillContext


class Player:
    def __init__(
        self,
        id: int,
        team: int,
        position: tuple[int],
        top_speed: float,
        size: int = PLAYER_SIZE,
    ):
        self.id = id
        self.team = team
        self.size = size
        self.top_speed = top_speed
        self.position = np.array(position, dtype=float)
        self.rotation = 0  # rad
        self.speed = 0
        self.ball = None

    def run(self):
        self.speed = self.top_speed

    def shoot(self):
        if self.ball is not None:
            self.ball.hit(SHOOT_SPEED)

    def possess(self, ball: "Ball"):
        ball.possessed_by(self)

    def update(self, dt: int, context: "BluelockDrillContext"):
        velocity = self.speed * self.tilt
        self.position = self.position + velocity * dt
        self.speed = self.angular_speed = 0

        # todo(ramko9999): more tightly restrict the player from leaving the bounds
        pos_x = max(
            min(self.position[0], context.bounds.bottom_right[0] - self.size),
            context.bounds.top_left[0] + self.size,
        )
        pos_y = max(
            min(self.position[1], context.bounds.bottom_right[1] - self.size),
            context.bounds.top_left[1] + self.size,
        )
        self.position = np.array([pos_x, pos_y])

    def set_rotation(self, angle: float):
        self.rotation = angle

    def has_possession(self):
        return self.ball is not None

    @property
    def tilt(self):
        return get_unit_vector(self.rotation)


class Offender(Player):
    def __init__(self, id: int, position: tuple[int]):
        super().__init__(id, Team.OFFEND, position, top_speed=OFFENDER_SPEED)


class Defender(Player):
    def __init__(self, id: int, position: tuple[int]):
        super().__init__(id, Team.DEFEND, position, top_speed=DEFENDER_SPEED)


BALL_FRICTION = 0.0005


class Ball:
    def __init__(
        self,
        id: int,
        position: tuple[int],
        size=BALL_SIZE,
        friction: float = BALL_FRICTION,
    ):
        self.id = id
        self.position = np.array(position)
        self.speed = 0
        self.friction = friction
        self.direction = 0
        self.possessor: Player | None = None
        self.size = size

    def detach_from_possessor(self):
        if self.possessor is not None:
            self.possessor.ball = None
            self.possessor = None

    def possessed_by(self, player: Player):
        self.detach_from_possessor()
        self.possessor = player
        self.possessor.ball = self
        self.speed = 0
        self.direction = 0

    def hit(self, speed: float):
        origin = self.possessor.position
        self.direction = self.possessor.rotation
        self.detach_from_possessor()
        self.speed = speed
        self.position = origin + (get_unit_vector(self.direction) * (PLAYER_SIZE * 2))

    def update(self, dt: int, context: "BluelockDrillContext"):
        if not self.is_possessed():
            dir_vector = get_unit_vector(self.direction)
            velocity = dir_vector * (self.speed)
            self.speed = max(0, self.speed - self.friction * dt)
            self.position += velocity * dt

    def is_possessed(self):
        return self.possessor is not None


@dataclass
class BluelockDrillContext:
    bounds: Rect
