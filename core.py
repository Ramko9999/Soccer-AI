import numpy as np
from enum import Enum
from util import get_unit_vector, Rect
from dataclasses import dataclass

OFFENDER_SPEED = 0.1
DEFENDER_SPEED = 0.08
SHOOT_SPEED = 0.5


class Team(int, Enum):
    OFFEND = 0
    DEFEND = 1


class Player:
    def __init__(
        self, id: int, team: int, position: tuple[int], top_speed: float, size: int = 16
    ):
        self.id = id
        self.team = team
        self.size = size
        self.top_speed = top_speed
        self.position = np.array(position, dtype=float)
        self.rotation = 0  # rad
        self.speed = 0
        self.possessed_ball = None

    def run(self):
        self.speed = self.top_speed

    def shoot(self):
        if self.possessed_ball is not None:
            self.possessed_ball.hit(SHOOT_SPEED, self.rotation, self.position)
            self.possessed_ball = None

    def possess(self, ball: "Ball"):
        ball.possess()
        self.possessed_ball = ball

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

    def has_possession(self, ball: "Ball"):
        return self.possessed_ball is not None and self.possessed_ball.id == ball.id

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
        self, id: int, position: tuple[int], size=12, friction: float = BALL_FRICTION
    ):
        self.id = id
        self.position = np.array(position)
        self.speed = 0
        self.friction = friction
        self.direction = 0  # rad
        self.is_possessed = False
        self.size = size

    def possess(self):
        self.is_possessed = True
        self.speed = 0
        self.direction = 0

    def hit(self, speed: float, direction: float, origin: np.ndarray):
        self.is_possessed = False
        self.speed = speed
        self.direction = direction
        self.position = origin + (get_unit_vector(self.direction) * 26)

    def update(self, dt: int, context: "BluelockDrillContext"):
        if not self.is_possessed:
            dir_vector = get_unit_vector(self.direction)
            velocity = dir_vector * (self.speed)
            self.speed = max(0, self.speed - self.friction * dt)
            self.position += velocity * dt


@dataclass
class BluelockDrillContext:
    bounds: Rect
