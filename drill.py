import numpy as np
import random
from enum import Enum
from util import get_unit_vector, can_circles_intersect, Circle, Rect
from dataclasses import dataclass

PLAYER_TILT_SPEED = 0.005
PLAYER_SPEED = 0.1
SHOOT_SPEED = 0.5


class Team(int, Enum):
    OFFEND = 0
    DEFEND = 1


class Player:
    def __init__(self, id: int, team: int, position: tuple[int], size: int = 16):
        self.__id = id
        self.__team = team
        self.__size = size
        self.__position = np.array(position, dtype=float)
        self.__rotation = 0  # rad
        self.__speed = 0
        self.__angular_speed = 0
        self.__possessed_ball = None

    def turn_left(self):
        self.__angular_speed = -PLAYER_TILT_SPEED

    def turn_right(self):
        self.__angular_speed = PLAYER_TILT_SPEED

    def run(self):
        self.__speed = PLAYER_SPEED

    def shoot(self):
        if self.__possessed_ball is not None:
            self.__possessed_ball.hit(SHOOT_SPEED, self.__rotation, self.__position)
            self.__possessed_ball = None

    def possess(self, ball: "Ball"):
        ball.possess()
        self.__possessed_ball = ball

    def update(self, dt: int, context: "BluelockDrillContext"):
        self.__rotation += self.__angular_speed * dt
        velocity = self.__speed * self.tilt
        self.__position = self.__position + velocity * dt
        self.__speed = self.__angular_speed = 0

        # todo(ramko9999): more tightly restrict the player from leaving the bounds
        pos_x = max(
            min(self.__position[0], context.bounds.bottom_right[0] - self.size),
            context.bounds.top_left[0] + self.size,
        )
        pos_y = max(
            min(self.__position[1], context.bounds.bottom_right[1] - self.size),
            context.bounds.top_left[1] + self.size,
        )
        self.__position = np.array([pos_x, pos_y])

    @property
    def position(self):
        return self.__position

    @property
    def tilt(self):
        return get_unit_vector(self.__rotation)

    @property
    def id(self):
        return self.__id

    @property
    def team(self):
        return self.__team

    @property
    def size(self):
        return self.__size

    def has_possession(self, ball: "Ball"):
        return self.__possessed_ball is not None and self.__possessed_ball.id == ball.id


class Offender(Player):
    def __init__(self, id: int, position: tuple[int]):
        super().__init__(id, Team.OFFEND, position)


class Defender(Player):
    def __init__(self, id: int, position: tuple[int]):
        super().__init__(id, Team.DEFEND, position)


BALL_FRICTION = 0.0005


class Ball:
    def __init__(self, id: int, position: tuple[int], size=12):
        self.__id = id
        self.__position = np.array(position)
        self.__speed = 0
        self.__direction = 0  # rad
        self.__is_possessed = False
        self.__size = size

    def possess(self):
        self.__is_possessed = True
        self.__speed = 0
        self.__direction = 0

    def hit(self, speed: float, direction: float, origin: np.ndarray):
        self.__is_possessed = False
        self.__speed = speed
        self.__direction = direction
        self.__position = origin + (get_unit_vector(self.__direction) * 26)

    def is_possessed(self):
        return self.__is_possessed

    def update(self, dt: int, context: "BluelockDrillContext"):
        if not self.is_possessed():
            dir_vector = get_unit_vector(self.__direction)
            velocity = dir_vector * (self.__speed)
            self.__speed = max(0, self.__speed - BALL_FRICTION * dt)
            self.__position += velocity * dt

    @property
    def position(self):
        return self.__position

    @property
    def id(self):
        return self.__id

    @property
    def size(self):
        return self.__size


@dataclass
class BluelockDrillContext:
    bounds: Rect


class BluelockDrill:
    def __init__(
        self,
        width: int,
        height: int,
        offensive_players: list[int],
        defense_players: list[int],
    ):
        self.__width = width
        self.__height = height
        self.__offensive_players = [
            Offender(id, (random.random() * width, random.random() * height))
            for id in offensive_players
        ]
        self.__defensive_players = [
            Defender(id, (random.random() * width, random.random() * height))
            for id in defense_players
        ]
        self.__ball = Ball(1, (random.random() * width, random.random() * height))

    def get_player(self, id: int):
        for offensive_player in self.__offensive_players:
            if offensive_player.id == id:
                return offensive_player
        for defensive_player in self.__defensive_players:
            if defensive_player.id == id:
                return defensive_player
        return None

    def get_players(self):
        return self.__offensive_players + self.__defensive_players

    def get_ball(self):
        return self.__ball

    def can_possess_ball(self, ball: Ball, player: Player):
        return can_circles_intersect(
            Circle(ball.position, ball.size), Circle(player.position, player.size)
        )

    def update(self, dt: int):
        context = BluelockDrillContext(
            bounds=Rect(
                top_left=np.array([0, 0]), height=self.__height, width=self.__width
            ),
        )

        self.__ball.update(dt, context)
        for defensive_player in self.__defensive_players:
            defensive_player.update(dt, context)

        for offensive_player in self.__offensive_players:
            offensive_player.update(dt, context)
            if not self.__ball.is_possessed() and self.can_possess_ball(
                self.__ball, offensive_player
            ):
                offensive_player.possess(self.__ball)
