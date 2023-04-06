import numpy as np
import math
import random
from enum import Enum

PLAYER_TILT_SPEED = 0.005
PLAYER_SPEED = 0.1


class Team(int, Enum):
    OFFEND = 0
    DEFEND = 1


class Player:
    def __init__(self, id: int, team: int, position: tuple[int]):
        self.__id = id
        self.__team = team
        self.__position = np.array(position, dtype=float)
        self.__rotation = 0  # rad
        self.__speed = 0
        self.__angular_speed = 0

    def turn_left(self):
        self.__angular_speed = -PLAYER_TILT_SPEED

    def turn_right(self):
        self.__angular_speed = PLAYER_TILT_SPEED

    def run(self):
        self.__speed = PLAYER_SPEED

    def update(self, dt: int):
        self.__rotation += self.__angular_speed * dt
        velocity = self.__speed * self.tilt
        self.__position += velocity * dt
        self.__speed = self.__angular_speed = 0

    @property
    def position(self):
        return self.__position

    @property
    def tilt(self):
        return np.array([math.cos(self.__rotation), math.sin(self.__rotation)])

    @property
    def id(self):
        return self.__id

    @property
    def team(self):
        return self.__team


class Offender(Player):
    def __init__(self, id: int, position: tuple[int]):
        super().__init__(id, Team.OFFEND, position)


class Defender(Player):
    def __init__(self, id: int, position: tuple[int]):
        super().__init__(id, Team.DEFEND, position)


BALL_FRICTION = 0.001


class Ball:
    def __init__(self, position: tuple[int]):
        self.__position = np.array(position)
        self.__speed = 0
        self.__direction = np.array([0, 0])
        self.__player_in_possession = None

    def possessed_by(self, player: Player):
        self.__player_in_possession = player

    def is_possessed(self):
        return self.__player_in_possession is not None

    def update(self, dt: int):
        if not self.is_possessed():
            velocity = self.__direction * (self.__speed * dt)
            self.__speed = max(0, self.__speed - BALL_FRICTION * dt)
            self.__position += velocity

    @property
    def possessor(self):
        return self.__player_in_possession

    @property
    def position(self):
        return self.__position


class Goal:
    def __init__(self, post1: tuple[int], post2: tuple[int]):
        pass


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
        self.__ball = Ball((random.random() * width, random.random() * height))

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

    def can_possess_ball(self, ball: Ball, player: Player):
        # todo(Ramko9999): figure the closest distance between 2 circles and use that instead
        distance_to_ball = math.sqrt(np.sum((player.position - ball.position) ** 2))
        return distance_to_ball < 30

    def update(self, dt: int):
        for defensive_player in self.__defensive_players:
            defensive_player.update(dt)

        for offensive_player in self.__offensive_players:
            offensive_player.update(dt)
            if not self.__ball.is_possessed() and self.can_possess_ball(
                self.ball, offensive_player
            ):
                self.ball.possessed_by(offensive_player)

        self.__ball.update(dt)

    @property
    def ball(self):
        return self.__ball
