import numpy as np
from environment.config import (
    BALL_FRICTION,
    BALL_SIZE,
    PLAYER_SIZE,
    PLAYER_SHOT_SPEED,
    PLAYER_DEFENDER_SPEED,
    PLAYER_OFFENDER_SPEED,
)
from enum import Enum
from util import get_unit_vector, can_circles_intersect, Circle


class Team(int, Enum):
    OFFEND = 0
    DEFEND = 1


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
            self.ball.hit(PLAYER_SHOT_SPEED)

    def possess(self, ball: "Ball"):
        ball.possessed_by(self)

    def update(self, dt: int):
        velocity = self.speed * self.tilt
        self.position = self.position + velocity * dt
        self.speed = self.angular_speed = 0

    def set_rotation(self, angle: float):
        self.rotation = angle

    def has_possession(self):
        return self.ball is not None

    @property
    def tilt(self):
        return get_unit_vector(self.rotation)


class Offender(Player):
    def __init__(self, id: int, position: tuple[int]):
        super().__init__(id, Team.OFFEND, position, top_speed=PLAYER_OFFENDER_SPEED)


class Defender(Player):
    def __init__(self, id: int, position: tuple[int]):
        super().__init__(id, Team.DEFEND, position, top_speed=PLAYER_DEFENDER_SPEED)


class Ball:
    def __init__(
        self,
        position: tuple[int],
        size=BALL_SIZE,
        friction: float = BALL_FRICTION,
    ):
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

    def update(self, dt: int):
        if not self.is_possessed():
            dir_vector = get_unit_vector(self.direction)
            velocity = dir_vector * (self.speed)
            self.speed = max(0, self.speed - self.friction * dt)
            self.position += velocity * dt

    def is_possessed(self):
        return self.possessor is not None


class BluelockEnvironment:
    def __init__(
        self,
        dims: tuple[int, int],
        offense: list[Offender],
        defense: list[Defender],
        ball: Ball,
    ):
        self.width, self.height = dims
        self.offense = offense
        self.defense = defense
        self.ball = ball

    def get_player(self, id: int):
        for offensive_player in self.offense:
            if offensive_player.id == id:
                return offensive_player
        for defensive_player in self.defense:
            if defensive_player.id == id:
                return defensive_player
        return None

    def get_players(self):
        return self.offense + self.defense

    def can_possess_ball(self, ball: Ball, player: Player):
        return can_circles_intersect(
            Circle(ball.position, ball.size), Circle(player.position, player.size)
        )

    def does_defense_have_possession(self):
        for defender in self.defense:
            if defender.has_possession():
                return True
        return False

    def clamp_players(self):
        for player in self.get_players():
            player.position[0] = max(
                min(player.position[0], self.width - player.size),
                player.size,
            )
            player.position[1] = max(
                min(player.position[1], self.height - player.size), player.size
            )

    def clamp_ball(self):
        self.ball.position[0] = max(
            min(self.ball.position[0], self.width - self.ball.size), self.ball.size
        )
        self.ball.position[1] = max(
            min(self.ball.position[1], self.height - self.ball.size), self.ball.size
        )

    def update(self, dt: int):
        does_defense_have_possession = self.does_defense_have_possession()
        self.ball.update(dt)
        for defensive_player in self.defense:
            defensive_player.update(dt)
            if not does_defense_have_possession and self.can_possess_ball(
                self.ball, defensive_player
            ):
                defensive_player.possess(self.ball)

        for offensive_player in self.offense:
            offensive_player.update(dt)
            if not self.ball.is_possessed() and self.can_possess_ball(
                self.ball, offensive_player
            ):
                offensive_player.possess(self.ball)
        self.clamp_players()
        self.clamp_ball()
