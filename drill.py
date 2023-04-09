import numpy as np
import random
from util import can_circles_intersect, Circle, Rect
from core import Offender, Defender, Ball, Player, BluelockDrillContext
from defense_policy import DefensePolicy


class BluelockDrill:
    def __init__(
        self,
        width: int,
        height: int,
        offensive_players: list[int],
        defense_players: list[int],
        defense_policy: DefensePolicy,
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
        self.__defense_policy = defense_policy

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

    def apply_defense_policy(self):
        assignments = self.__defense_policy(
            self.__defensive_players, self.__offensive_players, self.__ball
        )
        for def_id in assignments:
            angle, should_run = assignments[def_id]
            self.get_player(def_id).set_rotation(angle)
            if should_run:
                self.get_player(def_id).run()

    def does_defense_have_possession(self):
        for defender in self.__defensive_players:
            if defender.has_possession():
                return True
        return False

    def update(self, dt: int):
        context = BluelockDrillContext(
            bounds=Rect(
                top_left=np.array([0, 0]), height=self.__height, width=self.__width
            ),
        )

        self.apply_defense_policy()

        does_defense_have_possession = self.does_defense_have_possession()
        self.__ball.update(dt, context)
        for defensive_player in self.__defensive_players:
            defensive_player.update(dt, context)
            if not does_defense_have_possession and self.can_possess_ball(
                self.__ball, defensive_player
            ):
                defensive_player.possess(self.__ball)

        for offensive_player in self.__offensive_players:
            offensive_player.update(dt, context)
            if not self.__ball.is_possessed() and self.can_possess_ball(
                self.__ball, offensive_player
            ):
                offensive_player.possess(self.__ball)
