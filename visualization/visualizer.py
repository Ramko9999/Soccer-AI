import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame
from enum import Enum
from dataclasses import dataclass
from environment.core import Team, Player, Ball, BluelockEnvironment
from visualization.config import (
    Color,
    KIT_NUMBER_FONT_SIZE,
    VISUALIZATION_PLAYER_TILT_SPEED,
)


@dataclass
class Kit:
    primary: Color
    number: Color


team_to_kit = {
    Team.OFFEND: Kit(Color.WHITE, Color.BLACK),
    Team.DEFEND: Kit(Color.BLACK, Color.WHITE),
}


class PlayerEvent(int, Enum):
    TURN_LEFT = 0
    TURN_RIGHT = 1
    RUN = 2
    SHOOT = 3
    TOGGLE = 4


class BluelockEnvironmentVisualizer:
    def __init__(self, env: BluelockEnvironment):
        pygame.init()
        self.env = env
        self.toggled_player_id = self.env.offense[0].id
        self.font = pygame.font.SysFont(None, KIT_NUMBER_FONT_SIZE)
        self.screen = pygame.display.set_mode((env.width, env.height))

    def draw_pitch(self):
        self.screen.fill(Color.PITCH)
        elapsed_indicator = self.font.render(
            f"{self.env.simulation_time}", True, Color.WHITE
        )
        self.screen.blit(elapsed_indicator, (self.env.width // 2, self.env.height // 2))

    def draw_player_tilt_indicator(self, player: Player):
        kit = team_to_kit[player.team]
        indicator_start_pos = player.tilt * (player.size + 4) + player.position
        indicator_end_pos = player.tilt * (player.size + 8) + player.position
        pygame.draw.line(
            self.screen,
            kit.primary,
            tuple(indicator_start_pos),
            tuple(indicator_end_pos),
            width=3,
        )

    def draw_toggle_indicator(self, player: Player):
        pygame.draw.circle(
            self.screen, Color.TOGGLE, tuple(player.position), player.size + 2
        )

    def draw_player(self, player: Player):
        kit = team_to_kit[player.team]
        if player.id == self.toggled_player_id:
            self.draw_toggle_indicator(player)

        pygame.draw.circle(
            self.screen, kit.primary, tuple(player.position), player.size
        )
        kit_number = self.font.render(str(player.id), True, kit.number)
        pos_x, pos_y = tuple(player.position)

        # todo(Ramko9999): center the kit number within the player
        self.screen.blit(kit_number, (pos_x - player.size / 2, pos_y - player.size / 2))
        self.draw_player_tilt_indicator(player)

    def draw_ball(self, ball: Ball):
        ball_position = ball.position
        if ball.possessor is not None:
            ball_position = (
                ball.possessor.tilt * (ball.possessor.size + 6)
                + ball.possessor.position
            )

        pygame.draw.circle(
            self.screen, Color.OUTER_BALL, tuple(ball_position), ball.size
        )
        pygame.draw.circle(
            self.screen, Color.INNER_BALL, tuple(ball_position), ball.size - 2
        )

    def draw(self):
        self.draw_pitch()
        for player in self.env.get_players():
            self.draw_player(player)

        self.draw_ball(self.env.ball)
        pygame.display.flip()

    def update(self, dt: int):
        self.env.update(dt)
        self.draw()

    def on_event(self, event: PlayerEvent):
        player = self.env.get_player(self.toggled_player_id)

        def turn_left():
            player.rotation -= VISUALIZATION_PLAYER_TILT_SPEED

        def turn_right():
            player.rotation += VISUALIZATION_PLAYER_TILT_SPEED

        def toggle():
            offense_ids = sorted([offender.id for offender in self.env.offense])
            toggle_index = offense_ids.index(self.toggled_player_id)
            self.toggled_player_id = offense_ids[(toggle_index + 1) % len(offense_ids)]

        event_to_actions = {
            event.TURN_LEFT: turn_left,
            event.TURN_RIGHT: turn_right,
            event.RUN: player.run,
            event.SHOOT: player.shoot,
            event.TOGGLE: toggle,
        }
        event_to_actions[event]()

    def start(self):
        key_to_movement = {
            "a": PlayerEvent.TURN_LEFT,
            "d": PlayerEvent.TURN_RIGHT,
            "w": PlayerEvent.RUN,
        }
        keys_pressed = set([])
        is_running = True
        clock = pygame.time.Clock()
        while is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_running = False
                    continue
                elif event.type == pygame.KEYDOWN:
                    if event.unicode in key_to_movement:
                        keys_pressed.add(event.unicode)
                    elif event.unicode == "s":
                        self.on_event(PlayerEvent.SHOOT)
                    elif event.unicode == "t":
                        self.on_event(PlayerEvent.TOGGLE)
                elif event.type == pygame.KEYUP:
                    if event.unicode in key_to_movement:
                        keys_pressed.remove(event.unicode)
            for key in keys_pressed:
                self.on_event(key_to_movement[key])

            dt = clock.tick(60)
            self.update(dt)
        pygame.quit()
