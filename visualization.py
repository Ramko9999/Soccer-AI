import pygame
from enum import Enum
from dataclasses import dataclass
from environment.core import Team, Player, Ball, Offender
from environment.drill import BluelockDrill
from util import get_euclidean_dist


class Color(tuple, Enum):
    PITCH = (21, 121, 32)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    TOGGLE = (173, 216, 230)
    OUTER_BALL = (75, 54, 95)
    INNER_BALL = (105, 60, 94)


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


FONT_SIZE = 16
PLAYER_TILT_SPEED = 0.1


class InteractiveBluelockDrillVisualization:
    def __init__(self, drill: BluelockDrill):
        pygame.init()
        self.drill = drill
        self.toggled_player_id = self.drill.get_players()[0].id
        self.font = pygame.font.SysFont(None, FONT_SIZE)
        self.screen = pygame.display.set_mode((640, 480))

    def draw_pitch(self):
        self.screen.fill(Color.PITCH)

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
        if player.has_possession():
            self.draw_ball(self.drill.get_ball(), possessor=player)
        else:
            self.draw_player_tilt_indicator(player)

    def draw_ball(self, ball: Ball, possessor: Player | None = None):
        ball_position = ball.position
        if possessor is not None:
            ball_position = possessor.tilt * (possessor.size + 6) + possessor.position

        pygame.draw.circle(
            self.screen, Color.OUTER_BALL, tuple(ball_position), ball.size
        )
        pygame.draw.circle(
            self.screen, Color.INNER_BALL, tuple(ball_position), ball.size - 2
        )

    def update(self, dt: int):
        self.draw_pitch()
        self.drill.update(dt)
        closest_dist_to_ball, closest_offender = float("inf"), None
        for player in self.drill.get_players():
            if type(player) is Offender:
                dist_to_ball = get_euclidean_dist(
                    player.position, self.drill.get_ball().position
                )
                if player.has_possession():
                    dist_to_ball = 0
                if dist_to_ball < closest_dist_to_ball:
                    closest_dist_to_ball, closest_offender = dist_to_ball, player

        if closest_offender is not None:
            self.toggled_player_id = closest_offender.id

        for player in self.drill.get_players():
            self.draw_player(player)

        if not self.drill.get_ball().is_possessed():
            self.draw_ball(self.drill.get_ball())
        pygame.display.flip()

    def on_external_event(self, event: PlayerEvent):
        player = self.drill.get_player(self.toggled_player_id)

        def turn_left():
            player.rotation -= PLAYER_TILT_SPEED

        def turn_right():
            player.rotation += PLAYER_TILT_SPEED

        event_to_actions = {
            event.TURN_LEFT: turn_left,
            event.TURN_RIGHT: turn_right,
            event.RUN: player.run,
            event.SHOOT: player.shoot,
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
                        self.on_external_event(PlayerEvent.SHOOT)
                elif event.type == pygame.KEYUP:
                    if event.unicode in key_to_movement:
                        keys_pressed.remove(event.unicode)
            for key in keys_pressed:
                self.on_external_event(key_to_movement[key])

            dt = clock.tick(60)
            self.update(dt)
        pygame.quit()
