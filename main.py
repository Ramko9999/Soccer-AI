import pygame
from enum import Enum
from dataclasses import dataclass
from drill import BluelockDrill, Team, Player, Ball


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
    TOGGLE = 3


PLAYER_RADIUS = 16
BALL_RADIUS = 12
TILT_INDICATOR_RADIUS = PLAYER_RADIUS + 4
GOAL_SIZE = 100
GOAL_LINE_X = 620


class BluelockDrillVisualization:
    def __init__(self, drill: BluelockDrill):
        pygame.init()
        self.drill = drill
        self.toggled_player_id = self.drill.get_players()[0].id
        self.font = pygame.font.SysFont(None, 16)
        self.screen = pygame.display.set_mode((640, 480))

    def toggle(self):
        player_ids = list(map(lambda player: player.id, self.drill.get_players()))
        next_id_index = (player_ids.index(self.toggled_player_id) + 1) % len(player_ids)
        self.toggled_player_id = player_ids[next_id_index]

    def draw_pitch(self):
        self.screen.fill(Color.PITCH)
        pygame.draw.line(
            self.screen,
            Color.WHITE,
            (GOAL_LINE_X, 0),
            (GOAL_LINE_X, (self.screen.get_height() - GOAL_SIZE) // 2),
            width=2,
        )
        pygame.draw.line(
            self.screen,
            Color.WHITE,
            (GOAL_LINE_X, (self.screen.get_height() + GOAL_SIZE) // 2),
            (GOAL_LINE_X, self.screen.get_height()),
            width=2,
        )

    def draw_player_tilt_indicator(self, player: Player):
        kit = team_to_kit[player.team]
        indicator_start_pos = player.tilt * (TILT_INDICATOR_RADIUS) + player.position
        indicator_end_pos = player.tilt * (TILT_INDICATOR_RADIUS + 4) + player.position
        pygame.draw.line(
            self.screen,
            kit.primary,
            tuple(indicator_start_pos),
            tuple(indicator_end_pos),
            width=3,
        )

    def draw_toggle_indicator(self, player: Player):

        pygame.draw.circle(
            self.screen, Color.TOGGLE, tuple(player.position), PLAYER_RADIUS + 2
        )

    def draw_player(self, player: Player):
        kit = team_to_kit[player.team]
        if player.id == self.toggled_player_id:
            self.draw_toggle_indicator(player)

        pygame.draw.circle(
            self.screen, kit.primary, tuple(player.position), PLAYER_RADIUS
        )
        kit_number = self.font.render(str(player.id), True, kit.number)
        pos_x, pos_y = tuple(player.position)
        # todo(Ramko9999): center the kit number within the player
        self.screen.blit(
            kit_number, (pos_x - PLAYER_RADIUS / 2, pos_y - PLAYER_RADIUS / 2)
        )
        self.draw_player_tilt_indicator(player)

    def draw_ball(self, ball: Ball):
        ball_position = ball.position
        if ball.is_possessed():
            # todo(Ramko9999): figure out how to decouple the ball's movement when possessed from the visualization
            ball_position = (
                ball.possessor.tilt * (TILT_INDICATOR_RADIUS + 2)
                + ball.possessor.position
            )

        pygame.draw.circle(
            self.screen, Color.OUTER_BALL, tuple(ball_position), BALL_RADIUS
        )
        pygame.draw.circle(
            self.screen, Color.INNER_BALL, tuple(ball_position), BALL_RADIUS - 2
        )

    def update(self, dt: int):
        self.draw_pitch()
        self.drill.update(dt)
        for player in self.drill.get_players():
            self.draw_player(player)

        self.draw_ball(self.drill.ball)
        pygame.display.flip()

    def on_external_event(self, event: PlayerEvent):
        player = self.drill.get_player(self.toggled_player_id)
        event_to_actions = {
            event.TURN_LEFT: player.turn_left,
            event.TURN_RIGHT: player.turn_right,
            event.RUN: player.run,
            event.TOGGLE: self.toggle,
        }
        event_to_actions[event]()


vis = BluelockDrillVisualization(BluelockDrill(640, 480, [9, 1], [21]))
movement_keys = set(["a", "d", "w"])
keys_pressed = set([])
is_running = True
clock = pygame.time.Clock()
while is_running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False
            continue
        elif event.type == pygame.KEYDOWN:
            if event.unicode in movement_keys:
                keys_pressed.add(event.unicode)
            elif event.unicode == "t":
                vis.on_external_event(PlayerEvent.TOGGLE)
        elif event.type == pygame.KEYUP:
            if event.unicode in movement_keys:
                keys_pressed.remove(event.unicode)

    for key in keys_pressed:
        if key == "a":
            vis.on_external_event(PlayerEvent.TURN_LEFT)
        elif key == "d":
            vis.on_external_event(PlayerEvent.TURN_RIGHT)
        elif key == "w":
            vis.on_external_event(PlayerEvent.RUN)

    dt = clock.tick(60)
    vis.update(dt)


pygame.quit()