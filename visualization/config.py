from enum import Enum


class Color(tuple, Enum):
    PITCH = (21, 121, 32)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    TOGGLE = (173, 216, 230)
    OUTER_BALL = (75, 54, 95)
    INNER_BALL = (105, 60, 94)


KIT_NUMBER_FONT_SIZE = 16
VISUALIZATION_PLAYER_TILT_SPEED = 0.1
