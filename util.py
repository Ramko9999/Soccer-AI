import numpy as np
import math
from dataclasses import dataclass


@dataclass
class Circle:
    center: np.ndarray
    radius: float


@dataclass
class Rect:
    top_left: np.ndarray
    height: float
    width: float

    @property
    def bottom_right(self):
        return self.top_left + np.array([self.width, self.height])

    def get_vertices(self):
        return [
            self.top_left,
            np.ndarray([self.top_left[0], self.bottom_right[1]]),
            np.ndarray([self.bottom_right[0], self.top_left[1]]),
            self.bottom_right,
        ]


def get_unit_vector(angle: float):
    return np.array([math.cos(angle), math.sin(angle)])


def get_euclidean_dist(point1: np.ndarray, point2: np.ndarray):
    return math.sqrt(np.sum((point1 - point2) ** 2))


def can_circles_intersect(a: Circle, b: Circle):
    dist = get_euclidean_dist(a.center, b.center)
    return dist <= (a.radius + b.radius)


def can_circle_rect_intersect(a: Circle, b: Rect):
    is_center_x_in_bounds = (
        a.center[0] >= b.top_left[0] and a.center[0] <= b.bottom_right[0]
    )
    is_center_y_in_bounds = (
        a.center[1] >= b.top_left[1] and a.center[1] <= b.bottom_right[1]
    )
    if is_center_x_in_bounds and is_center_y_in_bounds:
        return True

    if is_center_x_in_bounds:
        y_dist_to_rect = min(
            abs(a.center[1] - b.top_left[1]), abs(a.center[1] - b.bottom_right[1])
        )
        return y_dist_to_rect < a.radius

    if is_center_y_in_bounds:
        x_dist_to_rect = min(
            abs(a.center[0] - b.top_left[0]), abs(a.center[0] - b.bottom_right[0])
        )
        return x_dist_to_rect < a.radius

    for vertex in b.get_vertices():
        if get_euclidean_dist(vertex, a.center) <= a.radius:
            return True
    return False
