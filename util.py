import numpy as np
import math
import random
from dataclasses import dataclass


@dataclass
class Circle:
    center: np.ndarray
    radius: float


@dataclass
class Rect:
    top_left: list[float]
    height: float
    width: float

    def get_vertices(self):
        return [
            self.top_left[:],
            [self.top_left[0], self.top_left[1] + self.height],
            [self.top_left[0] + self.width, self.top_left[1]],
            [self.top_left[0] + self.width, self.top_left[1] + self.height],
        ]


def get_unit_vector(angle: float):
    return np.array([math.cos(angle), math.sin(angle)])


def get_beeline_orientation(vector: np.ndarray):
    return math.atan2(vector[1], vector[0])


def get_euclidean_dist(point1: np.ndarray, point2: np.ndarray):
    return math.sqrt(np.sum((point1 - point2) ** 2))


def can_circles_intersect(a: Circle, b: Circle):
    dist = get_euclidean_dist(a.center, b.center)
    return dist <= (a.radius + b.radius)


def get_random_within_range(max: float, min: float):
    range = max - min
    return random.random() * range + min


def get_random_point(x_max: float, y_max: float, x_min: float = 0, y_min: float = 0):
    return get_random_within_range(x_max, x_min), get_random_within_range(y_max, y_min)


def get_fscore(tp, fp, fn):
    precision = recall = 0
    if tp + fp > 0:
        precision = tp / (tp + fp)
    if tp + fn > 0:
        recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


# todo: remove this if this is not useful
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
