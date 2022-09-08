import numpy as np


def point_to_segment_dist(x1, y1, x2, y2, x3, y3):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

    """
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return np.linalg.norm((x3-x1, y3-y1))

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return np.linalg.norm((x - x3, y-y3))

def point_to_clostest(x1, y1, x2, y2, x3, y3):
    """
    Calculate the closest distance between point(x3, y3) and a point segment with two endpoints (x1, y1), (x2, y2)

    """
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return [x3-x1, y3-y1]

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return [x, y]


def getCloestEdgeDist(x1, y1, x2, y2, robot_width, robot_length):
    if abs(x1-x2) > robot_width and abs(y1-y2) > robot_length:
        # right
        if x1 > 0:
            # if I else IV
            min_dist = ((x1-robot_width)**2 + (y1-robot_length)**2)**0.5 if y1 > 0 else ((x1-robot_width)**2 + (y1-(-robot_length))**2)**0.5
        # left
        else:
            robot_width = -robot_width
            # if II else III
            min_dist = ((x1-robot_width)**2 + (y1-robot_length)**2)**0.5 if y1 > 0 else ((x1-robot_width)**2 + (y1-(-robot_length))**2)**0.5
    # right or left side
    elif abs(x1-x2) > robot_width and abs(y1-y2) < robot_length:
        min_dist = x1-x2-robot_width if x1 > 0 else abs(x1-x2+robot_width)
    # top or bottom side
    elif abs(x1-x2) < robot_width and abs(y1-y2) > robot_length:
        min_dist = y1-y2-robot_length if y1 > 0 else abs(y1-y2+robot_length)
    else:
        min_dist = 0

    return min_dist

# ax + by + c = 0
def checkinLinearEquation(x1, y1, x2, y2, x3, y3):
    sign = 1
    a = y2 - y1
    if a < 0:
        sign=-1
        a = sign*a
    b = sign*(x1-x2)
    c = sign*(y1*x2-x1*y2)

    return True if a*x3 + b*y3 + c == 0 else False