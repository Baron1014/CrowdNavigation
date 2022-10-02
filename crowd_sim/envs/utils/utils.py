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


def getCloestEdgeDist(x1, y1, x2, y2, robot_hw, robot_hl):
    '''
    Parameters:
        x1, y1: point on robot edges. 
        robot_hw: half width of robot
        robot_hl: half length of robot

    if: distance from vertex
    elif: distance from right or left side
    elif: distance from top or bottom side
    else: inside of robot
    '''
    if abs(x1-x2) > robot_hw and abs(y1-y2) > robot_hl:
        # right
        if x1-x2 > 0:
            # if I else IV
            # min_dist = ((x1-robot_hw)-x2**2 + (y1-robot_hl)**2)**0.5 if y1-y2 > 0 else ((x1-robot_hw)**2 + (y1-(-robot_hl))**2)**0.5
            min_dist = np.linalg.norm(((x1-robot_hw)-x2, (y1-robot_hl)-y2)) if y1-y2 > 0 else np.linalg.norm(((x1-robot_hw)-x2, (y1+robot_hl)-y2))
        # left
        else:
            robot_hw = -robot_hw
            # if II else III
            min_dist = ((x1-robot_hw)**2 + (y1-robot_hl)**2)**0.5 if y1 > 0 else ((x1-robot_hw)**2 + (y1-(-robot_hl))**2)**0.5
    # right or left side
    elif abs(x1-x2) > robot_hw and abs(y1-y2) < robot_hl:
        min_dist = abs(x1-x2-robot_hw) if x1 > 0 else abs(x1-x2+robot_hw)
    # top or bottom side
    elif abs(x1-x2) < robot_hw and abs(y1-y2) > robot_hl:
        min_dist = abs(y1-y2-robot_hl) if y1 > 0 else abs(y1-y2+robot_hl)
    else:
        min_dist = 0

    return min_dist

# ax + by + c = 0
def checkonLinearEquationSide(x1, y1, x2, y2, x3, y3):
    sign = 1
    a = y2 - y1
    if a < 0:
        sign=-1
        a = sign*a
    b = sign*(x1-x2)
    c = sign*(y1*x2-x1*y2)

    # check point on side of line
    r = 0
    if a*x3 + b*y3 + c > 0:
        r = 1
    elif a*x3 + b*y3 + c < 0:
        r = -1

    return r