import math

def calc_dis(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def cmin(x, y, px, py, dis, direction, d):
    cur = calc_dis(x, y, px, py)
    if cur < dis:
        dis = cur
        direction = d
    return dis, direction

def calc(x, y, shape):
    """
    Calculate closest connection point on shape
    Shape categories for drawio:
    'rounded_rectangle': 0,
    'diamond': 1,
    'rectangle': 2,
    'circle': 3,
    'hexagon': 4,
    'parallelogram': 5,
    'text': 6,
    'arrow': 7,
    'line': 8
    """
    cls = shape[4]
    dis = 1e18
    direction = 0

    xmin, ymin, xmax, ymax = shape[0:4]
    
    # Calculate standard connection points (left, top, right, bottom)
    px, py = xmin, (ymin + ymax) / 2  # left
    dis, direction = cmin(x, y, px, py, dis, direction, 1)
    px, py = (xmin + xmax) / 2, ymin  # top
    dis, direction = cmin(x, y, px, py, dis, direction, 0)
    px, py = xmax, (ymin + ymax) / 2  # right
    dis, direction = cmin(x, y, px, py, dis, direction, 3)
    px, py = (xmin + xmax) / 2, ymax  # bottom
    dis, direction = cmin(x, y, px, py, dis, direction, 2)

    return dis, direction
