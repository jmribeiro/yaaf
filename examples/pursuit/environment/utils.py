import numpy as np

STAY = (0, 0)
RIGHT = (1, 0)
LEFT = (-1, 0)
DOWN = (0, 1)
UP = (0, -1)


def neighbors(position, world_size):
    directions = agent_directions()
    result = []
    for direction in directions: result.append(move(position, direction, world_size))
    return result


def direction(source, target, w, h):

    dx_forward = (target[0] - source[0]) % w
    dx_backward = (source[0] - target[0]) % w

    dy_forward = (target[1] - source[1]) % h
    dy_backward = (source[1] - target[1]) % h

    if dx_forward < dx_backward: return 1, 0
    elif dx_backward < dx_forward: return -1, 0
    elif dy_forward < dy_backward: return 0, 1
    elif dy_backward < dy_forward: return 0, -1
    else: return 0, 0


def direction_x(source, target, w):
    dx_forward = (target[0] - source[0]) % w
    dx_backward = (source[0] - target[0]) % w
    if dx_forward < dx_backward:
        return 1
    elif dx_backward < dx_forward:
        return -1
    else:
        return 0


def direction_y(source, target, h):
    dy_forward = (target[1] - source[1]) % h
    dy_backward = (source[1] - target[1]) % h
    if dy_forward < dy_backward:
        return 1
    elif dy_backward < dy_forward:
        return -1
    else:
        return 0


def distance(source, target, w, h):
    dx = min((source[0] - target[0]) % w, (target[0] - source[0]) % w)
    dy = min((source[1] - target[1]) % h, (target[1] - source[1]) % h)
    return dx, dy


def manhattan_distance(source, target, cols, rows):
    return sum(distance(source, target, cols, rows))


def move(entity_position, direction, world_size):

    w = world_size[0]
    h = world_size[1]

    x = entity_position[0]
    y = entity_position[1]

    dx = direction[0]
    dy = direction[1]

    new_position = (x + dx) % w, (y + dy) % h

    return new_position


def agent_directions():
    return RIGHT, LEFT, DOWN, UP


def prey_directions():
    return STAY, RIGHT, LEFT, DOWN, UP


def action_meanings():
    return "Right", "Left", "Down", "Up"


def action_meaning(action):
    return action_meanings()[action]


def argmin(arr):
    if len(arr) == 0:
        return None
    result = 0
    for i in range(len(arr)):
        if arr[i] < arr[result]:
            result = i
    return result


def argmax(arr):
    if len(arr) == 0: return None
    result = 0
    for i in range(len(arr)):
        if arr[i] > arr[result]:
            result = i
    return result


def softmax(array, factor=1.0):
    array = array * factor
    array = np.exp(array - np.max(array))
    return array / array.sum()
