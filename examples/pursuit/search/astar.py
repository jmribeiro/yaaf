from heapq import *

from examples.pursuit.environment.utils import distance, neighbors, direction


class Node(object):
    def __init__(self, position, parent, cost, heuristic):
        self.position = position
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return self.cost + self.heuristic < other.cost + other.heuristic

    def __hash__(self):
        return self.position.__hash__()

    def __eq__(self, other):
        return self.position == other.position


def A_star_search(source, obstacles, target, world_size):

    """A* Search for Pursuit"""

    if source == target:
        return (0, 0), 0

    w, h = world_size
    obstacles = obstacles - {target}

    def heuristic(position):
        return sum(distance(source, position, w, h))

    # each item in the queue contains (heuristic+cost, cost, position, parent)
    initial_node = Node(source, None, 0, heuristic(source))
    queue = [Node(n, initial_node, 1, sum(distance(n, target, w, h)))
             for n in neighbors(source, world_size) if n not in obstacles]

    heapify(queue)
    visited = set()
    visited.add(source)
    current = initial_node

    while len(queue) > 0:
        current = heappop(queue)

        if current.position in visited:
            continue

        visited.add(current.position)

        if current.position == target:
            break

        for position in neighbors(current.position, world_size):
            if position not in obstacles:
                new_node = Node(position, current, current.cost + 1, heuristic(position))
                heappush(queue, new_node)

    if target not in visited:
        return None, w * h

    i = 1
    while current.parent != initial_node:
        current = current.parent
        i += 1

    return direction(source, current.position, h, w), i
