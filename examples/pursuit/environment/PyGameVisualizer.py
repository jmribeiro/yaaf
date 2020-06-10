import sys
from threading import Thread

import pygame


class PyGameVisualizer:

    def __init__(self, num_agents, width=400, height=400):

        agent_colors = [(200, 100, 255, 255)] + [(0, 0, 255, 255) for _ in range(num_agents - 1)]
        prey_colors = (255, 255, 0)

        assert ((isinstance(agent_colors, tuple) or isinstance(agent_colors, list)) and len(agent_colors) > 0)
        assert ((isinstance(prey_colors, tuple) or isinstance(prey_colors, list)) and len(prey_colors) > 0)

        self._black = (0, 0, 0)
        self._white = (255, 255, 255)

        self._size = self._width, self._height = width, height

        pygame.init()

        self._screen = pygame.display.set_mode(self._size)

        if isinstance(agent_colors[0], int): agent_colors = (agent_colors,)
        if isinstance(prey_colors[0], int): prey_colors = (prey_colors,)

        self._agent_colors = agent_colors
        self._prey_color = prey_colors
        self._state = None
        self._thread = None
        self._running = False

    def start(self, state):
        self._state = state
        self._running = True
        self._thread = Thread(target=self.draw, args=())
        self._thread.start()

    def update(self, next_state):
        self._state = next_state

    def end(self):
        self._running = False
        self._thread.join(timeout=1.0)

    def draw(self):

        while self._running:

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self._screen.fill(self._white)

            wx, wy = self._state.world_size
            xstep, ystep = self._width / wx, self._height / wy
            padding = 0.5  # value between 0 and 1

            for x in range(wx - 1):
                pygame.draw.line(self._screen, self._black, ((x + 1) * xstep, 0), ((x + 1) * xstep, self._height))

            for y in range(wy - 1):
                pygame.draw.line(self._screen, self._black, (0, (y + 1) * ystep), (self._width, (y + 1) * ystep))

            agents = self._state.agents_positions
            preys = self._state.prey_positions

            for i, (x, y) in enumerate(agents):
                color = self._agent_colors if len(self._agent_colors) == 1 else self._agent_colors[i]

                pygame.draw.rect(self._screen, color,
                                 pygame.Rect((x + padding / 2) * xstep, (y + padding / 2) * ystep,
                                             xstep - padding * xstep, ystep - padding * ystep))

            for i, (x, y) in enumerate(preys):
                loose = self._prey_color
                caught = (255, 0, 0)
                color = caught if self._state.cornered_position(self._state.prey_positions[0]) else loose
                pygame.draw.rect(self._screen, color,
                                 pygame.Rect((x + padding / 2) * xstep, (y + padding / 2) * ystep,
                                             xstep - padding * xstep, ystep - padding * ystep))

            pygame.display.flip()
