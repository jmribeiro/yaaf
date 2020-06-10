from collections import defaultdict
from multiprocessing import RLock

import matplotlib.pyplot as plt
import numpy as np


class LinePlot:

    def __init__(self, title,
                 x_label, y_label,
                 num_measurements, x_tick_step=1,
                 confidence=0.95,
                 ymin=None, ymax=None, colors=None):

        self._title = title

        from yaaf.visualization import RunStats
        self._stats = defaultdict(lambda: RunStats(num_measurements, confidence))
        self._colors = {} if colors is None else colors
        self._markers = {}

        self._legend = {}

        self._x_label = x_label
        self._y_label = y_label

        self._x_tick_step = x_tick_step
        self._x_ticks = num_measurements

        self._current_figure = None
        self._highest_y_value = 1

        self._lock = RLock()

        self._ymin = ymin
        self._ymax = ymax

    @property
    def has_runs(self):
        return self.num_runs() > 0

    def num_runs(self, agent_name=None):
        if agent_name is not None:
            stats = self._stats[agent_name]
            return stats.N
        else:
            total_runs = 0
            for agent in self._stats:
                total_runs += self.num_runs(agent)
            return total_runs

    def add_run(self, agent_name, run, color=None, add_to_legend=True, marker="o"):

        if color is not None:
            self._colors[agent_name] = color

        self._markers[agent_name] = marker
        self._legend[agent_name] = add_to_legend

        with self._lock:

            # First measurement added for agent, create color
            if agent_name not in self._colors:
                color = self._random_color(list(self._colors.values()))
                self._colors[agent_name] = color

            if len(run) < self._x_ticks:
                padded_run = np.ones((self._x_ticks,)) * run[-1]
                padded_run[:run.shape[0]] = run
                run = padded_run

            stats = self._stats[agent_name]
            stats.add(run)

            self._highest_y_value = max(int(run.max()) + 1, self._highest_y_value)

    def show(self, error_fill=True, error_fill_transparency=0.25):
        if len(self._stats) > 0:
            if self._current_figure is not None: plt.close(self._current_figure)
            self._make_fig(error_fill, error_fill_transparency)
            self._current_figure.show()
            plt.close(self._current_figure)
            del self._current_figure
            self._current_figure = None

    def savefig(self, filename=None, error_fill=True, error_fill_transparency=0.25):
        if len(self._stats) > 0:
            if filename is None: filename = self._title
            if self._current_figure is not None: plt.close(self._current_figure)
            self._make_fig(error_fill, error_fill_transparency)
            self._current_figure.savefig(filename)
            plt.close(self._current_figure)
            del self._current_figure
            self._current_figure = None

    def save(self, directory):
        # 1 - Save Metadata
        # 2 - Save env models evaluation per agent (numpy matrix)
        pass

    def load(self, directory):
        # 1 - Load Metadata
        # 2 - Load env models evaluation per agent (numpy matrix)
        pass

    def _make_fig(self, error_fill=True, error_fill_transparency=0.25, show_legend=True):

        x_ticks = (np.arange(self._x_ticks) + 1) * self._x_tick_step
        self._current_figure, ax = plt.subplots(1)

        for agent_name, stats in self._stats.items():

            num_runs = stats.N
            means = stats.means
            errors = stats.errors

            color = self._colors[agent_name]
            marker = self._markers[agent_name]

            if self._legend[agent_name]:
                ax.plot(x_ticks, means, lw=2, label=f"{agent_name} (N={num_runs})", color=color, marker=marker)
            else:
                ax.plot(x_ticks, means, lw=2, color=color, marker=marker)

            if error_fill:
                ax.fill_between(x_ticks, means + errors, means - errors, facecolor=color, alpha=error_fill_transparency)

        ax.set_title(self._title)
        ax.set_xlabel(self._x_label)
        ax.set_ylabel(self._y_label)

        if self._ymin is not None:
            ax.set_ylim(bottom=self._ymin)
        else:
            ax.set_ylim(top=self._highest_y_value)

        if self._ymax is not None:
            ax.set_ylim(top=self._ymax)

        if show_legend:
            #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fancybox=True, shadow=True)
            ax.legend()

        ax.grid()

    @staticmethod
    def _random_color(excluded_colors):
        excluded_colors = excluded_colors or []
        color_map = plt.get_cmap('gist_rainbow')
        if len(excluded_colors) == 0:
            color = color_map(np.random.uniform())
        else:
            color = excluded_colors[0]
            while color in excluded_colors:
                color = color_map(np.random.uniform())
        return color

