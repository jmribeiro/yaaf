from yaaf.policies import lazy_epsilon_greedy_policy, linear_annealing, greedy_policy


class EpsilonGreedyPolicy:

    def __init__(self,
                 num_actions: int,
                 initial_exploration_rate: float,
                 final_exploration_rate: float,
                 initial_exploration_steps: int,
                 final_exploration_step: int):

        self._initial_epsilon = initial_exploration_rate
        self._final_epsilon = final_exploration_rate
        self._initial_exploration_steps = initial_exploration_steps
        self._final_exploration_step = final_exploration_step

        self._num_actions = num_actions
        self._total_steps = 0

    def __call__(self, q_function, eval=False):
        if eval:
            return greedy_policy(q_function())
        else:
            epsilon = self.exploration_rate()
            self._total_steps += 1
            return lazy_epsilon_greedy_policy(q_function, self._num_actions, epsilon)

    def exploration_rate(self):
        if self._total_steps < self._initial_exploration_steps:
            return 1.0
        else:
            return linear_annealing(self._total_steps - self._initial_exploration_steps,
                                    self._final_exploration_step,
                                    self._initial_epsilon,
                                    self._final_epsilon)
