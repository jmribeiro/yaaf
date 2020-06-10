from yaaf.policies import linear_annealing, greedy_policy, boltzmann_policy


class BoltzmannPolicy:

    def __init__(self, initial_temperature: float,
                 final_temperature: float,
                 final_temperature_step: int):

        self._initial_tau = initial_temperature
        self._final_tau = final_temperature
        self._final_step = final_temperature_step

        self._total_steps = 0

    def __call__(self, q_function, eval=False):
        if eval:
            return greedy_policy(q_function())
        else:
            tau = self.temperature()
            self._total_steps += 1
            return boltzmann_policy(q_function(), tau)

    def temperature(self):
        return linear_annealing(self._total_steps, self._final_step, self._initial_tau, self._final_tau)
