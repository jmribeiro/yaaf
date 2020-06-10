from yaaf.agents import QLearningAgent


class SARSAAgent(QLearningAgent):

    def __init__(self, num_actions, learning_rate=0.3, discount_factor=0.99, exploration_rate=0.15, name="SARSA"):
        super(SARSAAgent, self).__init__(num_actions, learning_rate, discount_factor, exploration_rate, name=name)

    def _reinforce(self, timestep):

        s1 = tuple(timestep.observation)
        a = timestep.action
        r = timestep.reward
        s2 = tuple(timestep.next_observation)

        alpha = self._learning_rate
        gamma = self._discount_factor

        Q_s1_a = self._Q[s1][a]
        Q_s2_a = self._Q[s2][self.action(timestep.next_observation)]  # On-Policy update

        self._Q[s1][a] = Q_s1_a + alpha * (r + gamma * Q_s2_a - Q_s1_a)
