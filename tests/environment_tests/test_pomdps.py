from unittest import TestCase

from yaaf.agents import RandomAgent
from yaaf.environments.mdp.pomdp import TigerProblemPOMDP
from yaaf.execution import TimestepRunner


class POMDPTests(TestCase):

    def test_tiger_problem_pomdp(self):
        pomdp = TigerProblemPOMDP(state_in_info=True)
        TimestepRunner(1000, RandomAgent(pomdp.num_actions), pomdp).run()
