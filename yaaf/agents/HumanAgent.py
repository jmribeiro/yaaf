from yaaf.agents import Agent


class HumanAgent(Agent):

    def __init__(self, action_meanings=None, num_actions=None, name="Human Agent", prompt="Enter an action"):
        super().__init__(name)

        if action_meanings is None:
            assert num_actions is not None, "When action_meanings not passed, num_actions required"
            self._num_actions = num_actions
        else:
            self._action_meanings = action_meanings
            self._num_actions = len(action_meanings)

        self._prompt = prompt

    def action(self, observation):

        invalid = True
        while invalid:

            print(f"{self._prompt}\n", end="")
            if len(self._action_meanings) > 0:
                [print(f"{a}: {meaning}\n", end="") for a, meaning in enumerate(self._action_meanings)]
            else:
                [print(f"{a}\n", end="") for a in range(self._num_actions)]

            try:
                action = int(input("> "))
                if action in tuple(range(self._num_actions)):
                    invalid = False
                else:
                    print(f"{action} is not a valid action")
            except Exception as e:
                print(f"Not a valid action")

        return action

    def policy(self, observation):
        pass

    def _reinforce(self, timestep):
        pass
