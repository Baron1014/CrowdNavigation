from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState, ObservableState

class Human(Agent):
    def __init__(self, _id, config, section, static=False):
        super().__init__(config, section)
        self.id = _id

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def get_id_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius, self.id)