from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState, RobotState


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.width = getattr(config, section).width
        self.length = getattr(config, section).length

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')

        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def get_full_state(self):
        # return FullState(self.px, self.py, self.vx, self.vy, self.gx, self.gy, self.v_pref, radius=0.3, theta=3.1415/2)
        return RobotState(self.px, self.py, self.vx, self.vy, self.gx, self.gy, self.v_pref, self.theta, robot_size=(self.length, self.width))
