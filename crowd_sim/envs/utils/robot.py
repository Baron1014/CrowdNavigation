from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState, JointState_noV
from collections import deque


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.ego_memory = deque(maxlen=config.robot.obs_len-2)

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')

        state = JointState(self.get_full_state(), ob)
        if self.policy.name=='SSTGCNN_RL':
            action = self.policy.predict(state, self.ego_memory)
        else:
            action = self.policy.predict(state)
        self.ego_memory.append(state)
        return action

    def act_noV(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState_noV(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def actWithJointState(self,ob):
        action = self.policy.predict(ob)
        return action

