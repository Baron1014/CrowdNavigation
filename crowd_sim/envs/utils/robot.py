from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState, JointState_noV
from collections import deque


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.obs_len = config.robot.obs_len
        self.ego_memory = {'ego':deque(maxlen=self.obs_len), 'humans':deque(maxlen=self.obs_len)} 

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')

        state = JointState(self.get_full_state(), ob)
        if self.policy.name=='SSTGCNN_RL':
            action = self.policy.predict(state, self.ego_memory)
        else:
            action = self.policy.predict(state)
        self.push_ego_memory(state)
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

    def push_ego_memory(self, state):
        if isinstance(state, JointState):
            robot_tensor, humans_tensor = state.to_id_tensor()
        else:
            robot_tensor, humans_tensor = state

        self.ego_memory['ego'].append(robot_tensor)
        self.ego_memory['humans'].append(humans_tensor)
    
    def ego_memory_fill(self):
        return True if len(self.ego_memory['ego']) >= self.obs_len else False
    
    def get_ego_jointstate(self):
        return self.ego_memory['ego'], self.ego_memory['humans']