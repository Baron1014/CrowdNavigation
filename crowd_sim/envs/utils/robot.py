import numpy as np
from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from collections import deque
import itertools
import collections


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.obs_len = config.robot.obs_len
        self.ego_memory = {'ego':sliceable_deque(maxlen=self.obs_len), 'humans':sliceable_deque(maxlen=self.obs_len)} 

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')

        state = JointState(self.get_full_state(), ob)
        if self.policy.name=='SSTGCNN_RL':
            if len(self.ego_memory['ego'])<(self.obs_len-2):
                for _ in range(self.obs_len-2):
                    self.push_ego_memory(state)
            action = self.policy.predict(state, self.ego_memory)
            self.push_ego_memory(state)
        else:
            action = self.policy.predict(state)
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
    
    def clean_ego_memory(self):
        self.ego_memory = {'ego':sliceable_deque(maxlen=self.obs_len), 'humans':sliceable_deque(maxlen=self.obs_len)} 

    def set_fov(self, fov):
        self.FoV = np.pi * fov if fov is not None else None
        if self.FoV is not None:
            self.sensor = 'RGBD'

    def get_fov_degree(self):
        return int(self.FoV*180/np.pi)

class sliceable_deque(deque):
    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(itertools.islice(self, index.start,
                                               index.stop, index.step))
        return collections.deque.__getitem__(self, index)
