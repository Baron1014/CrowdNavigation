import torch
import numpy as np
import math
import networkx as nx
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import deque


class ReplayMemory(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()
        self.position = 0

    def push(self, item, graph=False):
        # replace old experience with new experience
        if len(self.memory) < self.position + 1:
            self.memory.append(item)
        else:
            self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def is_full(self):
        return len(self.memory) == self.capacity

    def __getitem__(self, item):
        return self.memory[item]

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = list()


class TemporalMemory(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, capacity, obs_len=4, pred_len=8, skip=1, threshold=0.002,
        min_ped=1, delim='\t'):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TemporalMemory, self).__init__()
        self.obs_len = obs_len

        self.capacity = capacity
        self.temporal_memory = {
            'robot':deque(maxlen=obs_len), 
            'humans': deque(maxlen=obs_len)
        }
        self.memory = deque(maxlen=capacity)

    def push_temporal(self, item):
        robot_state, human_states = item
        self.temporal_memory['robot'].append(robot_state)
        self.temporal_memory['humans'].append(human_states)
    
    def temporal_memory_fill(self):
        return True if len(self.temporal_memory['robot']) >= self.obs_len else False

    def push(self, item):
        self.memory.append(item)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, index):
        return self.memory[index]
