import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from itertools import combinations


class ReplayMemory(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()
        self.graph_memory = list()
        self.position = 0

    def push(self, item, graph=False):
        # replace old experience with new experience
        if len(self.memory) < self.position + 1:
            self.memory.append(item)
            if graph:
                snapshot = self.to_graph(item)
                self.graph_memory.append(snapshot)
        else:
            self.memory[self.position] = item
            if graph:
                snapshot = self.to_graph(item)
                self.graph_memory[self.position] = snapshot
        self.position = (self.position + 1) % self.capacity

    def is_full(self):
        return len(self.memory) == self.capacity

    def __getitem__(self, item):
        return self.memory[item]

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = list()

    def to_graph(self, item):
        robot_state, human_states, values, _, _, _ = item

        human_batch = len(human_states)
        r_edge = torch.norm(robot_state[:, 2:4], 2, keepdim=True) # velocity
        h_edge = torch.norm(human_states[:, 2:4], 2, dim=1, keepdim=True)
        edge_index = [[i, i] for i in range(human_batch+1)] # create temporal_edges
        edge_weight = torch.cat((torch.flatten(r_edge), torch.flatten(h_edge)), 0)
        # node feature
        r_feature = torch.cat((robot_state[:, :2], robot_state[:, 4:]), axis=1) # remove velocity 
        h_feature = torch.cat((human_states[:, :2], human_states[:, 4:]), axis=1)
        # expand human feature
        put_position = torch.tensor([7*i+j for i in range(human_batch) for j in range(r_feature.shape[1]) if j%7<3])
        h_feature = torch.zeros(h_feature.shape[0], r_feature.shape[1]).put_(put_position, h_feature)
        node_feature = torch.cat((r_feature, h_feature), axis=0)
        # spatial edge
        rh_weights = torch.norm(torch.cat([(human_states[:, 0] - robot_state[:, 0]).reshape((human_batch, -1)), (human_states[:, 1] - robot_state[:, 1]).
                                reshape((human_batch, -1))], dim=1), 2, dim=1, keepdim=True) # h_position - r_posiotion
        edge_index.extend([[0, i] for i in range(1, human_batch+1)]) # create rh_edges
        # hh_edge & weight
        hh_weights = torch.tensor([])
        for i, j in combinations([*range(human_batch)], 2):
            node_i, node_j = i+1, j+1
            edge_index.append([node_i, node_j]) # create hh_edges
            hh_w = torch.norm(human_states[i, 0:2] - human_states[j, 0:2], 2, keepdim=True) # h_position - h_posiotion
            hh_weights = torch.cat((hh_weights, hh_w), 0)

        edge_weight = torch.cat((edge_weight, torch.flatten(rh_weights), hh_weights), 0)
        target = values

        return Data(x=node_feature, y=target, edge_index=torch.tensor(edge_index).t().contiguous(), edge_attr=edge_weight)
