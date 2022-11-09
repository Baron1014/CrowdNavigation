from crowd_sim.envs.policy.policy import Policy
from torch import nn
from torch_geometric.nn import RGCNConv, GCNConv
from torch_geometric_temporal.nn import TemporalConv
from torch_geometric_temporal.nn.recurrent import DCRNN, AGCRN
import networkx as nx
from torch_geometric.utils.convert import from_networkx
# from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.model_predictive_rl import ModelPredictiveRL
from crowd_nav.policy.cadrl import CADRL
from crowd_sim.envs.utils.action import ActionRot, ActionXY
import numpy as np
from crowd_nav.policy.state_predictor import StatePredictor
from crowd_nav.policy.value_estimator import GraphValueEstimator
from itertools import permutations
import torch
from crowd_nav.policy.helpers import mlp
from torch.nn.functional import relu
import numba

class ValueNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, node_feature, edge_index, edge_weight):
        node_feature = torch.tensor(node_feature, dtype=torch.float)
        node_feature = self.conv1(node_feature, edge_index, edge_weight)
        output = self.conv2(node_feature, edge_index, edge_weight)
        print(output)



class DGCRNN(nn.Module):
    def __init__(self, config, robot_state_dim, human_state_dim):
        super().__init__()
        wr_dims = config.gcn.wr_dims
        wh_dims = config.gcn.wh_dims

        self.X_dim = config.gcn.X_dim
        self.num_layer = config.gcn.num_layer
        self.skip_connection = config.gcn.skip_connection
        self.gcn = GCNConv(16, 16)
        self.dcrnn = DCRNN(self.X_dim, self.X_dim, 3)
        # self.tcn = TemporalConv(self.X_dim, 16, kernel_size=4)

        self.w_r = mlp(robot_state_dim, wr_dims, last_relu=True)
        self.w_h = mlp(human_state_dim, wh_dims, last_relu=True)

    def forward(self, graphs):
        """
        Embed current state tensor pair (robot_state, human_states) into a latent space
        Each tensor is of shape (batch_size, # of agent, features)
        :param state:
        :return:
        """
        graph_embedding = []
        for graph in graphs:
            robot_state_embedings = self.w_r(graph['robot'].x)
            human_state_embedings = self.w_h(graph['human'].x)
            X = torch.cat([robot_state_embedings, human_state_embedings], dim=0)

            edge_index = torch.tensor([])
            edge_attr = torch.tensor([])
            for edge in graph.edge_types:
                edge_index = torch.cat([edge_index, graph[edge]['edge_index']], 1)
                edge_attr = torch.cat([edge_attr, graph[edge]['edge_weight']], 0)
            edge_index = edge_index.to(torch.int64)

            dcrnn_out = self.dcrnn(X, edge_index, edge_attr)
            # next_H = H = x
            # for j in range(self.num_layer):
            #     next_H = relu(self.gcn(dcrnn_out, edge_index, edge_attr))

            #     if self.skip_connection:
            #         next_H = next_H.clone() + H
            #     H = next_H
            graph_embedding.append(dcrnn_out[0, :])
            # next_H = self.dcrnn(X[i, :, :], edge_index, edge_attr)
        return torch.stack(graph_embedding)

    

class DGCNRL(ModelPredictiveRL):
    def __init__(self):
        super().__init__()
        self.name = 'DGCNRL'
        self.robot_state_dim = 9
        self.human_state_dim = 5
    
    def configure(self, config):
        self.set_common_parameters(config)
        self.planning_depth = config.model_predictive_rl.planning_depth
        self.do_action_clip = config.model_predictive_rl.do_action_clip
        if hasattr(config.model_predictive_rl, 'sparse_search'):
            self.sparse_search = config.model_predictive_rl.sparse_search
        self.planning_width = config.model_predictive_rl.planning_width
        self.share_graph_model = config.model_predictive_rl.share_graph_model
        self.linear_state_predictor = config.model_predictive_rl.linear_state_predictor
        self.nodes = config.gcn.nodes
        if hasattr(config.model_predictive_rl, 'with_lstm'):
            self.with_lstm = config.model_predictive_rl.with_lstm

        
        # create edge
        graph_model1 = DGCRNN(config, self.robot_state_dim, self.human_state_dim)
        self.model = GraphValueEstimator(config, graph_model1)
        
    def get_edge(self):
        return self.edge_index

    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def get_model(self):
        return self.model
    