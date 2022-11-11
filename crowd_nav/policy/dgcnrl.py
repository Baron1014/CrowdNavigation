from crowd_sim.envs.policy.policy import Policy
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, ChebConv
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
from crowd_sim.envs.utils.state import ObservableState, FullState
from torch_geometric.data import Data, HeteroData
from itertools import combinations

class DGCRNN(nn.Module):
    def __init__(self, config, robot_state_dim, human_state_dim):
        super().__init__()
        wr_dims = config.gcn.wr_dims
        wh_dims = config.gcn.wh_dims

        self.X_dim = config.gcn.X_dim
        self.num_layer = config.gcn.num_layer
        self.skip_connection = config.gcn.skip_connection
        self.dcrnn = ChebConv(self.X_dim, self.X_dim, 3)
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
        robot_feature, human_feature = [], []
        for graph in graphs:
            robot_feature.append(graph['robot'].x)
            human_feature.append(graph['human'].x)
            edge_index = torch.tensor([])
            for edge in graph.edge_types:
                edge_index = torch.cat([edge_index, graph[edge]['edge_index']], 1)
        
        robot_state_embedings = self.w_r(torch.stack(robot_feature))
        human_state_embedings = self.w_h(torch.stack(human_feature))
        X = torch.cat([robot_state_embedings, human_state_embedings], dim=1)
        edge_index = edge_index.type(torch.LongTensor)

        dcrnn_out = self.dcrnn(X, edge_index)
        return dcrnn_out[:, 0, :]

    

class DGCNRL(ModelPredictiveRL):
    def __init__(self):
        super().__init__()
        self.name = 'DGCNRL'
        self.robot_state_dim = 9
        self.human_state_dim = 6
    
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

    def save_model(self, file):
        torch.save(self.model.state_dict(), file)

    def set_time_step(self, time_step):
        self.time_step = time_step
        self.model.time_step = time_step

    def load_model(self, state_dict):
        self.model.load_state_dict(state_dict)

    def predict(self, state):
        """
        Input state is the joint state of robot concatenated by the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.robot_state.v_pref)
        if not state.human_states:
            assert self.phase != 'train'
            return self.select_greedy_action(state.robot_state)

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            self.action_values = list()
            max_min_value = float('-inf')
            max_action = None
            for action in self.action_space:
                next_self_state = self.propagate(state.robot_state, action)
                next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                                        for human_state in state.human_states]
                reward = self.compute_reward(next_self_state, next_human_states)
                next_self_state = torch.Tensor([next_self_state.to_tuple()]).to(self.device)
                next_human_states = torch.Tensor([human_state.to_tuple() for human_state in next_human_states]). \
                    to(self.device)

                # VALUE UPDATE
                outputs = self.model(self.to_graph([next_self_state, next_human_states]))
                min_output, min_index = torch.min(outputs, 0)
                min_value = reward + pow(self.gamma, self.time_step * state.robot_state.v_pref) * min_output.data.item()
                self.action_values.append(min_value)
                if min_value > max_min_value:
                    max_min_value = min_value
                    max_action = action

        if self.phase == 'train':
            self.last_state = self.transform(state)

        return max_action
    
    def propagate(self, state, action):
        if isinstance(state, ObservableState):
            # propagate state of humans
            next_px = state.px + action.vx * self.time_step
            next_py = state.py + action.vy * self.time_step
            next_state = ObservableState(next_px, next_py, action.vx, action.vy, state.radius)
        elif isinstance(state, FullState):
            # propagate state of current agent
            # perform action without rotation
            if self.kinematics == 'holonomic':
                next_px = state.px + action.vx * self.time_step
                next_py = state.py + action.vy * self.time_step
                next_state = FullState(next_px, next_py, action.vx, action.vy,
                                    state.gx, state.gy, state.v_pref, state.theta, state.radius)
                # next_state = RobotState(next_px, next_py, action.vx, action.vy,
                #                        state.gx, state.gy, state.v_pref, state.theta, robot_size=(state.length, state.width))
            else:
                next_theta = state.theta + action.r
                next_vx = action.v * np.cos(next_theta)
                next_vy = action.v * np.sin(next_theta)
                next_px = state.px + next_vx * self.time_step
                next_py = state.py + next_vy * self.time_step
                # next_state = RobotState(next_px, next_py, next_vx, next_vy, state.gx, state.gy,
                #                        state.v_pref, next_theta, robot_size=(state.length, state.width))
        else:
            raise ValueError('Type error')

        return next_state

    def compute_reward(self, nav, humans):
        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(humans):
            # dist = getCloestEdgeDist(nav.px, nav.py, human.px, human.py, nav.width/2, nav.length/2) - human.radius
            dist = np.linalg.norm((nav.px - human.px, nav.py - human.py)) - nav.radius - human.radius
            if dist < 0:
                collision = True
                break
            if dist < dmin:
                dmin = dist

        # check if reaching the goal
        reaching_goal = np.linalg.norm((nav.px - nav.gx, nav.py - nav.gy)) < nav.radius
        # goal_delta_x, goal_delta_y = nav.px - nav.gx, nav.py - nav.gy
        # reaching_goal = abs(goal_delta_x) < nav.width/2 and abs(goal_delta_y) < nav.length/2
        if collision:
            reward = -0.25
        elif reaching_goal:
            reward = 1
        elif dmin < 0.2:
            reward = (dmin - 0.2) * 0.5 * self.time_step
        else:
            reward = 0

        return reward

    def to_graph(self, state):
        robot_state, human_states = state
        human_batch = len(human_states)
        # spatial edge
        rh_weights = torch.norm(torch.cat([(human_states[:, 0] - robot_state[:, 0]).reshape((human_batch, -1)), (human_states[:, 1] - robot_state[:, 1]).
                                reshape((human_batch, -1))], dim=1), 2, dim=1, keepdim=True) # h_position - r_posiotion
        edge_index = [[0, i] for i in range(1, human_batch+1)] # create rh_edges
        # hh_edge & weight
        hh_weights = torch.tensor([])
        for i, j in combinations([*range(human_batch)], 2):
            node_i, node_j = i+1, j+1
            edge_index.append([node_i, node_j]) # create hh_edges

        human_states = torch.cat((human_states, rh_weights), 1)
        data = HeteroData({
            'robot':{
                'x': robot_state
            }, 
            'human': {
                'x': human_states
            }            
        })
        data['robot', 'human'].edge_index = torch.tensor(edge_index).t().contiguous() # robot_human to human_human
        data['human', 'robot'].edge_index = torch.flip(torch.tensor(edge_index),  dims=(1,)).t().contiguous() # human_human to robot_human
        
        return [data]