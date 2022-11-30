import torch
import torch.nn as nn
import numpy as np
import math
import networkx as nx
from crowd_nav.policy.helpers import mlp
from crowd_nav.policy.model_predictive_rl import ModelPredictiveRL
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import ObservableState, FullState
from crowd_sim.envs.utils.robot import sliceable_deque


class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical,self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        # assert A.size(0) == self.kernel_size
        x = self.conv(x)
        x = torch.einsum('nctv,ntvw->nctw', (x, A))
        return x.contiguous(), A
    

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn = False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn,self).__init__()
        
#         print("outstg",out_channels)

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)

        x = self.tcn(x) + res
        
        if not self.use_mdn:
            x = self.prelu(x)

        return x, A

class social_stgcnn(nn.Module):
    def __init__(self, config, robot_state_dim, human_state_dim, n_stgcnn =1,n_txpcnn=1,input_feat=32,output_feat=32,
                 seq_len=4,seq_hidden=12, pred_seq_len=1,kernel_size=3):
        super(social_stgcnn,self).__init__()
        self.n_stgcnn= n_stgcnn
        self.n_txpcnn = n_txpcnn
                
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat,output_feat,(kernel_size,seq_len)))
        for j in range(1,self.n_stgcnn):
            self.st_gcns.append(st_gcn(output_feat,output_feat,(kernel_size,seq_len)))
        
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len,seq_hidden,3,padding=1))
        for j in range(1,self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(seq_hidden,seq_hidden,3,padding=1))
        self.tpcnn_ouput = nn.Conv2d(seq_hidden,pred_seq_len,3,padding=1)
            
            
        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())

        self.value_network = mlp(config.gcn.X_dim, config.model_predictive_rl.value_network_dims)
        self.w_r = mlp(robot_state_dim, config.gcn.wr_dims, last_relu=True)
        self.w_h = mlp(human_state_dim, config.gcn.wh_dims, last_relu=True)


        
    def forward(self,rv,hv,a):
        robot_state_embedings = self.w_r(rv)
        # if have humans just doing st_gcns
        if a.shape[-1]!=0:
            human_state_embedings = self.w_h(hv)
            v = torch.cat([robot_state_embedings, human_state_embedings], dim=2)
            v = v.permute(0,3,1,2)
            for k in range(self.n_stgcnn):
                v,a = self.st_gcns[k](v,a)
        else:
            v = robot_state_embedings.permute(0,3,1,2).contiguous()
            
        v = v.view(v.shape[0],v.shape[2],v.shape[1],v.shape[3])
        
        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1,self.n_txpcnn-1):
            v =  self.prelus[k](self.tpcnns[k](v)) + v
            
        v = self.tpcnn_ouput(v)
        v = v.view(v.shape[0],v.shape[2],v.shape[1],v.shape[3])
        robot_state_dim = v[:, :, :, 0]
        robot_state_dim = robot_state_dim.squeeze()
        value = self.value_network(robot_state_dim) # robot state dim
        
        return value

class SSTGCNN_RL(ModelPredictiveRL):
    def __init__(self):
        super().__init__()
        self.name = 'SSTGCNN_RL'
        self.robot_state_dim = 9
        self.human_state_dim = 4
        self.norm_lap_matr = True

    def transform(self, state):
        """
        Take the JointState to tensors

        :param state:
        :return: tensor of shape (# of agent, len(state))
        """
        robot_state_tensor = torch.Tensor([state.robot_state.to_tuple()]).to(self.device)
        human_states_tensor = torch.Tensor([human_state.to_id_tuple() for human_state in state.human_states]). \
            to(self.device)

        return robot_state_tensor, human_states_tensor
    
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
        self.model = social_stgcnn(config, self.robot_state_dim, self.human_state_dim)
    
    def predict(self, state, ego_memory_state):
        """
        Input state is the joint state of robot concatenated by the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """
        # memory state
        self_states, humans_states = [], []
        for i in [-2, -1]: # closest 2 memory
            self_states.append(ego_memory_state['ego'][i].to(self.device))
            humans_states.append(ego_memory_state['humans'][i].to(self.device))
        # current state
        self_states.append(torch.Tensor([state.robot_state.to_tuple()]).to(self.device))
        humans_states.append(torch.Tensor([human_state.to_id_tuple() for human_state in state.human_states]). \
                    to(self.device))

        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.robot_state.v_pref)
        # if not state.human_states:
        #     assert self.phase != 'train'
        #     return self.select_greedy_action(state.robot_state)

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            self.action_values = list()
            max_min_value = float('-inf')
            max_action = None
            all_action_self_states, all_action_human_states, rewards = [], [], []
            for action in self.action_space:
                next_self_state = self.propagate(state.robot_state, action)
                next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                                        for human_state in state.human_states]
                reward = self.compute_reward(next_self_state, next_human_states)
                next_self_state = torch.Tensor([next_self_state.to_tuple()]).to(self.device)
                next_human_states = torch.Tensor([human_state.to_id_tuple() for human_state in next_human_states]). \
                    to(self.device)

                # VALUE UPDATE
                _next_self_state = self_states + [next_self_state]
                _next_human_states = humans_states + [next_human_states]
                all_action_self_states.append(next_self_state)
                all_action_human_states.append(next_human_states)
                rewards.append(reward)
            [r_graph, hs_graph], adj_matrix = self.to_graph([self_states, humans_states], [all_action_self_states, all_action_human_states])
            outputs = self.model(r_graph, hs_graph, adj_matrix).to(self.device)
            self.action_values = [rewards[i] + pow(self.gamma, self.time_step * state.robot_state.v_pref) * outputs[i].data.item() for i in range(len(rewards))]
            max_action = self.action_space[self.action_values.index(max(self.action_values))]


        if self.phase == 'train':
            self.last_state = self.transform(state)

        return max_action

    def to_graph(self, state, next_state=None):
        robot_state, human_states = state
        obs_len = len(robot_state) if isinstance(robot_state, sliceable_deque) else len(robot_state)+1
        # init robot
        if next_state is not None:
            all_next_self_state, all_next_human_states = next_state
            batch = len(all_next_self_state)
            all_history_self_state = [torch.cat([*robot_state], dim=0) for _ in range(batch)]
            robot_seq_feature = torch.zeros((batch, 1, robot_state[0].shape[1], obs_len))
            next_robot_feature = torch.stack(all_next_self_state)
            curr_robot_seq = torch.cat((torch.stack(all_history_self_state), next_robot_feature), dim=1)
        else:
            batch = 1
            all_history_self_state = [torch.cat([*robot_state], dim=0) for _ in range(batch)]
            robot_seq_feature = torch.zeros((batch, 1, robot_state[0].shape[1], obs_len))
            curr_robot_seq = torch.stack(all_history_self_state)
        curr_robot_seq = curr_robot_seq.unsqueeze(1)
        robot_seq_feature = curr_robot_seq.permute(0,1,3,2)
        # init human
        if len(human_states[-1])==0:
            a_ = torch.tensor([]).to(self.device)
            vh_ = torch.tensor([]).to(self.device)
        else:
            all_history_human_states = [torch.cat([*human_states], dim=0) for _ in range(batch)]
            curr_seq_humans = torch.stack(all_history_human_states)
            if next_state is not None:
                next_human_feature = torch.stack(all_next_human_states)
                curr_seq_humans = torch.cat((curr_seq_humans, next_human_feature), dim=1)
            peds_in_curr_seq = torch.unique(curr_seq_humans[:, :, 0])
            curr_seq_rel = torch.zeros((batch, len(peds_in_curr_seq)+1, 2,
                                            obs_len))
            human_seq_feature = torch.zeros((batch, len(peds_in_curr_seq), human_states[-1].shape[1]-1, obs_len)) # remove human id
            for i, ped_id in enumerate([0]+peds_in_curr_seq.tolist()):
                if ped_id == 0: # robot
                    curr_ped_seq = curr_robot_seq[:, :, :, :2].permute(0,1,3,2)
                    curr_ped_seq = curr_ped_seq.squeeze(1)
                else:
                    curr_ped_seq = curr_seq_humans[curr_seq_humans[:, :, 0] ==
                                                        ped_id, :].reshape(batch, -1, curr_seq_humans.shape[2])
                    # if observate lens is not enough just fill last state
                    if curr_ped_seq.shape[1] != obs_len:
                        last_ped_state = curr_ped_seq[:,-1:, :]
                        fill_state_num = obs_len - curr_ped_seq.shape[1]
                        for _ in range(fill_state_num):
                            curr_ped_seq = torch.cat((curr_ped_seq, last_ped_state), dim=1)
                    human_seq_feature[:, i-1, :, :] = curr_ped_seq[:, :, 1:].permute(0,2,1) # remove human id
                    curr_ped_seq = torch.round(curr_ped_seq, decimals=4)
                    curr_ped_seq = curr_ped_seq[:, :, 1:3].permute(0,2,1)
                # Make coordinates relative
                rel_curr_ped_seq = torch.zeros(curr_ped_seq.shape)
                rel_curr_ped_seq[:, :, 1:] = \
                    curr_ped_seq[:, :, 1:] - curr_ped_seq[:, :, :-1]
                curr_seq_rel[:, i, :, :] = rel_curr_ped_seq
            #Convert to Graphs
            a_ = self.seq_to_attrgraph(curr_seq_rel,self.norm_lap_matr).to(self.device)
            vh_ = self.seq_to_nodes(human_seq_feature).to(self.device)  
        vr_ = self.seq_to_nodes(robot_seq_feature).to(self.device)
        return [vr_, vh_], a_


    def propagate(self, state, action):
        if isinstance(state, ObservableState):
            # propagate state of humans
            next_px = state.px + action.vx * self.time_step
            next_py = state.py + action.vy * self.time_step
            next_state = ObservableState(next_px, next_py, action.vx, action.vy, state.radius, state.id)
        elif isinstance(state, FullState):
            # propagate state of current agent
            # perform action without rotation
            if self.kinematics == 'holonomic':
                next_px = state.px + action.vx * self.time_step
                next_py = state.py + action.vy * self.time_step
                next_state = FullState(next_px, next_py, action.vx, action.vy, state.radius,
                                    state.gx, state.gy, state.v_pref, state.theta)
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

    def anorm(self, p1,p2): 
        NORM = torch.sqrt((p1[:, 0]-p2[:, 0])**2+ (p1[:, 1]-p2[:, 1])**2)
        NORM = 1/(NORM)
        NORM[NORM == float("Inf")] = 0
        return NORM

    def seq_to_attrgraph(self, seq_rel,norm_lap_matr = True):
        seq_len = seq_rel.shape[3]
        max_nodes = seq_rel.shape[1]
        batch=seq_rel.shape[0]

        A = np.zeros((batch, seq_len,max_nodes,max_nodes))
        for s in range(seq_len):
            step_rel = seq_rel[:,:,:,s]
            for h in range(step_rel.shape[1]): 
                A[:,s,h,h] = 1
                for k in range(h+1,step_rel.shape[1]):
                    l2_norm = self.anorm(step_rel[:,h],step_rel[:,k])
                    A[:,s,h,k] = l2_norm
                    A[:,s,k,h] = l2_norm
            if norm_lap_matr: 
                for i in range(len(A)):
                    G = nx.from_numpy_matrix(A[i,s,:,:])
                    A[i,s,:,:] = nx.normalized_laplacian_matrix(G).toarray()
                    
        return torch.from_numpy(A).type(torch.float)
    
    def seq_to_nodes(self, seq_rel):
        seq_len = seq_rel.shape[3]
        max_nodes = seq_rel.shape[1]
        node_feature_num = seq_rel.shape[2]
        batch=seq_rel.shape[0]
        
        V = np.zeros((batch, seq_len,max_nodes,node_feature_num))
        for s in range(seq_len):
            step_rel = seq_rel[:,:,:,s]
            for h in range(step_rel.shape[1]): 
                V[:,s,h,:] = step_rel[:,h]
                
        return torch.from_numpy(V).type(torch.float)


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

    def get_state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

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