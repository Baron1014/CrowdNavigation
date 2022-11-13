import os
import logging
import copy
import torch
from tqdm import tqdm
from crowd_sim.envs.utils.info import *


class Explorer(object):
    def __init__(self, env, robot, device, writer, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.writer = writer
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.statistics = None

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None, epoch=None,
                       print_failure=False):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        discomfort = 0
        interrupt = 0
        min_dist = []
        cumulative_rewards = []
        average_returns = []
        collision_cases = []
        timeout_cases = []

        if k != 1:
            pbar = tqdm(total=k)
        else:
            pbar = None

        for i in range(k):
            ob = self.env.reset(phase)
            done = False
            states = []
            actions = []
            rewards = []
            while not done:
                action = self.robot.act(ob)
                ob, reward, done, info = self.env.step(action)
                states.append(self.robot.policy.last_state)
                actions.append(action)
                rewards.append(reward)
                
                if len(info)==2:
                    if isinstance(info[-1], Interrupt):
                        interrupt+=1
                
                info=info[0]
                if isinstance(info, Discomfort):
                    discomfort += 1
                    min_dist.append(info.min_dist)

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    if self.target_policy.name=='SSTGCNN_RL':
                        self.update_temporal_memory(states, actions, rewards, imitation_learning)
                    else:
                        self.update_memory(states, actions, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))
            returns = []
            for step in range(len(rewards)):
                step_return = sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                   * reward for t, reward in enumerate(rewards[step:])])
                returns.append(step_return)
            average_returns.append(average(returns))

            if pbar:
                pbar.update(1)
        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        extra_info = extra_info + '' if epoch is None else extra_info + ' in epoch {} '.format(epoch)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f},'
                     ' average return: {:.4f}'. format(phase.upper(), extra_info, success_rate, collision_rate,
                                                       avg_nav_time, average(cumulative_rewards),
                                                       average(average_returns)))
        if phase in ['val', 'test']:
            total_time = sum(success_times + collision_times + timeout_times)
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         discomfort / total_time, average(min_dist))
            logging.info("Interrupt human interaction: {:.2f}".format(interrupt/total_time))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

        self.statistics = success_rate, collision_rate, avg_nav_time, average(cumulative_rewards), average(average_returns)

        return self.statistics

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')
        
        for i, state in enumerate(states[:-1]):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                next_state = self.target_policy.transform(states[i+1])
                value = sum([pow(self.gamma, (t - i) * self.robot.time_step * self.robot.v_pref) * reward *
                             (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                next_state = states[i+1]
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    value = 0
            value = torch.Tensor([value]).to(self.device)
            reward = torch.Tensor([rewards[i]]).to(self.device)
            # state = state.to(self.device)
            # next_state = next_state.to(self.device)

            if self.target_policy.name == 'ModelPredictiveRL' or self.target_policy.name == 'SSTGCNN_RL':
                self.memory.push((state[0], state[1], value, reward, next_state[0], next_state[1]))
            elif self.target_policy.name == 'DGCNRL':
                self.memory.push((self.target_policy.to_graph((state[0], state[1])), value, reward, self.target_policy.to_graph((next_state[0], next_state[1]))))
            else:
                self.memory.push((state, value, reward, next_state))

    def update_temporal_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')
        
        graphs, adj_matrixs = [], []
        for i, state in enumerate(states):
            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
            self.memory.push_temporal(state)
            if self.memory.temporal_memory_fill():
                graph, adj = self.to_graph(state)

                graphs.append(graph)
                adj_matrixs.append(adj)

        for i, graph in enumerate(graphs[:-1]):
            reward = rewards[i]
            if imitation_learning:
                value = sum([pow(self.gamma, (t - i) * self.robot.time_step * self.robot.v_pref) * reward *
                             (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    value = 0
            value = torch.Tensor([value]).to(self.device)
            reward = torch.Tensor([rewards[i]]).to(self.device)

            self.memory.push((graph[0], graph[1], adj_matrixs[i], value, reward, graphs[i+1][0], graphs[i+1][1], adj_matrixs[i+1]))
    
    def to_graph(self, state):
        robot_state, human_states = state
        curr_seq_humans = torch.cat([*self.memory.temporal_memory['humans']], dim=0)
        peds_in_curr_seq = torch.unique(curr_seq_humans[:, 0])
        curr_seq_rel = torch.zeros((len(peds_in_curr_seq)+1, 2,
                                        self.memory.obs_len))
        human_seq_feature = torch.zeros((len(peds_in_curr_seq), human_states.shape[1]-1, self.memory.obs_len)) # remove human id
        robot_seq_feature = torch.zeros((1, robot_state.shape[1], self.memory.obs_len)) 
        for i, ped_id in enumerate([0]+peds_in_curr_seq.tolist()):
            if ped_id == 0: # robot
                curr_robot_seq = torch.cat([*self.memory.temporal_memory['robot']], dim=0)
                robot_seq_feature[i, :, :] = curr_robot_seq.t()
                curr_ped_seq = curr_robot_seq[:, :2].t()
            else:
                curr_ped_seq = curr_seq_humans[curr_seq_humans[:, 0] ==
                                                    ped_id, :]
                human_seq_feature[i-1, :, :] = curr_ped_seq[:, 1:].t() # remove human id
                curr_ped_seq = torch.round(curr_ped_seq, decimals=4)
                if len(curr_ped_seq) != self.memory.obs_len:
                    raise NotImplementedError
                curr_ped_seq = curr_ped_seq[:, 1:3].t()
            # Make coordinates relative
            rel_curr_ped_seq = torch.zeros(curr_ped_seq.shape)
            rel_curr_ped_seq[:, 1:] = \
                curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
            curr_seq_rel[i, :, :] = rel_curr_ped_seq

        #Convert to Graphs
        a_ = self.target_policy.seq_to_attrgraph(curr_seq_rel,self.target_policy.norm_lap_matr)
        vh_ = self.target_policy.seq_to_nodes(human_seq_feature)
        vr_ = self.target_policy.seq_to_nodes(robot_seq_feature)
        return [vr_, vh_], a_
            


    def log(self, tag_prefix, global_step):
        sr, cr, time, reward, avg_return = self.statistics
        self.writer.log({tag_prefix + '/success_rate': sr}, step=global_step)
        self.writer.log({tag_prefix + '/collision_rate': cr}, step=global_step)
        self.writer.log({tag_prefix + '/time': time}, step=global_step)
        self.writer.log({tag_prefix + '/reward': reward}, step=global_step)
        self.writer.log({tag_prefix + '/avg_return': avg_return}, step=global_step)
    


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
