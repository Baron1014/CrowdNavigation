import os
import logging
import copy
import torch
from tqdm import tqdm
from crowd_sim.envs.utils.info import *
from numpy.linalg import norm


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
        self.freq_in_danger = 0
        self.min_seperate = 0

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
        min_dist = []
        cumulative_rewards = []
        average_returns = []
        collision_cases = []
        timeout_cases = []
        success_dists = []

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

                if isinstance(info, Discomfort):
                    discomfort += 1
                    min_dist.append(info.min_dist)

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
                dists = []
                for v in actions:
                    dists.append(-norm(v) if v[1]<0 else norm(v))
                success_dists.append(sum(dists)*self.robot.time_step)

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
            self.freq_in_danger = discomfort / total_time
            self.min_seperate = average(min_dist)
            self.avg_dist = average(success_dists)
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         self.freq_in_danger, self.min_seperate)
            logging.info('Average moving distance: %.4f', self.avg_dist)
            
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
        self.robot.clean_ego_memory()
        for i, state in enumerate(states):
            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
            self.robot.push_ego_memory(state)
            if self.robot.ego_memory_fill():
                state = self.robot.get_ego_jointstate()
                graph, adj = self.target_policy.to_graph(state)

                graphs.append(graph)
                adj_matrixs.append(adj)

        for i in range(self.robot.obs_len-1, len(states[:-1])):
            g_idx = i-(self.robot.obs_len-1)
            graph = graphs[g_idx]
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
            current_human_num = 0 if len(graph[1])==0 else graph[1].shape[2]
            next_human_num = 0 if len(graphs[g_idx+1][1])==0 else graphs[g_idx+1][1].shape[2]
            if current_human_num == next_human_num:
                # only push human number of current state equal to next state
                self.memory.push((graph[0], graph[1], adj_matrixs[g_idx], value, reward, graphs[g_idx+1][0], graphs[g_idx+1][1], adj_matrixs[g_idx+1]))
    


    def log(self, tag_prefix, global_step):
        sr, cr, time, reward, avg_return = self.statistics
        self.writer.log({tag_prefix + '/success_rate': sr}, step=global_step)
        self.writer.log({tag_prefix + '/collision_rate': cr}, step=global_step)
        self.writer.log({tag_prefix + '/time': time}, step=global_step)
        self.writer.log({tag_prefix + '/reward': reward}, step=global_step)
        self.writer.log({tag_prefix + '/avg_return': avg_return}, step=global_step)
        if 'test' in tag_prefix or 'val' in tag_prefix:
            self.writer.log({tag_prefix + '/frequency_in_danger':self.freq_in_danger}, step=global_step)
            self.writer.log({tag_prefix + '/avg_min_separate_dist':self.min_seperate}, step=global_step)

    


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
