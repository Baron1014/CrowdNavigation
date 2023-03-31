import logging
import random

import gym
import matplotlib.lines as mlines
from matplotlib import patches
import numpy as np
from numpy.linalg import norm

from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.state import tensor_to_joint_state
from crowd_sim.envs.utils.action import ActionRot
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.robot_sensor_range = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_scenario = None
        self.test_scenario = None
        self.current_scenario = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        self.nonstop_human = None
        self.centralized_planning = None
        self.centralized_planner = None

        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        self.robot_actions = None
        self.rewards = None
        self.As = None
        self.Xs = None
        self.feats = None
        self.trajs = list()
        self.panel_width = 10
        self.panel_height = 10
        self.panel_scale = 1
        self.test_scene_seeds = []
        self.dynamic_human_num = []
        self.human_starts = []
        self.human_goals = []

        self.phase = None

    def configure(self, config):
        self.config = config
        self.time_limit = config.env.time_limit
        self.time_step = config.env.time_step
        self.randomize_attributes = config.env.randomize_attributes
        self.robot_sensor_range = config.env.robot_sensor_range
        self.success_reward = config.reward.success_reward
        self.collision_penalty = config.reward.collision_penalty
        self.discomfort_dist = config.reward.discomfort_dist
        self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': config.env.train_size, 'val': config.env.val_size,
                          'test': config.env.test_size}
        self.train_val_scenario = config.sim.train_val_scenario
        self.test_scenario = config.sim.test_scenario
        self.square_width = config.sim.square_width
        self.circle_radius = config.sim.circle_radius
        self.human_num = config.sim.human_num

        self.nonstop_human = config.sim.nonstop_human
        self.centralized_planning = config.sim.centralized_planning
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        human_policy = config.humans.policy
        if self.centralized_planning:
            if human_policy == 'socialforce':
                logging.warning('Current socialforce policy only works in decentralized way with visible robot!')
            self.centralized_planner = policy_factory['centralized_' + human_policy]()

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_scenario, self.test_scenario))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_robot(self, robot):
        self.robot = robot

    def generate_human(self, _id, human=None):
        if self.current_scenario == 'circle_crossing':
            human = self.random_circle_crossing_human(_id)

        elif self.current_scenario == 'square_crossing':
            human = self.random_square_crossing_human(_id)

        return human
    
    def generate_static_human(self, _id):
        human = Human(_id, self.config, 'humans')
        human.sample_random_attributes()
        while True:
            # add some noise to simulate all the possible cases robot could meet with human
            px = 0
            py = 0
            collide = False
            for i, agent in enumerate([self.robot] + self.humans):
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)

        return human

    def random_circle_crossing_human(self, _id):
        human = Human(_id, self.config, 'humans')
        human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for i, agent in enumerate([self.robot] + self.humans):
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)

        return human


    def random_square_crossing_human(self, _id):
        human = Human(_id, self.config, 'humans')
        human.sample_random_attributes()

        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width
            collide = False
            for i, agent in enumerate([self.robot] + self.humans):
                if i == 0:
                    agent_dist = (self.robot.width**2+self.robot.length**2)**0.5
                else:
                    agent_dist = agent.radius
                if norm((px - agent.px, py - agent.py)) < human.radius + agent_dist + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * - sign
            gy = np.random.random() * self.square_width 
            collide = False
            stay = False
            for i, agent in enumerate([self.robot] + self.humans):
                if i == 0:
                    agent_dist = (self.robot.width**2+self.robot.length**2)**0.5
                else:
                    agent_dist = agent.radius
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent_dist + self.discomfort_dist:
                    collide = True
                    break
                width = abs(self.square_width*0.5)
                if abs(gx) < width and abs(gy) < width:
                    stay = True
                    break
            if not collide and not stay:
                break
        human.set(px, py, gx, gy, 0, 0, 0)

        return human

    def random_static_human(self):
        human = Human(self.config, 'humans', static=True)
        human.sample_random_attributes()

        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.4 * sign
            py = (np.random.random() + 0.1) * self.square_width * 0.3
            collide = False
            for i, agent in enumerate([self.robot] + self.humans):
                if i == 0:
                    agent_dist = (self.robot.width**2+self.robot.length**2)**0.5
                else:
                    agent_dist = agent.radius
                if norm((px - agent.px, py - agent.py)) < human.radius + agent_dist + self.discomfort_dist:
                    collide = True
                    break
            robot_goal = [self.robot.gx, self.robot.gy]
            if norm((px - robot_goal[0], py - robot_goal[1])) < human.radius + self.discomfort_dist:
                collide = True
            if not collide:
                break
        human.set(px, py, px, py, 0, 0, 0)
        
        return human

    def generate_robot(self):
       # while True:
       #     px, gx = np.random.uniform(-self.circle_radius, self.circle_radius, 2)
            # always to up
       #     py = np.random.uniform(-self.circle_radius-1, -self.circle_radius+1)
       #     gy = np.random.uniform(self.circle_radius-1, self.circle_radius+1)
       #     if np.linalg.norm([px - gx, py - gy]) >= 8:
       #         break
       
       start_or_goal = self.circle_radius+1
       pos = 1 if random.random()>0.5 else -1
       if self.config.robot.gx and self.config.robot.gy:
            px, py = 0, 0
            gx, gy = self.config.robot.gx , self.config.robot.gy
       else:
            px, gx = 0, 0
            py, gy = -pos*(start_or_goal), pos*start_or_goal
       self.robot.set(px, py, gx, gy, 0, 0, pos*np.pi/2)
        

    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        if self.robot is None:
            raise AttributeError('Robot has to be set!')

        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0

        base_seed = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                     'val': 0, 'test': self.case_capacity['val']}
        # clean robot temporal memory
        self.robot.clean_ego_memory()

        # (px, py, gx, gy, vx, vy, theta)
        # self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, 0)
        if self.case_counter[phase] >= 0:
            np.random.seed(base_seed[phase] + self.case_counter[phase])
            random.seed(base_seed[phase] + self.case_counter[phase])
            # if phase == 'test':
            #     logging.info('current test seed is:{}'.format(base_seed[phase] + self.case_counter[phase]))
            if not self.robot.policy.multiagent_training and phase in ['train', 'val']:
                # only CADRL trains in circle crossing simulation
                human_num = 1
                self.current_scenario = 'circle_crossing'
            else:
                self.current_scenario = self.test_scenario
                human_num = self.human_num
            self.generate_robot()
            self.generate_all_humans(human_num)
            

            # case_counter is always between 0 and case_size[phase]
            self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
        else:
            assert phase == 'test'
            if self.case_counter[phase] == -1:
                # for debugging purposes
                self.human_num = 3
                self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
            else:
                raise NotImplementedError
        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        if self.centralized_planning:
            self.centralized_planner.time_step = self.time_step

        self.states = list()
        self.robot_actions = list()
        self.rewards = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()
        if hasattr(self.robot.policy, 'get_matrix_A'):
            self.As = list()
        if hasattr(self.robot.policy, 'get_feat'):
            self.feats = list()
        if hasattr(self.robot.policy, 'get_X'):
            self.Xs = list()
        if hasattr(self.robot.policy, 'trajs'):
            self.trajs = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob = self.compute_observation_for(self.robot)
        elif self.robot.sensor == 'RGBD':
            ob, _ = self.compute_fov_observation_for()

        return ob
    
    def generate_all_humans(self, human_num):
        self.humans = list()
        if self.current_scenario == "social_aware":
            humun_inter_num =int(human_num*self.human_social_rate) 
            human_static_num = int(human_num*self.human_static_rate)
            if humun_inter_num >= 2:
                humun_inter_num = humun_inter_num if humun_inter_num % 2 == 0 else humun_inter_num - 1
                human_pair = humun_inter_num//2
                for _ in range(human_pair):
                    self.humans += self.generate_human_pair()
                # generate static human
                for _ in range(human_static_num):
                    self.humans.append(self.random_static_human())
            else:
                human_move_num = human_num
            human_move_num = human_num - humun_inter_num - human_static_num
            for _ in range(human_move_num):
                # self.humans.append(self.random_square_crossing_human())
                self.humans.append(self.random_circle_crossing_human())
                
        else:
            for id in range(human_num):
                self.humans.append(self.generate_human(id+1))


    def generate_human_pair(self):
        human1 = Human(self.config, 'humans')
        human2 = Human(self.config, 'humans')
        if self.randomize_attributes:
            human1.sample_random_attributes()
            human2.sample_random_attributes()
        
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = random.random() * self.square_width * 0.25 * sign
            py = (random.random() - 0.5) * self.square_width * 0.1
            min_path_dist = human1.radius + human2.radius  + (self.robot.width**2+self.robot.length**2)**0.5 + self.discomfort_dist
            while True:
                a = random.uniform(-1, 1)
                px1 = px + random.uniform(-2, 2) * ((self.robot.width**2+self.robot.length**2)**0.5) 
                py1 = py + random.uniform(-2, 2) * sign
                collide = False
                if norm((px-px1, py-py1)) > min_path_dist:
                    break
            for i, agent in enumerate([self.robot] + self.humans):
                if i == 0:
                    agent_dist = (self.robot.width**2+self.robot.length**2)**0.5
                else:
                    agent_dist = agent.radius
                if norm((px - agent.px, py - agent.py)) < human1.radius + agent_dist + self.discomfort_dist:
                    collide = True
                    break
                if norm((px1 - agent.px, py1 - agent.py)) < human2.radius + agent_dist + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human1.set(px, py, px, py, 0, 0, 0, interaction=[human2])
        human2.set(px1, py1, px1, py1, 0, 0, 0, interaction=[human1])

        return [human1, human2]

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        if self.centralized_planning:
            agent_states = [human.get_full_state() for human in self.humans]
            if self.robot.visible:
                agent_states.append(self.robot.get_full_state())
                human_actions = self.centralized_planner.predict(agent_states)[:-1]
            else:
                human_actions = self.centralized_planner.predict(agent_states)
        else:
            human_actions = []
            for human in self.humans:
                ob = self.compute_observation_for(human)
                human_actions.append(human.act(ob))

        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                vx = human.vx - action.vx
                vy = human.vy - action.vy
            else:
                vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                logging.debug("Collision: distance between robot and p{} is {:.2E} at time {:.2E}".format(human.id, closest_dist, self.global_time))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Discomfort(dmin)
        else:
            reward = 0
            done = False
            info = Nothing()

        if update:
            # store state, action value and attention weights
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())
            if hasattr(self.robot.policy, 'get_matrix_A'):
                self.As.append(self.robot.policy.get_matrix_A())
            if hasattr(self.robot.policy, 'get_feat'):
                self.feats.append(self.robot.policy.get_feat())
            if hasattr(self.robot.policy, 'get_X'):
                self.Xs.append(self.robot.policy.get_X())
            if hasattr(self.robot.policy, 'traj'):
                self.trajs.append(self.robot.policy.get_traj())

            # update all agents
            self.robot.step(action)
            for human, action in zip(self.humans, human_actions):
                human.step(action)
                if self.nonstop_human and human.reached_destination():
                    self.update_human_goal(human)

            self.global_time += self.time_step
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans],
                                [human.id for human in self.humans]])
            self.robot_actions.append(action)
            self.rewards.append(reward)

            # compute the observation
            if self.robot.sensor == 'coordinates':
                ob = self.compute_observation_for(self.robot)
            elif self.robot.sensor == 'RGBD':
                ob, _ = self.compute_fov_observation_for()
        else:
            if self.robot.sensor == 'coordinates':
                ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError

        return ob, reward, done, info
    
    def update_human_goal(self, human):
        while True:
            humans_other = []
            for h in self.humans:
                if h != human:
                    humans_other.append(h)
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            v_pref = 1.0 if human.v_pref == 0 else human.v_pref
            gx_noise = (np.random.random() - 0.5) * v_pref
            gy_noise = (np.random.random() - 0.5) * v_pref
            gx = self.circle_radius * np.cos(angle) + gx_noise
            gy = self.circle_radius * np.sin(angle) + gy_noise
            collide = False
            for agent in [self.robot] + humans_other:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((gx - agent.px, gy - agent.py)) < min_dist or \
                        norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break

        # Give human new goal
        human.gx = gx
        human.gy = gy


    def detect_visible(self, state1, state2):
        # if self.robot.kinematics == 'holonomic':
        #     real_theta = np.arctan2(state1.vy, state1.vx)
        # else:
        #     real_theta = state1.theta
        real_theta = state1.theta
        # angle of center line of FOV of agent1
        v_fov = [np.cos(real_theta), np.sin(real_theta)]

        # angle between agent1 and agent2
        v_12 = [state2.px - state1.px, state2.py - state1.py]
        # angle between center of FOV and agent 2

        v_fov = v_fov / np.linalg.norm(v_fov)
        v_12 = v_12 / np.linalg.norm(v_12)

        offset = np.arccos(np.clip(np.dot(v_fov, v_12), a_min=-1, a_max=1))

        fov = self.robot.FoV
        if np.abs(offset) <= fov / 2:
            return True
        else:
            return False
    
    def compute_fov_observation_for(self):
        ob, ob_ids = [], []
        for i in range(self.human_num):
            visible = self.detect_visible(self.robot, self.humans[i])
            if visible:
                ob.append(self.humans[i].get_id_observable_state())
            ob_ids.append(visible)
        
        return ob, ob_ids


    def compute_observation_for(self, agent):
        if agent == self.robot:
            ob = []
            for human in self.humans:
                # ob.append(human.get_observable_state())
                ob.append(human.get_id_observable_state())
        else:
            # ob = [other_human.get_observable_state() for other_human in self.humans if other_human != agent]
            ob = [other_human.get_id_observable_state() for other_human in self.humans if other_human != agent]
            if self.robot.visible:
                # ob += [self.robot.get_observable_state()]
                ob += [self.robot.get_id_observable_state()]
        return ob

    def render(self, mode='video', output_file=None, info=False):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        frame = 0
        x_offset = 0.2
        y_offset = 0.4
        cmap = plt.cm.get_cmap('Pastel1', 10)
        robot_color = 'black'
        # human_color = 'black'
        goal_color = robot_color
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)
        display_numbers = True

        if mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-9, 9)
            ax.set_ylim(-9, 9)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add human start positions and goals
            # human_colors = [cmap(i) for i in range(len(self.humans))]
            # disable showing FoV
            human_colors = []
            for state in self.states:
                colors = []
                for h in range(len(self.humans)):
                    # green: visible; red: invisible
                    colors.append('#A6FFA6' if self.detect_visible(state[0], state[1][h]) else '#FF7575')
                human_colors.append(colors)

            for i in range(len(self.humans)):
                human = self.humans[i]
                # human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                #                            color=human_colors[i],
                #                            marker='*', linestyle='None', markersize=15)
                # ax.add_artist(human_goal)
                human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                            color=human_colors[i][0],
                                            marker='o', linestyle='None', markersize=13)
                ax.add_artist(human_start)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]
            
            # add robot goal
            robot_goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                           color=robot_color,
                                           marker='*', linestyle='None', markersize=15)
            ax.add_artist(robot_goal)
            robot_start = mlines.Line2D([self.robot.get_start_position()[0]], [self.robot.get_start_position()[1]],
                                            color=robot_color,
                                            marker='o', linestyle='None', markersize=13)
            ax.add_artist(robot_start)

            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=False, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=human_colors[k][i])
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)

                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 1)]
                    for time in times:
                       ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=human_colors[k][i], ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            if output_file:
                plt.savefig(output_file + ".png", dpi=600)
            else:
                plt.show()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=12)
            ax.set_xlim(-9, 9)
            ax.set_ylim(-9, 9)
            ax.set_xlabel('x(m)', fontsize=14)
            ax.set_ylabel('y(m)', fontsize=14)
            show_human_start_goal = False

            # add human start positions and goals
            # human_colors = [cmap(i) for i in range(len(self.humans))]
            if show_human_start_goal:
                for i in range(len(self.humans)):
                    human = self.humans[i]
                    human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                               color=human_colors[i],
                                               marker='*', linestyle='None', markersize=8)
                    ax.add_artist(human_goal)
                    human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                                color=human_colors[i],
                                                marker='o', linestyle='None', markersize=8)
                    ax.add_artist(human_start)
            # add robot start position
            robot_start = mlines.Line2D([self.robot.get_start_position()[0]], [self.robot.get_start_position()[1]],
                                        color=robot_color,
                                        marker='o', linestyle='None', markersize=8)
            ax.add_artist(robot_start)
            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                 color=robot_color, marker='*', linestyle='None',
                                 markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=False, color=robot_color, label='Robot')
            # sensor_range = plt.Circle(robot_positions[0], self.robot_sensor_range, fill=False, ls='dashed')
            ax.add_artist(robot)
            ax.add_artist(goal)
            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=14)

            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False, color=cmap(i))
                      for i in range(len(self.humans))]

            # disable showing human numbers
            if display_numbers:
                human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] + y_offset, str(i),
                                          color='black') for i in range(len(self.humans))]
            # disable showing FoV
            human_colors = []
            for state in self.states:
                colors = []
                for h in range(len(humans)):
                    # green: visible; red: invisible
                    colors.append('g' if self.detect_visible(state[0], state[1][h]) else 'r')
                human_colors.append(colors)

            for i, human in enumerate(humans):
                human.set_color(c=human_colors[0][i])
                ax.add_artist(human)
                if display_numbers:
                    ax.add_artist(human_numbers[i])

            # add time annotation
            time = plt.text(0.4, 0.9, 'Time: {}'.format(0), fontsize=16, transform=ax.transAxes)
            ax.add_artist(time)

            # visualize attention scores
            # if hasattr(self.robot.policy, 'get_attention_weights'):
            #     attention_scores = [
            #         plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
            #                  fontsize=16) for i in range(len(self.humans))]

            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            orientations = []
            for i in range(self.human_num + 1):
                orientation = []
                for state in self.states:
                    agent_state = state[0] if i == 0 else state[1][i - 1]
                    if self.robot.kinematics == 'unicycle' and i == 0:
                        direction = (
                        (agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(agent_state.theta),
                                                           agent_state.py + radius * np.sin(agent_state.theta)))
                    else:
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        direction = ((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                                                        agent_state.py + radius * np.sin(theta)))
                    orientation.append(direction)
                orientations.append(orientation)
                if i == 0:
                    arrow_color = 'black'
                    arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)]
                else:
                    arrows.extend(
                        [patches.FancyArrowPatch(*orientation[0], color=human_colors[0][i - 1], arrowstyle=arrow_style)])

            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            def calcFOVLineEndPoint(ang, point, extendFactor):
                # choose the extendFactor big enough
                # so that the endPoints of the FOVLine is out of xlim and ylim of the figure
                FOVLineRot = np.array([[np.cos(ang), -np.sin(ang), 0],
                                    [np.sin(ang), np.cos(ang), 0],
                                    [0, 0, 1]])
                point.extend([1])
                # apply rotation matrix
                newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
                # increase the distance between the line start point and the end point
                newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
                return newPoint
            
            def draw_fov(frame):            
                # draw FOV for the robot
                # add robot FOV
                artists = []
                if self.robot.FoV < np.pi * 2:
                    FOVAng = self.robot.FoV / 2
                    FOVLine1 = mlines.Line2D([0, 0], [0, 0], linestyle='--')
                    FOVLine2 = mlines.Line2D([0, 0], [0, 0], linestyle='--')


                    startPointX, startPointY = robot_positions[frame]
                    endPointX = startPointX + radius* np.cos(self.robot.theta)
                    endPointY = startPointY + radius* np.sin(self.robot.theta)

                    # transform the vector back to world frame origin, apply rotation matrix, and get end point of FOVLine
                    # the start point of the FOVLine is the center of the robot
                    FOVEndPoint1 = calcFOVLineEndPoint(FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
                    FOVLine1.set_xdata(np.array([startPointX, startPointX + FOVEndPoint1[0]]))
                    FOVLine1.set_ydata(np.array([startPointY, startPointY + FOVEndPoint1[1]]))
                    FOVEndPoint2 = calcFOVLineEndPoint(-FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
                    FOVLine2.set_xdata(np.array([startPointX, startPointX + FOVEndPoint2[0]]))
                    FOVLine2.set_ydata(np.array([startPointY, startPointY + FOVEndPoint2[1]]))

                    ax.add_artist(FOVLine1)
                    ax.add_artist(FOVLine2)
                    artists.append(FOVLine1)
                    artists.append(FOVLine2)
                return artists

            artists = draw_fov(0)
            # if len(self.trajs) != 0:
            #     human_future_positions = []
            #     human_future_circles = []
            #     for traj in self.trajs:
            #         human_future_position = [[tensor_to_joint_state(traj[step+1][0]).human_states[i].position
            #                                   for step in range(self.robot.policy.planning_depth)]
            #                                  for i in range(self.human_num)]
            #         human_future_positions.append(human_future_position)

            #     for i in range(self.human_num):
            #         circles = []
            #         for j in range(self.robot.policy.planning_depth):
            #             circle = plt.Circle(human_future_positions[0][i][j], self.humans[0].radius/(1.7+j), fill=False, color=cmap(i))
            #             ax.add_artist(circle)
            #             circles.append(circle)
            #         human_future_circles.append(circles)
            


            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                nonlocal artists
                global_step = frame_num
                robot.center = robot_positions[frame_num]

                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human.set_color(c=human_colors[frame_num][i])
                    if display_numbers:
                        human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] + y_offset))

                for arrow in arrows:
                    arrow.remove()
                for artist in artists:
                    artist.remove()

                artists = draw_fov(frame_num)

                for i in range(self.human_num + 1):
                    orientation = orientations[i]
                    if i == 0:
                        arrows = [patches.FancyArrowPatch(*orientation[frame_num], color='black',
                                                          arrowstyle=arrow_style)]
                    else:
                        arrows.extend([patches.FancyArrowPatch(*orientation[frame_num], color=human_colors[frame_num][i - 1],
                                                               arrowstyle=arrow_style)])

                for arrow in arrows:
                    ax.add_artist(arrow)
                    # if hasattr(self.robot.policy, 'get_attention_weights'):
                    #     attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

                # if len(self.trajs) != 0:
                #     for i, circles in enumerate(human_future_circles):
                #         for j, circle in enumerate(circles):
                #             circle.center = human_future_positions[global_step][i][j]

                if info:
                    if len(self.states)-1 == frame_num:
                        if str(info)=='Collision':
                            plt.title(info, color='red', fontsize=20, fontweight="bold")
                        elif str(info)=='Reaching goal':
                            plt.title(info, color='#22C32E', fontsize=20, fontweight="bold")
                        else:
                            plt.title(info, color='#FFBF00', fontsize=20, fontweight="bold")

            def plot_value_heatmap():
                if self.robot.kinematics != 'holonomic':
                    print('Kinematics is not holonomic')
                    return
                # for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                #     print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                #                                              agent.vx, agent.vy, agent.theta))

                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (self.robot.policy.rotation_samples, self.robot.policy.speed_samples))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def print_matrix_A():
                # with np.printoptions(precision=3, suppress=True):
                #     print(self.As[global_step])
                h, w = self.As[global_step].shape
                print('   ' + ' '.join(['{:>5}'.format(i - 1) for i in range(w)]))
                for i in range(h):
                    print('{:<3}'.format(i-1) + ' '.join(['{:.3f}'.format(self.As[global_step][i][j]) for j in range(w)]))
                # with np.printoptions(precision=3, suppress=True):
                #     print('A is: ')
                #     print(self.As[global_step])

            def print_feat():
                with np.printoptions(precision=3, suppress=True):
                    print('feat is: ')
                    print(self.feats[global_step])

            def print_X():
                with np.printoptions(precision=3, suppress=True):
                    print('X is: ')
                    print(self.Xs[global_step])

            def on_click(event):
                if anim.running:
                    anim.event_source.stop()
                    if event.key == 'a':
                        if hasattr(self.robot.policy, 'get_matrix_A'):
                            print_matrix_A()
                        if hasattr(self.robot.policy, 'get_feat'):
                            print_feat()
                        if hasattr(self.robot.policy, 'get_X'):
                            print_X()
                        # if hasattr(self.robot.policy, 'action_values'):
                        #    plot_value_heatmap()
                else:
                    anim.event_source.start()
                anim.running ^= True

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 500, repeat=True)
            anim.running = True

            if output_file is not None:
                # save as video
                # ffmpeg_writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                # writer = ffmpeg_writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                # anim.save(output_file, writer=ffmpeg_writer)

                # save output file as gif if imagemagic is installed
                # anim.save(output_file, writer='imagemagic', fps=12)
                anim.save(output_file, writer='imagemagic')
            else:
                plt.show()
        else:
            raise NotImplementedError
