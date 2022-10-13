from crowd_nav.configs.config import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig, Config


class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)
        self.env.randomize_attributes = True
        self.env.seed = 0
        self.env.env_name = 'CrowdSimDict-v0'  # name of the environment

        self.reward.success_reward = 10
        self.reward.collision_penalty = -20
        # discomfort distance for the front half of the robot
        self.reward.discomfort_dist_front = 0.25
        # discomfort distance for the back half of the robot
        self.reward.discomfort_dist_back = 0.25
        self.reward.discomfort_penalty_factor = 10
        self.reward.gamma = 0.99  # discount factor for rewards

        self.sim.render = True # show GUI for visualization
        self.sim.group_human = False # Group environment: set to true; FoV environment: false

        # cofig for RL ppo
        self.ppo = Config()
        self.ppo.num_mini_batch = 2  # number of batches for ppo
        self.ppo.num_steps = 30  # number of forward steps
        self.ppo.recurrent_policy = True  # use a recurrent policy
        self.ppo.epoch = 5  # number of ppo epochs
        self.ppo.clip_param = 0.2  # ppo clip parameter
        self.ppo.value_loss_coef = 0.5  # value loss coefficient
        self.ppo.entropy_coef = 0.0  # entropy term coefficient
        self.ppo.use_gae = True  # use generalized advantage estimation
        self.ppo.gae_lambda = 0.95  # gae lambda parameter

        # FOV = this values * PI
        self.humans.FOV = 2.
        # a human may change its goal before it reaches its old goal
        self.humans.random_goal_changing = True
        self.humans.goal_change_chance = 0.25
        # a human may change its goal after it reaches its old goal
        self.humans.end_goal_changing = True
        self.humans.end_goal_change_chance = 1.0
        # a human may change its radius and/or v_pref after it reaches its current goal
        self.humans.random_radii = False
        self.humans.random_v_pref = False
        # one human may have a random chance to be blind to other agents at every time step
        self.humans.random_unobservability = False
        self.humans.unobservable_chance = 0.3
        self.humans.random_policy_changing = False

        self.robot.policy = 'srnn'
         # robot FOV = this values * PI
        self.robot.FOV = 2.

        # add noise to observation or not
        self.noise = Config()
        self.noise.add_noise = False
        # uniform, gaussian
        self.noise.type = "uniform"
        self.noise.magnitude = 0.1


class PolicyConfig(BasePolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)
        self.name = 'srnn'

        # SRNN config
        self.SRNN = Config()
        # RNN size
        self.SRNN .human_node_rnn_size = 128  # Size of Human Node RNN hidden state
        self.SRNN .human_human_edge_rnn_size = 256  # Size of Human Human Edge RNN hidden state
        # Input and output size
        self.SRNN .human_node_input_size = 3  # Dimension of the node features
        self.SRNN .human_human_edge_input_size = 2  # Dimension of the edge features
        self.SRNN .human_node_output_size = 256  # Dimension of the node output
        # Embedding size
        self.SRNN .human_node_embedding_size = 64  # Embedding size of node features
        self.SRNN .human_human_edge_embedding_size = 64  # Embedding size of edge features
        # Attention vector dimension
        self.SRNN .attention_size = 64  # Attention size



class TrainConfig(BaseTrainConfig):
    def __init__(self, debug=False):
        super(TrainConfig, self).__init__(debug)

        self.train.freeze_state_predictor = False
        self.train.detach_state_predictor = False
        self.train.reduce_sp_update_frequency = False
        self.train.lr = 4e-5  # learning rate (default: 7e-4)
        self.train.eps = 1e-5  # RMSprop optimizer epsilon
        self.train.max_grad_norm = 0.5  # max norm of gradients
        self.train.train_with_pretend_batch = False
        self.train.output_dir = 'data/dummy'  # the saving directory for train.py
        self.train.overwrite = True  # whether to overwrite the output directory in training
        self.train.resume = False  # resume training from an existing checkpoint or not
        self.train.num_processes = 12 # how many training CPU processes to use
        self.train.num_threads = 1  # number of threads used for intraop parallelism on CPU
        self.train.num_env_steps = 10e6  # number of environment steps to train: 10e6 for holonomic, 20e6 for unicycle
        self.train.use_linear_lr_decay = False  # use a linear schedule on the learning rate: True for unicycle, False for holonomic
    
