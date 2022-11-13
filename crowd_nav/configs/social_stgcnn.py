from crowd_nav.configs.config import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig, Config
from crowd_nav.configs import rgl

class EnvConfig(rgl.EnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)
        self.robot.obs_len = 4


class PolicyConfig(rgl.PolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)
        self.name = 'social_stgcnn'


class TrainConfig(rgl.TrainConfig):
    def __init__(self, debug=False):
        super(TrainConfig, self).__init__(debug)
        self.trainer.batch_size = 1
        self.train.train_batches = 100
