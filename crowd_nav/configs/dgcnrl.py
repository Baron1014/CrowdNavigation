from crowd_nav.configs.config import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig, Config
from crowd_nav.configs import rgl

class EnvConfig(rgl.EnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)


class PolicyConfig(rgl.PolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)
        self.name = 'dgcnrl'

        gcnrl = Config()
        gcnrl.num_layer = 2


class TrainConfig(rgl.TrainConfig):
    def __init__(self, debug=False):
        super(TrainConfig, self).__init__(debug)
