from crowd_nav.configs.config import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig, Config

class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)
        self.robot.sensor = 'RGBD'


class PolicyConfig(BasePolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)
        self.name = 'social_stgcnn'
        self.gcn.X_dim = 32

        self.social_stgcnn = Config()
        self.social_stgcnn.value_network_dims = [32, 100, 100, 1]
        self.social_stgcnn.multiagent_training = True
        self.social_stgcnn.n_stgcnn = 1
        self.social_stgcnn.n_txpcnn = 1
        self.social_stgcnn.stgcn_input_feat = self.gcn.X_dim
        self.social_stgcnn.stgcn_output_feat = 32
        self.social_stgcnn.seq_len = 4
        self.social_stgcnn.kernel_size = 3
        self.social_stgcnn.seq_hidden = 12
        self.social_stgcnn.predict_seq_len = 1



class TrainConfig(BaseTrainConfig):
    def __init__(self, debug=False):
        super(TrainConfig, self).__init__(debug)
        self.trainer.batch_size = 100
        self.train.train_batches = 100
