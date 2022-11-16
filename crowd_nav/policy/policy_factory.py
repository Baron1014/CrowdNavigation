from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.model_predictive_rl import ModelPredictiveRL
from crowd_nav.policy.sarl import SARL
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
# from crowd_nav.policy.dgcnrl import DGCNRL
from crowd_nav.policy.social_stgcnn import SSTGCNN_RL


policy_factory['model_predictive_rl'] = ModelPredictiveRL
policy_factory['sarl'] = SARL
policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
# policy_factory['dgcnrl'] = DGCNRL
policy_factory['social_stgcnn'] = SSTGCNN_RL
