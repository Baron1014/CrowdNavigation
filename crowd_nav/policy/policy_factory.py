from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.model_predictive_rl import ModelPredictiveRL
from crowd_nav.policy.sarl import SARL

policy_factory['model_predictive_rl'] = ModelPredictiveRL
policy_factory['sarl'] = SARL

