from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.model_predictive_rl import ModelPredictiveRL


policy_factory['gcn'] = ModelPredictiveRL

