import sys
import logging
import os
import shutil
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from pytorchBaselines.a2c_ppo_acktr import algo, utils

from pytorchBaselines.a2c_ppo_acktr.envs import make_vec_envs
from pytorchBaselines.a2c_ppo_acktr.model import Policy
from pytorchBaselines.a2c_ppo_acktr.storage import RolloutStorage


from crowd_nav.configs.srnn import EnvConfig, TrainConfig, PolicyConfig
from crowd_sim import *

def main():
	training_config = TrainConfig()

	# save policy to output_dir
	if os.path.exists(training_config.train.output_dir) and training_config.train.overwrite: # if I want to overwrite the directory
		shutil.rmtree(training_config.train.output_dir)  # delete an entire directory tree

	if not os.path.exists(training_config.train.output_dir):
		os.makedirs(training_config.train.output_dir)

	shutil.copytree('crowd_nav/configs', os.path.join(training_config.train.output_dir, 'configs'))


	# configure logging
	log_file = os.path.join(training_config.train.output_dir, 'output.log')
	mode = 'a' if training_config.train.resume else 'w'
	file_handler = logging.FileHandler(log_file, mode=mode)
	stdout_handler = logging.StreamHandler(sys.stdout)
	level = logging.INFO
	logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
						format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

	env_config = EnvConfig()
	torch.manual_seed(env_config.env.seed)
	torch.cuda.manual_seed_all(env_config.env.seed)

	device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")

	# if training_config.training.cuda and torch.cuda.is_available():
	# 	if config.training.cuda_deterministic:
	# 		# reproducible but slower
	# 		torch.backends.cudnn.benchmark = False
	# 		torch.backends.cudnn.deterministic = True
	# 	else:
	# 		# not reproducible but faster
	# 		torch.backends.cudnn.benchmark = True
	# 		torch.backends.cudnn.deterministic = False


	torch.set_num_threads(training_config.train.num_threads)
	


	logging.info('Create other envs with new settings')


	# For fastest training: use GRU
	env_name = env_config.env.env_name
	recurrent_cell = 'GRU'

	if env_config.sim.render:
		fig, ax = plt.subplots(figsize=(7, 7))
		ax.set_xlim(-6, 6)
		ax.set_ylim(-6, 6)
		ax.set_xlabel('x(m)', fontsize=16)
		ax.set_ylabel('y(m)', fontsize=16)
		plt.ion()
		plt.show()
	else:
		ax = None


	if env_config.sim.render:
		training_config.train.num_processes = 1
		env_config.ppo.num_mini_batch = 1

	# create a manager env
	envs = make_vec_envs(env_name, env_config.env.seed, training_config.train.num_processes,
						 env_config.reward.gamma, None, device, False, config=env_config, ax=ax)


	policy_config = PolicyConfig()
	actor_critic = Policy(
		envs.observation_space.spaces, # pass the Dict into policy to parse
		base_kwargs=env_config,
		base=env_config.robot.policy)
	actor_critic.base.configure(policy_config, training_config)
	actor_critic.set_action_space(envs.action_space)

	rollouts = RolloutStorage(env_config.ppo.num_steps,
							  training_config.train.num_processes,
							  envs.observation_space.spaces,
							  envs.action_space,
							  policy_config.SRNN.human_node_rnn_size,
							  policy_config.SRNN.human_human_edge_rnn_size,
							  recurrent_cell_type=recurrent_cell)

	if training_config.train.resume: #retrieve the model if resume = True
		load_path = training_config.train.load_path
		actor_critic.load_state_dict(torch.load(load_path))
		print("Loaded the following checkpoint:", load_path)


	# allow the usage of multiple GPUs to increase the number of examples processed simultaneously
	nn.DataParallel(actor_critic).to(device)


	agent = algo.PPO(
		actor_critic,
		env_config.ppo.clip_param,
		env_config.ppo.epoch,
		env_config.ppo.num_mini_batch,
		env_config.ppo.value_loss_coef,
		env_config.ppo.entropy_coef,
		lr=training_config.train.lr,
		eps=training_config.train.eps,
		max_grad_norm=training_config.train.max_grad_norm)



	obs = envs.reset()
	if isinstance(obs, dict):
		for key in obs:
			rollouts.obs[key][0].copy_(obs[key])
	else:
		rollouts.obs[0].copy_(obs)

	rollouts.to(device)

	episode_rewards = deque(maxlen=100)

	start = time.time()
	num_updates = int(
		training_config.train.num_env_steps) // env_config.ppo.num_steps // training_config.train.num_processes

	for j in range(num_updates):

		if training_config.train.use_linear_lr_decay:
			utils.update_linear_schedule(
				agent.optimizer, j, num_updates, training_config.train.lr)

		for step in range(env_config.ppo.num_steps):
			# Sample actions
			with torch.no_grad():

				rollouts_obs = {}
				for key in rollouts.obs:
					rollouts_obs[key] = rollouts.obs[key][step]
				rollouts_hidden_s = {}
				for key in rollouts.recurrent_hidden_states:
					rollouts_hidden_s[key] = rollouts.recurrent_hidden_states[key][step]
				value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
					rollouts_obs, rollouts_hidden_s,
					rollouts.masks[step])

			if env_config.sim.render:
				envs.render()
			# Obser reward and next obs
			obs, reward, done, infos = envs.step(action)
			# print(done)

			for info in infos:
				# print(info.keys())
				if 'episode' in info.keys():
					episode_rewards.append(info['episode']['r'])

			# If done then clean the history of observations.
			masks = torch.FloatTensor(
				[[0.0] if done_ else [1.0] for done_ in done])
			bad_masks = torch.FloatTensor(
				[[0.0] if 'bad_transition' in info.keys() else [1.0]
				 for info in infos])
			rollouts.insert(obs, recurrent_hidden_states, action,
							action_log_prob, value, reward, masks, bad_masks)

		with torch.no_grad():
			rollouts_obs = {}
			for key in rollouts.obs:
				rollouts_obs[key] = rollouts.obs[key][-1]
			rollouts_hidden_s = {}
			for key in rollouts.recurrent_hidden_states:
				rollouts_hidden_s[key] = rollouts.recurrent_hidden_states[key][-1]
			next_value = actor_critic.get_value(
				rollouts_obs, rollouts_hidden_s,
				rollouts.masks[-1]).detach()




		rollouts.compute_returns(next_value, env_config.ppo.use_gae, env_config.reward.gamma,
								 env_config.ppo.gae_lambda, training_config.train.use_proper_time_limits)

		value_loss, action_loss, dist_entropy = agent.update(rollouts)

		rollouts.after_update()

		# save the model for every interval-th episode or for the last epoch
		if (j % training_config.train.save_interval == 0
			or j == num_updates - 1) :
			save_path = os.path.join(training_config.train.output_dir, 'checkpoints')
			if not os.path.exists(save_path):
				os.mkdir(save_path)

			# if you normalized the observation, you may also want to save rms
			# torch.save([
			# 	actor_critic,
			# 	getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
			# ], os.path.join(save_path, '%.5i'%j + ".pt"))

			torch.save(actor_critic.state_dict(), os.path.join(save_path, '%.5i' % j + ".pt"))

		if j % training_config.train.log_interval == 0 and len(episode_rewards) > 1:
			total_num_steps = (j + 1) * training_config.train.num_processes * env_config.ppo.num_steps
			end = time.time()
			print(
				"Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward "
				"{:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
					.format(j, total_num_steps,
							int(total_num_steps / (end - start)),
							len(episode_rewards), np.mean(episode_rewards),
							np.median(episode_rewards), np.min(episode_rewards),
							np.max(episode_rewards), dist_entropy, value_loss,
							action_loss))

			df = pd.DataFrame({'misc/nupdates': [j], 'misc/total_timesteps': [total_num_steps],
							   'fps': int(total_num_steps / (end - start)), 'eprewmean': [np.mean(episode_rewards)],
							   'loss/policy_entropy': dist_entropy, 'loss/policy_loss': action_loss,
							   'loss/value_loss': value_loss})

			if os.path.exists(os.path.join(training_config.train.output_dir, 'progress.csv')) and j > 20:
				df.to_csv(os.path.join(training_config.train.output_dir, 'progress.csv'), mode='a', header=False, index=False)
			else:
				df.to_csv(os.path.join(training_config.train.output_dir, 'progress.csv'), mode='w', header=True, index=False)




if __name__ == '__main__':
	main()
