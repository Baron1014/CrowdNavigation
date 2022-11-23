import argparse
import os
import shutil
import logging
import importlib.util
import torch
import gym
import copy
import wandb
from configs import logger
from configs import config as global_config
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import MPRLTrainer, VNRLTrainer, GRAPHTrainer, TGRLTrainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory


def set_random_seeds(seed):
    """
    Sets the random seeds for pytorch cpu and gpu
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None

class WandbWriter:
    def __init__(self, args, config):
        self.use_writer = args.wandb
        if self.use_writer:
            wandb.init(project=config.wan.project, entity=config.wan.entity, config=args, name=args.wandb_display_name)
            self.writer = wandb
        else:
            self.writer = None
        
    def log(self, info, step):
        if self.writer is not None:
            self.writer.log(info, step=step)
    
    def summary(self, key, value):
        self.writer.run.summary[key] = value
    
    def finish(self):
        if self.writer is not None:
            wandb.finish()
    

def main(args):
    set_random_seeds(args.randomseed)
    # configure paths
    train_cg = global_config.BaseTrainConfig()
    writer = WandbWriter(args, train_cg)

    make_new_dir = True
    if os.path.exists(args.output_dir):
        if args.overwrite:
            shutil.rmtree(args.output_dir)
        else:
            key = input('Output directory already exists! Overwrite the folder? (y/n)')
            if key == 'y' and not args.resume:
                shutil.rmtree(args.output_dir)
            else:
                make_new_dir = False
    if make_new_dir:
        pocliy_config = {
            'social_stgcnn': 'configs/social_stgcnn.py',
            'dgcnrl': 'configs/dgcnrl.py',
            'sstgcn': 'configs/sstgcn.py',
            'rgl': 'configs/rgl.py',
            'sarl': 'configs/sarl.py', 
            'cadrl': 'configs/cadrl.py',
            'lstm_rl': 'configs/lstm_rl.py'
        }

        os.makedirs(args.output_dir)
        shutil.copy(pocliy_config[args.policy], os.path.join(args.output_dir, 'config.py'))


    cg = os.path.join(args.output_dir, 'config.py')
    log_file = os.path.join(args.output_dir, 'output.log')
    il_weight_file = os.path.join(args.output_dir, 'il_model.pth')
    rl_weight_file = os.path.join(args.output_dir, 'rl_model.pth')

    spec = importlib.util.spec_from_file_location('config', cg)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # configure logging
    logger.log_setting(args, log_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure policy
    policy_config = config.PolicyConfig()
    policy = policy_factory[policy_config.name]()
    if not policy.trainable:
        parser.error('Policy has to be trainable')
    policy.configure(policy_config)
    policy.set_device(device)

    # configure environment
    env_config = config.EnvConfig(args.debug)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    robot.time_step = env.time_step
    env.set_robot(robot)

    # read training parameters
    train_config = config.TrainConfig(args.debug)
    rl_learning_rate = train_config.train.rl_learning_rate
    train_batches = train_config.train.train_batches
    train_episodes = train_config.train.train_episodes
    sample_episodes = train_config.train.sample_episodes
    target_update_interval = train_config.train.target_update_interval
    evaluation_interval = train_config.train.evaluation_interval
    capacity = train_config.train.capacity
    epsilon_start = train_config.train.epsilon_start
    epsilon_end = train_config.train.epsilon_end
    epsilon_decay = train_config.train.epsilon_decay
    checkpoint_interval = train_config.train.checkpoint_interval

    # configure trainer and explorer
    memory = ReplayMemory(capacity)
    model = policy.get_model()
    batch_size = train_config.trainer.batch_size
    optimizer = train_config.trainer.optimizer
    if policy_config.name == 'model_predictive_rl':
        trainer = MPRLTrainer(model, policy.state_predictor, memory, device, policy, writer, batch_size, optimizer, env.human_num,
                              reduce_sp_update_frequency=train_config.train.reduce_sp_update_frequency,
                              freeze_state_predictor=train_config.train.freeze_state_predictor,
                              detach_state_predictor=train_config.train.detach_state_predictor,
                              share_graph_model=policy_config.model_predictive_rl.share_graph_model)
    elif policy_config.name == 'dgcnrl':
        trainer = GRAPHTrainer(model, memory, device, policy, batch_size, optimizer, writer)
    elif policy_config.name == 'social_stgcnn':
        trainer = TGRLTrainer(model, memory, device, policy, batch_size, optimizer, writer)
    else:
        trainer = VNRLTrainer(model, memory, device, policy, batch_size, optimizer, writer)
    explorer = Explorer(env, robot, device, writer, memory, policy.gamma, target_policy=policy)

    # imitation learning
    # if args.resume:
    #     if not os.path.exists(rl_weight_file):
    #         logging.error('RL weights does not exist')
    #     model.load_state_dict(torch.load(rl_weight_file))
    #     rl_weight_file = os.path.join(args.output_dir, 'resumed_rl_model.pth')
    #     logging.info('Load reinforcement learning trained weights. Resume training')
    if os.path.exists(il_weight_file):
        model.load_state_dict(torch.load(il_weight_file, map_location=device), False)
        # model.load_state_dict(torch.load(il_weight_file), False)
        logging.info('Load imitation learning trained weights.')
    else:
        il_episodes = train_config.imitation_learning.il_episodes
        il_policy = train_config.imitation_learning.il_policy
        il_epochs = train_config.imitation_learning.il_epochs
        il_learning_rate = train_config.imitation_learning.il_learning_rate
        trainer.set_learning_rate(il_learning_rate)
        if robot.visible:
            safety_space = 0
        else:
            safety_space = train_config.imitation_learning.safety_space
        il_policy = policy_factory[il_policy]()
        il_policy.multiagent_training = policy.multiagent_training
        il_policy.safety_space = safety_space
        robot.set_fov(2) # for imitation learning of target policy use fov
        robot.set_policy(il_policy)
        explorer.run_k_episodes(il_episodes, 'train', update_memory=True, imitation_learning=True)
        explorer.log('IL', global_step=0)
        trainer.optimize_epoch(il_epochs)
        policy.save_model(il_weight_file)
        logging.info('Finish imitation learning. Weights saved.')
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)

    trainer.update_target_model(model)

    # reinforcement learning
    policy.set_env(env)
    robot.set_policy(policy)
    robot.set_fov(env_config.robot.FOV)
    robot.print_info()
    trainer.set_learning_rate(rl_learning_rate)
    # fill the memory pool with some RL experience
    if args.resume:
        robot.policy.set_epsilon(epsilon_end)
        explorer.run_k_episodes(100, 'train', update_memory=True, episode=0)
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
    episode = 0
    best_val_reward = -1
    best_val_model = None

    # evaluate the model after imitation learning
    if episode % evaluation_interval == 0:
        logging.info('Evaluate the model instantly after imitation learning on the validation cases')
        explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)
        explorer.log('val', episode)

    episode = 0
    while episode < train_episodes:
        if args.resume:
            epsilon = epsilon_end
        else:
            if episode < epsilon_decay:
                epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
            else:
                epsilon = epsilon_end
        robot.policy.set_epsilon(epsilon)

        # sample k episodes into memory and optimize over the generated memory
        explorer.run_k_episodes(sample_episodes, 'train', update_memory=True, episode=episode)
        explorer.log('train', episode)

        trainer.optimize_batch(train_batches, episode)
        episode += 1

        if episode % target_update_interval == 0:
            trainer.update_target_model(model)
        # evaluate the model
        if episode % evaluation_interval == 0:
            _, _, _, reward, _ = explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)
            explorer.log('val', episode)

            if episode % checkpoint_interval == 0 and reward > best_val_reward:
                best_val_reward = reward
                best_val_model = copy.deepcopy(policy.get_state_dict())
                torch.save(best_val_model, os.path.join(args.output_dir, 'best_val.pth'))
                logging.info('Save the best val model with the reward: {}'.format(best_val_reward))

        if episode != 0 and episode % checkpoint_interval == 0:
            current_checkpoint = episode // checkpoint_interval - 1
            save_every_checkpoint_rl_weight_file = rl_weight_file.split('.')[0] + '_' + str(current_checkpoint) + '.pth'
            policy.save_model(save_every_checkpoint_rl_weight_file)

    # # test with the best val model
    if best_val_model is not None:
        policy.load_state_dict(best_val_model)
        torch.save(best_val_model, os.path.join(args.output_dir, 'best_val.pth'))
        logging.info('Save the best val model with the reward: {}'.format(best_val_reward))
    explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode, print_failure=True)
    explorer.log('test', episode)

    writer.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--randomseed', type=int, default=17)
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--wandb_display_name', '-wdn', type=str, default=None)
    parser.add_argument('--policy', type=str, default='social_stgcnn')

    sys_args = parser.parse_args()

    main(sys_args)
