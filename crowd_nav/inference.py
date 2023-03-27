import logging
import argparse
import importlib.util
import os
import torch
import numpy as np
import gym
import time
import cv2
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.action import ActionXY
from configs import logger
from ros_detection import robot_detection
from crowd_sim.envs.utils.human import Human


def init(args):
    # camera setting
    video_detector, detector = robot_detection.camera_setting(args)

    log_file = os.path.join(args.model_dir, "inference.log")
    logger.log_setting(args, log_file)
    # configure logging and device
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s, %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info("Using device: %s", device)

    if args.model_dir is not None:
        if args.config is not None:
            config_file = args.config
        else:
            config_file = os.path.join(args.model_dir, "config.py")

        model_weights = os.path.join(args.model_dir, "best_val.pth")
        logging.info("Loaded RL weights with best VAL")

    else:
        config_file = args.config

    spec = importlib.util.spec_from_file_location("config", config_file)
    if spec is None:
        parser.error("Config file not found.")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # configure policy
    policy_config = config.PolicyConfig(args.debug)
    policy = policy_factory[policy_config.name]()

    policy.configure(policy_config)
    policy.set_device(device)
    if policy.trainable:
        if args.model_dir is None:
            parser.error("Trainable policy must be specified with a model weights directory")
        policy.load_model(model_weights)

    # configure environment
    env_config = config.EnvConfig(args.debug)
    if args.gx is not None and args.gy is not None:
        env_config.robot.gx = args.gx
        env_config.robot.gy = args.gy

    if args.human_num is not None:
        env_config.sim.human_num = args.human_num
    env = gym.make("CrowdSim-v0")
    env.configure(env_config)

    robot = Robot(env_config, "robot")
    env.set_robot(robot)
    robot.time_step = env.time_step
    robot.set_policy(policy)
    train_config = config.TrainConfig(args.debug)
    epsilon_end = train_config.train.epsilon_end
    if not isinstance(robot.policy, ORCA):
        robot.policy.set_epsilon(epsilon_end)

    policy.set_phase(args.phase)
    policy.set_device(device)
    policy.set_env(env)

    _ = env.reset(args.phase, args.test_case)
    robot.print_info()
    env = None
    return video_detector, detector, robot, env_config, env


def inference(pos_x, pos_y, old_vel, robot=None, video_detector=None, detector=None, env_config=None, idx_frame=0, env=None):
    done = False
    start = time.time()
    if pos_x is None and pos_y is None:
        last_pos = np.array(robot.get_position())
    else:
        last_pos = np.array([pos_x, pos_y])

    robot.set_position(last_pos)
    reaching_goal = np.linalg.norm(last_pos - np.array(robot.get_goal_position())) < robot.radius
    try:
        print(f"goal dis:{last_pos} - {np.array(robot.get_goal_position())}")
        if reaching_goal:
            done = True
            action = ActionXY(0, 0)
        else:
            position, velocity, key = robot_detection.camera_detection(video_detector, detector, start, idx_frame)
            ob = compute_observation(last_pos, position, velocity, env_config)
            action = robot.act(ob)
            if env is not None:
                _, _, _, info = env.step(action)
    except:
        out = video_detector.get_writer()
        out.release()
        raise EOFError("inference error!!")

    if old_vel is None:
        accel = action
    else:
        accel = ActionXY(action[0] - old_vel[0], action[1] - old_vel[1])
    # logging.info('Robot position: {}, Velocity: {}), Speed: {:.2f}'.format(last_pos, action, np.linalg.norm(action)))

    return action, accel, done, key


def main(args):
    video_detector, detector, robot, eg, env = init(args)
    idx_frame = 0
    done = False
    p_x, p_y = None, None
    old_vel = None
    while not done:
        idx_frame += 1
        vel, accel, done, key = inference(p_x, p_y, old_vel, robot, video_detector, detector, eg, idx_frame, env)
        old_vel = vel

        # End loop once video finishes
        if key == 27:
            cv2.destroyAllWindows()
            break


def compute_observation(robot_pos, position, velocity, config):
    ob = []
    for i in range(len(position)):
        human = Human(i + 1, config, "humans")
        human_xy = [position[i][0], position[i][2]]
        pos, vel = robot_pos + human_xy, velocity[i]
        human.set(pos[0], pos[1], 0, 0, vel[0], vel[1], 0.1)
        ob.append(human.get_id_observable_state())

    return ob


if __name__ == "__main__":
    from datetime import datetime

    t = datetime.now()
    video_name = f"{t.year}{t.month}{t.day}{t.hour}"

    parser = argparse.ArgumentParser("Parse configuration file")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("-m", "--model_dir", type=str, default="data/inference")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("-v", "--visualize", default=True, action="store_true")
    parser.add_argument("--phase", type=str, default="test")
    parser.add_argument("-c", "--test_case", type=int, default=None)
    parser.add_argument("--video_file", type=str, default=None)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--traj", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--human_num", type=int, default=5)
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--bag_file", type=str, default="/data/20221024_142540.bag")
    parser.add_argument("--video_output_dir", type=str, default="data/video")
    parser.add_argument("--video_output_name", type=str, default=f"{video_name}.avi")
    # camera
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--display", default=True, action="store_true")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--config_deepsort", type=str, default="configs/deep_sort.yaml")
    parser.add_argument("--config_detection", type=str, default="configs/yolov3.yaml")
    parser.add_argument("--gx", type=float, default=0.0)
    parser.add_argument("--gy", type=float, default=16.5)

    sys_args = parser.parse_args()

    main(sys_args)
