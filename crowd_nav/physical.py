from inference import inference, init
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
import rospy
import cv2
import argparse

x = 0.0
y = 0.0


def callback(data):
    global x
    global y
    x = data.pose.pose.position.x
    y = data.pose.pose.position.y


def main(args):

    rospy.init_node("crowd_nav")

    rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, callback)
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    v_detector, detector, robot, eg, _ = init(args)
    done = False
    idx_frame = 0
    old_vel = None
    try:
        while not done:
            # last_pos = input_coodinate()
            # last_pos = np.array(robot.get_position())
            idx_frame += 1

            vel, accel, done, key = inference(-y, x, old_vel, robot=robot, video_detector=v_detector, detector=detector, env_config=eg, idx_frame=idx_frame)
            old_vel = vel
            msg = Twist()

            if key == 27:
                cv2.destroyAllWindows()
                break

            msg.linear.y = -vel.vx / 10
            msg.linear.x = vel.vy / 10

            pub.publish((msg))
    finally:
        detector.pipe.stop()


if __name__ == "__main__":
    from datetime import datetime
    t = datetime.now()
    video_name = f'{t.year}{t.month}{t.day}{t.hour}{t.minute}{t.second}'

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
    parser.add_argument('--video_output_name', type=str, default=f'{video_name}.avi')
    # camera
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--display", default=True, action="store_true")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--config_deepsort", type=str, default="configs/deep_sort.yaml")
    parser.add_argument("--config_detection", type=str, default="configs/yolov3.yaml")
    parser.add_argument("gx", type=float)
    parser.add_argument("gy", type=float)
    sys_args = parser.parse_args()

    main(sys_args)
