from rmp import RMPRoot
from rmp_leaf import CollisionAvoidance, GoalAttractorUni, TransitionArbitraryPoint

from geometry_msgs.msg import PoseWithCovarianceStamped

import rospy
import numpy as np
from numpy.linalg import norm

import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import fcl

import time

from rmp import RMPNode, RMPRoot, RMPLeaf
import numpy as np
from numpy.linalg import norm, inv
from scipy.spatial.transform import Rotation as Rot

from inference import inference, init
from geometry_msgs.msg import Twist

robotPos = np.array([0.0, 0.0])


class CrowdAvoidanceLeaf(RMPLeaf):
    """
    Goal Attractor RMP leaf
    """

    def __init__(self, name, parent, v_detector, detector, robot, eg, idx_frame):

        psi = lambda y: y
        J = lambda y: np.eye(2)
        J_dot = lambda y, y_dot: np.zeros((2, 2))

        def RMP_func(x, x_dot):
            G = np.eye(2) * 1.0
            M = G

            #x_o = x_dot[0]
            #y_o = x_dot[1]
            #x_dot[0] = -y_o
            #x_dot[1] = x_o

            _, f, _, _ = inference(x[0][0], x[1][0], x_dot, robot=robot, video_detector=v_detector, detector=detector, env_config=eg, idx_frame=idx_frame)
            #x_c = a[0]
            #y_c = a[1]

            #f = np.array([[0.0], [0.0]])
            #f[0][0] = y_c
            #f[1][0] = -x_c
            return (f, M)

        RMPLeaf.__init__(self, name, parent, None, psi, J, J_dot, RMP_func)


def main(args):

    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    rospy.init_node("rmpflow")

    v_detector, detector, robot, eg, _ = init(args)
    done = False
    idx_frame = 0
    old_vel = None

    LeftWall = fcl.Box(1.0, 9.86, 1e5)
    RightWall = fcl.Box(1.0, 7.65, 1e5)
    TopWall = fcl.Box(2.98, 1.0, 1e5)
    CollisionBox = fcl.Box(0.22, 0.1, 1e5)

    robotBox = fcl.Box(0.5, 0.8, 1e5)

    T = fcl.Transform(np.array([-1.3, 3.93, 0.0]))
    collisionLeftWall = fcl.CollisionObject(LeftWall, T)
    T = fcl.Transform(np.array([2.5, 4.825, 0.0]))
    collisionRightWall = fcl.CollisionObject(RightWall, T)
    T = fcl.Transform(np.array([0.69, 9.75, 0.0]))
    collisionTopWall = fcl.CollisionObject(TopWall, T)
    T = fcl.Transform(np.array([1.11, 1.05, 0.0]))
    collision = fcl.CollisionObject(CollisionBox, T)

    x = np.array([0.0, 0.0])
    x_dot = np.array([0.0, 0.0])
    x_g = np.array([0.0, 9.0])
    # x_g = np.array([2.12, 4.619])

    state_0 = np.concatenate((x, x_dot), axis=None)

    # --------------------------------------------
    def getWayPoint(x, count):
        wayPoint = np.array([[0.0, 9.0]])

        if norm(wayPoint[count] - x) < 0.20:
            count = count + 1

        if count >= wayPoint.shape[0]:
            count = wayPoint.shape[0] - 1
        return (wayPoint[count], count)

    # dynamics
    def dynamics(r, state):
        state = state.reshape(2, -1)
        x = state[0]
        x_dot = state[1]
        x_ddot = r.solve(x, x_dot)
        # if x_ddot[0]>0.5:
        #    x_ddot[0]=0.5
        # elif x_ddot[0]<-0.5:
        #    x_ddot[0]=-0.5

        # if x_ddot[1]>0.5:
        #    x_ddot[1]=0.5
        # elif x_ddot[1]<-0.5:
        #    x_ddot[1]=-0.5

        state_dot = np.concatenate((x_dot, x_ddot), axis=None)
        return state_dot

    # ---------------------------------------------
    def stateTrans(state):
        ret = np.array([0.0, 0.0])
        ret[1] = state[0]
        ret[0] = -(state[1])

        return ret

    # ----------------------------------------------
    def commandTrans(vel):
        ret = np.array([0.0, 0.0])
        ret[1] = -vel[0]
        ret[0] = vel[1]

        return ret

    # -------------------------------------------

    def callback(data):
        global robotPos
        robotPos = np.array([data.pose.pose.position.x, data.pose.pose.position.y])

    state = state_0
    stateListX = []
    stateListY = []
    # solve the diff eq
    count = 0

    rate = rospy.Rate(100)
    rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, callback)

    for i in range(5000):
        global robotPos

        lidarPos = stateTrans(robotPos)

        state[0] = lidarPos[0]
        state[1] = lidarPos[1]

        print("\n")
        #print(state)

        start = time.time()
        T = fcl.Transform(np.array([state[0], state[1], 0]))
        robotRMP = fcl.CollisionObject(robotBox, T)

        distInfo = fcl.DistanceResult()
        distInfo2 = fcl.DistanceResult()
        distInfo3 = fcl.DistanceResult()
        distInfo4 = fcl.DistanceResult()
        request = fcl.DistanceRequest()
        fcl.distance(collision, robotRMP, request, distInfo)
        fcl.distance(collisionRightWall, robotRMP, request, distInfo2)
        fcl.distance(collisionLeftWall, robotRMP, request, distInfo3)
        fcl.distance(collisionTopWall, robotRMP, request, distInfo4)

        # set rmp root
        Root = RMPRoot("Root")

        #set collision point transition
        T1 = TransitionArbitraryPoint("T1", Root, distInfo.nearest_points[1][:2])
        C1 = CollisionAvoidance("C1", T1, None, epsilon=0.01, c=distInfo.nearest_points[0][:2], R=0.300)

        T2 = TransitionArbitraryPoint("T2", Root, distInfo2.nearest_points[1][:2])
        C2 = CollisionAvoidance("C2", T2, None, epsilon=0.01, c=distInfo2.nearest_points[0][:2], R=0.300)

        T3 = TransitionArbitraryPoint("T3", Root, distInfo3.nearest_points[1][:2])
        C3 = CollisionAvoidance("C3", T3, None, epsilon=0.01, c=distInfo3.nearest_points[0][:2], R=0.300)
        T4 = TransitionArbitraryPoint("T4", Root, distInfo3.nearest_points[1][:2])
        C4 = CollisionAvoidance("C4", T4, None, epsilon=0.01, c=distInfo3.nearest_points[0][:2], R=0.300)
        (x_g, count) = getWayPoint(state[:2], count)
        #leafG = GoalAttractorUni("goal_attractor", Root, x_g)

        leafCrowd = CrowdAvoidanceLeaf("crowd", Root, v_detector, detector, robot, eg, i + 1)

        state_dot = dynamics(Root, state)
        state[0] = state[0] + state_dot[0] * 0.05
        state[1] = state[1] + state_dot[1] * 0.05
        state[2] = state[2] + state_dot[2] * 0.05
        state[3] = state[3] + state_dot[3] * 0.05

        end = time.time()
        if state[2] > 0.1:
            state[2] = 0.1
        elif state[2] < -0.1:
            state[2] = -0.1

        if state[3] > 0.1:
            state[3] = 0.1
        elif state[3] < -0.1:
            state[3] = -0.1

        vel = commandTrans(state[2:])
        cmd = Twist()
        cmd.linear.x = vel[0]
        cmd.linear.y = vel[1]
        pub.publish(cmd)

        stateListX.append(state[0])
        stateListY.append(state[1])

        plt.cla()
        plt.xlim([-2, 4])
        plt.ylim([-2, 10])
        plt.plot(stateListX, stateListY)
        rect = plt.Rectangle((state[0] - 0.25, state[1] - 0.4), 0.5, 0.8, ec="k")
        plt.gca().add_artist(rect)
        plt.plot(distInfo2.nearest_points[0][0], distInfo2.nearest_points[0][1], "go")
        plt.plot(distInfo2.nearest_points[1][0], distInfo2.nearest_points[1][1], "go")

        # wall
        rect = plt.Rectangle((-1.8, -1), 1, 9.86, fc="black", ec="black")
        plt.gca().add_artist(rect)
        rect = plt.Rectangle((2, 1), 1, 7.65, fc="black", ec="black")
        plt.gca().add_artist(rect)
        rect = plt.Rectangle((-0.8, 9.25), 2.98, 1, fc="black", ec="black")
        plt.gca().add_artist(rect)

        rect = plt.Rectangle((1, 1), 0.22, 0.1, fc="black", ec="black")
        plt.gca().add_artist(rect)
        plt.gca().set_aspect("equal", "box")
        plt.draw()
        plt.pause(0.001)


if __name__ == "__main__":
    from datetime import datetime

    t = datetime.now()
    video_name = f"{t.year}{t.month}{t.day}{t.hour}{t.minute}{t.second}"

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
    parser.add_argument("gx", type=float)
    parser.add_argument("gy", type=float)
    sys_args = parser.parse_args()

    main(sys_args)
