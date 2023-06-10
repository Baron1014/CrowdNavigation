import pyrealsense2 as rs
import cv2
import os
import copy
import numpy as np
import time

# from persondetection import DetectorAPI
from itertools import product
from detector import build_detector
from deep_sort import build_tracker
from deepsort_utils.log import get_logger
from deepsort_utils.draw import draw_boxes, compute_color_for_labels
import torch
import warnings


class BagVis:
    def __init__(self, sys_args, config, file, repeat=True, threshold=0.7, maxm=16):
        """
        repeat: Image or Video
            True: Video
            False: Image
        """
        path = os.path.dirname(os.path.realpath(__file__))
        self.pipe = rs.pipeline()
        self.config = rs.config()
        self.threshold = threshold

        self.maxm = maxm
        self.args = sys_args
        self.cfg = config
        rs.align(rs.stream.color)
        rs.config.enable_device_from_file(self.config, path + file, repeat_playback=repeat)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # profile = self.pipe.start(self.config)
        # self.playback = profile.get_device().as_playback()
        # self.playback.set_real_time(False)

    def get_video_detector(self, output):
        return VideoVis(self.cfg, self.args, output=output)

    def get_color_img(self, frame):
        color_rs = frame.get_color_frame()
        img = np.asanyarray(color_rs.get_data())

        return img

    def get_depth_img(self, frame):
        depth_rs = self.get_depth_frames(frame)
        depth = np.asanyarray(depth_rs.get_data())
        dimg_gray = cv2.convertScaleAbs(depth, alpha=255 / (self.maxm * 1000))
        depth_colored = cv2.applyColorMap(dimg_gray, cv2.COLORMAP_BONE)

        return depth_colored

    def get_depth_frames(self, frame):
        return frame.get_depth_frame()

    def detect_color(self, output=False):
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = self.pipe.start(self.config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        align_to = rs.stream.color
        align = rs.align(align_to)
        try:
            detector = VideoVis(self.cfg, self.args, output=output)
            idx_frame = 0
            while True:
                idx_frame += 1
                start = time.time()
                going, frame = self.pipe.try_wait_for_frames(timeout_ms=20000)
                playback.pause()
                if going is False:
                    break
                # align depth to color
                frame = align.process(frame)
                bgr_img = self.get_color_img(frame)
                # deepsort
                rgb_im = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                bbox_xywh, cls_conf, cls_ids = detector.detector(bgr_img)
                # select person class
                mask = cls_ids == 0

                bbox_xywh = bbox_xywh[mask]
                # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
                bbox_xywh[:, 3:] *= 1.2
                cls_conf = cls_conf[mask]

                # do tracking
                outputs = detector.deepsort.update(bbox_xywh, cls_conf, bgr_img)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_tlwh = []
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    rgb_im = draw_boxes(rgb_im, bbox_xyxy, identities)

                    for bb_xyxy in bbox_xyxy:
                        bbox_tlwh.append(detector.deepsort._xyxy_to_tlwh(bb_xyxy))

                    depth_frame = self.get_depth_frames(frame)
                    camera_coor = detector.get_depth_infor(depth_frame, bbox_xyxy)
                    velocity = detector.get_velocity(identities, camera_coor, time.time() - start, idx_frame)
                    self.draw_information(rgb_im, identities, velocity, bbox_xyxy, camera_coor)

                if self.args.display:
                    cv2.imshow("test", rgb_im)
                    key = cv2.waitKey(1)

                # End loop once video finishes
                playback.resume()
                if key == 27:
                    cv2.destroyAllWindows()
                    break

            if detector.writer:
                detector.writer.release()
        finally:
            self.pipe.stop()

    def draw_information(self, img, persons, velocitys, bb_boxs, coordinates):
        for i in range(len(persons)):
            velocity, box, camera_coordinate = velocitys[i], bb_boxs[i], coordinates[i]
            person = persons[i]
            color = compute_color_for_labels(person)
            position_label = f"Position:({camera_coordinate[0]:5.2f} {camera_coordinate[2]:5.2f})"
            velocity_label = f"Velocity:({velocity[0]:5.2f} {velocity[1]:5.2f}) m/s"
            p_size = cv2.getTextSize(position_label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            v_size = cv2.getTextSize(velocity_label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (box[0] - 3, box[1] - 2 * (p_size[1] + v_size[1])), (box[0] + v_size[0], box[1]), color, -1)
            cv2.putText(img, position_label, (box[0], box[1] - p_size[1] - v_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)  # (75,0,130),
            cv2.putText(img, velocity_label, (box[0], box[1] - (v_size[1] // 2)), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)  # (75,0,130),


class BasicDetector:
    def __init__(self, cfg, args):
        self.max_count = 0
        self.max_acc = 0
        self.max_avg_acc = 0
        # self.odapi = DetectorAPI()
        self.threshold = 0.7
        self.old_time = None
        self.old_coordinate = None
        self.person_coordinate = dict()
        # deepsort
        self.cfg = cfg
        self.args = args
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        # if args.display:
        # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)

    def get_center_depth_value(self, depth, center_x, center_y):
        buffer = 10
        all_x, all_y = list(), list()
        for i, j in product(range(center_x - buffer, center_x + buffer), range(center_y - buffer, center_y + buffer)):
            all_x.append(i)
            all_y.append(j)
        all_depth = list(map(depth.get_distance, all_x, all_y))
        filled_zero = list(filter(lambda x: x != 0, all_depth))

        return min(filled_zero) if len(filled_zero) > 0 else 0

    def get_velocity(self, persons, current_coor, delta_time, idx_frame):
        v = []
        for i in range(len(persons)):
            person = persons[i]
            current = current_coor[i] + [idx_frame]
            self.old_coordinate = self.get_person_coordinate(person)

            pref_vel = np.array([0, 0])
            if current[1] is np.inf:
                pass
            elif self.old_coordinate is None:
                self.set_person_coordinate(person, current)
            else:
                delta_x, delta_y = current[0] - self.old_coordinate[0], current[2] - self.old_coordinate[2]
                delta_frame = current[3] - self.old_coordinate[3]
                velocity = np.array([delta_x / delta_time / delta_frame, delta_y / delta_time / delta_frame])
                speed = np.linalg.norm(velocity)
                pref_vel = velocity / speed if speed > 1 else velocity
                self.set_person_coordinate(person, current)

            v.append(pref_vel)  # m/s
        return v

    def get_person_coordinate(self, person):
        if person not in self.person_coordinate.keys():
            self.person_coordinate[person] = None
        return self.person_coordinate[person]

    def set_person_coordinate(self, person, value):
        self.person_coordinate[person] = value

    def get_depth_infor(self, depth_img, bb_box):
        coordinates = []
        depth_intrin = depth_img.profile.as_video_stream_profile().intrinsics
        for i in range(len(bb_box)):
            box = bb_box[i]
            center_x, center_y = box[0] + (box[2] - box[0]) // 2, box[1] + (box[3] - box[1]) // 2
            dis = self.get_center_depth_value(depth_img, center_x, center_y)
            if dis == 0:
                if self.check_inf_close(box):
                    camera_coordinate = [0, np.inf, 1]
                else:
                    camera_coordinate = [0, np.inf, 8]
            else:
                camera_coordinate = rs.rs2_deproject_pixel_to_point(intrin=depth_intrin, pixel=[center_x, center_y], depth=dis)
            coordinates.append(camera_coordinate)

        return coordinates

    def check_inf_close(self, bbox_xyxy):
        '''
        0: far inf
        1: close inf
        '''
        img_x, img_y = 1280, 720
        close_x_recognize_ratio, close_y_recognize_ratio = 0.7, 0.95
        left_x, left_y, right_x, right_y = bbox_xyxy
        if (right_x-left_x)/img_x>close_x_recognize_ratio and (right_y-left_y)/img_y>close_y_recognize_ratio:
            return 1
        else:
            return 0



class VideoVis(BasicDetector):
    def __init__(self, cfg, args, output=False, threshold=0.7):
        super().__init__(cfg, args)
        self.writer = output
        self.threshold = threshold
        if output:
            self.writer = self.set_writer(output)

    def set_writer(self, file_name):
        return cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'XVID'), 20, (1280, 720))

    def get_writer(self):
        return self.writer
