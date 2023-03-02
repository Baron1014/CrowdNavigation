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
from utils.log import get_logger
from utils.draw import draw_boxes, compute_color_for_labels
import torch
import warnings

class BagVis:
    def __init__(self, sys_args, config, file, repeat=True, threshold=0.7, maxm=16):
        '''
        repeat: Image or Video
            True: Video
            False: Image
        '''
        path = os.path.dirname(os.path.realpath(__file__))
        self.pipe = rs.pipeline()
        self.config = rs.config()
        self.threshold = threshold
        self.maxm = maxm
        self.args = sys_args
        self.cfg = config
        rs.align(rs.stream.color)
        rs.config.enable_device_from_file(self.config, path+file, repeat_playback=repeat)

    def get_color_img(self, frame):
        color_rs = frame.get_color_frame()
        img = np.asanyarray(color_rs.get_data())
        
        return img

    def get_depth_img(self, frame):
        depth_rs = self.get_depth_frames(frame)
        depth = np.asanyarray(depth_rs.get_data())
        dimg_gray = cv2.convertScaleAbs(depth, alpha=255/(self.maxm*1000))
        depth_colored = cv2.applyColorMap(dimg_gray, cv2.COLORMAP_BONE)

        return depth_colored

    def get_depth_frames(self, frame):
        return frame.get_depth_frame()
    

    def detect_depth(self, show='image'):
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipe.start(self.config)
        align_to = rs.stream.color
        align = rs.align(align_to)
        try:
            if show=='image':
                frame = self.pipe.wait_for_frames()
                # align depth to color
                frame = align.process(frame)
                color_img = self.get_color_img(frame)
                depth_img = self.get_depth_img(frame)

                detector = ImgVis()
                detector.detect(color_img, depth=depth_img)

            elif show== 'video':
                detector = VideoVis()
                # align depth to color
                while True:
                    frame = self.pipe.wait_for_frames()
                    frame = align.process(frame)
                    color_img = self.get_color_img(frame)
                    depth_img = self.get_depth_img(frame)

                    detector.detect(color_img, depth=depth_img)
                    key = cv2.waitKey(1)
                    #End loop once video finishes
                    if key == 27:
                        cv2.destroyAllWindows()
                        break
        finally:
            self.pipe.stop()
            if detector.writer:
                detector.writer.release()

    def detect_color(self, show='image', depth_info=False, output=False):
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = self.pipe.start(self.config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        align_to = rs.stream.color
        align = rs.align(align_to)
        try:
            if show=='image':
                frame = self.pipe.wait_for_frames()
                # align depth to color
                frame = align.process(frame)
                color_img = self.get_color_img(frame)
                depth_frames = self.get_depth_frames(frame)
                detector = ImgVis()
                if depth_info:
                    detector.detect_with_depth_info(color_img, depth_frames, threshold=self.threshold)
                else:
                    detector.detect(color_img, threshold=self.threshold)
            
            elif show== 'video':
                detector = VideoVis(self.cfg, self.args, output=output)
                idx_frame = 0
                while True:
                    idx_frame +=1
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
                        velocity = detector.get_velocity(identities, camera_coor, time.time()-start, idx_frame)
                        self.draw_information(rgb_im, identities, velocity, bbox_xyxy, camera_coor)

                    if self.args.display:
                        cv2.imshow("test", rgb_im)
                        key =cv2.waitKey(1)

                    
                    # key = cv2.waitKey(1)
                    # if depth_info:
                    #     detector.detect_with_depth_info(color_img, depth_frame, start_time=start, threshold=self.threshold)
                    # else:
                    #     detector.detect(color_img, threshold=self.threshold)

                    #End loop once video finishes
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
            if 0<=sum(map(abs,velocity))<10:
                color = compute_color_for_labels(person)
                position_label = f'Position:({camera_coordinate[0]:5.2f} {camera_coordinate[2]:5.2f})'
                velocity_label = f'Velocity:({velocity[0]:5.2f} {velocity[1]:5.2f}) m/s'
                p_size = cv2.getTextSize(position_label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
                v_size = cv2.getTextSize(velocity_label, cv2.FONT_HERSHEY_PLAIN, 1 , 2)[0]
                cv2.rectangle(img,(box[0]-3, box[1]-2*(p_size[1]+v_size[1])),(box[0]+v_size[0],box[1]), color,-1)
                cv2.putText(img, position_label, (box[0], box[1]-p_size[1]-v_size[1]), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 2)  # (75,0,130),
                cv2.putText(img, velocity_label, (box[0], box[1]-v_size[1]), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 2)  # (75,0,130),
        
class BasicDetector:
    def __init__(self,  cfg, args):
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

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        # if args.cam != -1:
        #     print("Using webcam " + str(args.cam))
        #     self.vdo = cv2.VideoCapture(args.cam)
        self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)

    def _detect(self, img, depth=None, threshold=None):
        boxes, scores, classes, num = self.odapi.processFrame(img)
        person = 0
        acc=0
        img = depth if depth is not None else img
        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                person += 1
                cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255,0,0), 2)  # cv2.FILLED #BGR
                cv2.putText(img, f'P{person, round(scores[i], 2)}', (box[1] - 30, box[0] - 8), cv2.FONT_HERSHEY_COMPLEX,0.5, (0, 0, 255), 1)  # (75,0,130),
                acc += scores[i]
                if (scores[i] > self.max_acc):
                    self.max_acc = scores[i]

        if (person > self.max_count):
            self.max_count = person
        if(person>=1):
            if((acc / person) > self.max_avg_acc):
                self.max_avg_acc = (acc / person)

        return img

    def _detect_color_with_depth(self, img, depth, start_time, threshold=None):
        boxes, scores, classes, num = self.odapi.processFrame(img)
        person = 0
        acc=0
        depth_intrin = depth.profile.as_video_stream_profile().intrinsics
        end_time = time.time()
        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                person += 1
                center_y, center_x = box[0]+(box[2]-box[0])//2, box[1]+(box[3]-box[1])//2
                left_top, right_bottom = [box[1], box[0]], [box[3], box[2]]
                dis = self.get_depth_value(depth, left_top, right_bottom)
                camera_coordinate = rs.rs2_deproject_pixel_to_point(intrin=depth_intrin, pixel=[center_x, center_y], depth=dis)
                velocity = self.get_velocity(person, camera_coordinate, end_time-start_time)
                cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (138, 43, 226), 2)  # cv2.FILLED #BGR
                cv2.putText(img, f'P{person, round(scores[i], 2)}', (box[1] - 30, box[0] - 8), cv2.FONT_HERSHEY_COMPLEX,0.5, (255,69,0), 1)  # (75,0,130),
                if 0.01<sum(map(abs,velocity))<10:
                    cv2.putText(img, f'Position:({camera_coordinate[0]:5.2f} {camera_coordinate[2]:5.2f})', (box[1] + 8, box[0] + 16), cv2.FONT_HERSHEY_PLAIN,1, (255,69,0), 1)  # (75,0,130),
                    cv2.putText(img, f'Velocity:({velocity[0]:5.2f} {velocity[1]:5.2f}) m/s', (box[1] + 8, box[0] + 32), cv2.FONT_HERSHEY_PLAIN,1, (255,69,0), 1)  # (75,0,130),
                
                acc += scores[i]
                if (scores[i] > self.max_acc):
                    self.max_acc = scores[i]

        if (person > self.max_count):
            self.max_count = person
        if(person>=1):
            if((acc / person) > self.max_avg_acc):
                self.max_avg_acc = (acc / person)

        return img

    def get_depth_value(self, depth, left_top, right_bottom):
        max_value = 0
        all_x , all_y = list(), list()
        for i, j in product(range(left_top[0], right_bottom[0]), range(left_top[1], right_bottom[1])):
            all_x.append(i)
            all_y.append(j)
        all_depth = list(map(depth.get_distance, all_x, all_y))
        filled_zero = list(filter(lambda x: x != 0, all_depth))

        return min(filled_zero)
    
    def get_center_depth_value(self, depth, center_x, center_y):
        buffer = 10
        all_x , all_y = list(), list()
        for i, j in product(range(center_x-buffer, center_x+buffer), range(center_y-buffer, center_y+buffer)):
            all_x.append(i)
            all_y.append(j)
        all_depth = list(map(depth.get_distance, all_x, all_y))
        filled_zero = list(filter(lambda x: x != 0, all_depth))

        return min(filled_zero) if len(filled_zero)>0 else 0

    def get_velocity(self, persons, current_coor, delta_time, idx_frame):
        v = []
        for i in range(len(persons)):
            person = persons[i]
            current = current_coor[i] + [idx_frame]
            self.old_coordinate = self.get_person_coordinate(person)
            
            if self.old_coordinate is None:
                v_x, v_y = 0, 0
            else:
                delta_x, delta_y = current[0] - self.old_coordinate[0], current[2] - self.old_coordinate[2]
                delta_frame =  current[3] - self.old_coordinate[3]
                v_x, v_y = delta_x/delta_time/delta_frame, delta_y/delta_time/delta_frame

            self.set_person_coordinate(person, current)
            v.append([v_x, v_y])# m/s
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
            center_x, center_y = box[0]+(box[2]-box[0])//2, box[1]+(box[3]-box[1])//2
            left_top, right_bottom = [box[0], box[1]], [box[2], box[3]]
            # dis = self.get_depth_value(depth_img, left_top, right_bottom)
            dis = self.get_center_depth_value(depth_img, center_x, center_y)
            camera_coordinate = rs.rs2_deproject_pixel_to_point(intrin=depth_intrin, pixel=[center_x, center_y], depth=dis)
            coordinates.append(camera_coordinate)

        return coordinates

class VideoVis(BasicDetector):
    def __init__(self, cfg, args, output=False, threshold=0.7):
        super().__init__(cfg, args)
        self.writer = output
        self.threshold = threshold
        if output:
            self.writer = self.set_writer(output)

    def set_writer(self, file_name):
        return cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'XVID'), 20, (1280, 720))

    def detect(self, frame, depth=None):
        frame = self._detect(frame, depth=depth, threshold=self.threshold)
        # color problem
        if depth is not None:
            frame =  cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.writer:
            self.writer.write(frame)

        cv2.imshow("Human Detection from Video", frame)

    def detect_with_depth_info(self, frame, depth, start_time, threshold=None):
        frame = self._detect_color_with_depth(frame, depth, start_time, threshold)

        if self.writer:
            self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        cv2.imshow("Human Detection from Video", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        

    def detect_video(self, file):
        path = os.path.dirname(os.path.realpath(__file__))
        self.file = path+file
        video = self.vdo(self.file)
        check, _ = video.read()
        if check == False:
            print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
            return None

        while video.isOpened():
            # check is True if reading was successful
            # s = time.start()
            check, frame = video.read()
            
            if(check==True):
                self._detect(frame)
                key = cv2.waitKey(1)

                #End loop once video finishes
                if key == 27:
                    cv2.destroyAllWindows()
                    break
            else:
                break

        video.release()
        cv2.destroyAllWindows()


class ImgVis(BasicDetector):
    def __init__(self, threshold=0.7):
        super().__init__()
        self.threshold = threshold

    def get_img(self, file):
        path = os.path.dirname(os.path.realpath(__file__))
        self.file = path+file
        image = cv2.imread(self.file)
        img = cv2.resize(image, (image.shape[1], image.shape[0]))

        return img

    def detect(self, img, depth=None, threshold=None):
        img = self._detect(img, depth, threshold)

        cv2.imshow("Human Detection from Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_with_depth_info(self, img, depth, threshold=None):
        img = self._detect_color_with_depth(img, depth, threshold)

        cv2.imshow("Human Detection from Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

