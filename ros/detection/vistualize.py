import pyrealsense2 as rs
import cv2
import os
import copy
import numpy as np
from persondetection import DetectorAPI
from itertools import product

class BagVis:
    def __init__(self, file, repeat=True, threshold=0.7, maxm=16):
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
                detector = VideoVis(output=output)
                cout = 0
                while True:
                    going, frame = self.pipe.try_wait_for_frames(timeout_ms=20000)
                    playback.pause()
                    if going is False:
                        break
                    # align depth to color
                    frame = align.process(frame)
                    color_img = self.get_color_img(frame)
                    depth_frame = self.get_depth_frames(frame)
                    if depth_info:
                        detector.detect_with_depth_info(color_img, depth_frame, threshold=self.threshold)
                    else:
                        detector.detect(color_img, threshold=self.threshold)
                    key = cv2.waitKey(1)

                    #End loop once video finishes
                    playback.resume()
                    if key == 27:
                        cv2.destroyAllWindows()
                        break
                
                if detector.writer:
                    detector.writer.release()
        finally:
            self.pipe.stop()

class BasicDetector:
    def __init__(self):
        self.max_count = 0
        self.max_acc = 0
        self.max_avg_acc = 0
        self.odapi = DetectorAPI()
        self.threshold = 0.7
        self.old_time = None
        self.old_coordinate = None

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

    def _detect_color_with_depth(self, img, depth, threshold=None):
        boxes, scores, classes, num = self.odapi.processFrame(img)
        person = 0
        acc=0
        depth_intrin = depth.profile.as_video_stream_profile().intrinsics

        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                person += 1
                center_y, center_x = box[0]+(box[2]-box[0])//2, box[1]+(box[3]-box[1])//2
                left_top, right_bottom = [box[1], box[0]], [box[3], box[2]]
                dis = self.get_depth_value(depth, left_top, right_bottom)
                camera_coordinate = rs.rs2_deproject_pixel_to_point(intrin=depth_intrin, pixel=[center_x, center_y], depth=dis)
                velocity = self.get_velocity(depth, camera_coordinate)
                cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255,0,0), 2)  # cv2.FILLED #BGR
                cv2.putText(img, f'P{person, round(scores[i], 2)}', (box[1] - 30, box[0] - 8), cv2.FONT_HERSHEY_COMPLEX,0.5, (255, 255, 0), 1)  # (75,0,130),
                if 0.01<sum(map(abs,velocity))<10:
                    cv2.putText(img, f'Position:({camera_coordinate[0]:5.2f} {camera_coordinate[2]:5.2f})', (box[1] + 8, box[0] + 16), cv2.FONT_HERSHEY_PLAIN,1, (255, 255, 0), 1)  # (75,0,130),
                    cv2.putText(img, f'Velocity:({velocity[0]:5.2f} {velocity[1]:5.2f}) m/s', (box[1] + 8, box[0] + 32), cv2.FONT_HERSHEY_PLAIN,1, (255, 255, 0), 1)  # (75,0,130),
                
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

    def get_velocity(self, current_frame, current):
        current_timestemp = current_frame.get_timestamp()
        if self.old_time is None:
            self.old_time = current_timestemp
            self.old_coordinate = current
            return [0, 0]
        else:
            duration = (current_timestemp-self.old_time)/1000 + 1e-9 # msec to sec
            delta_x, delta_y = current[0] - self.old_coordinate[0], current[2] - self.old_coordinate[2]
            v_x, v_y = delta_x/duration, delta_y/duration
            
            self.old_time = current_timestemp
            self.old_coordinate = current
            return [v_x, v_y] # m/s


class VideoVis(BasicDetector):
    def __init__(self, output=False, threshold=0.7):
        super().__init__()
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

    def detect_with_depth_info(self, frame, depth, threshold=None):
        frame = self._detect_color_with_depth(frame, depth, threshold)

        if self.writer:
            self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        cv2.imshow("Human Detection from Video", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        

    def detect_video(self, file):
        path = os.path.dirname(os.path.realpath(__file__))
        self.file = path+file
        video = cv2.VideoCapture(self.file)
        check, _ = video.read()
        if check == False:
            print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
            return None

        while video.isOpened():
            # check is True if reading was successful
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

