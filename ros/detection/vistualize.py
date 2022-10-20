import pyrealsense2 as rs
import cv2
import os
import copy
import numpy as np
from persondetection import DetectorAPI

class BagVis:
    def __init__(self, file, repeat=True, threshold=0.7, maxm=4):
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
                detector.detect(color_img, depth=depth_img, threshold=self.threshold)

            elif show== 'video':
                detector = VideoVis()
                # align depth to color
                while True:
                    frame = self.pipe.wait_for_frames()
                    frame = align.process(frame)
                    color_img = self.get_color_img(frame)
                    depth_img = self.get_depth_img(frame)

                    detector.detect(color_img, depth=depth_img, threshold=self.threshold)
                    key = cv2.waitKey(1)
                    #End loop once video finishes
                    if key == 27:
                        cv2.destroyAllWindows()
                        break
        finally:
            self.pipe.stop()
            if detector.writer:
                detector.writer.release()

    def detect_color(self, show='image', depth_info=False):
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
                depth_frames = self.get_depth_frames(frame)
                detector = ImgVis()
                if depth_info:
                    detector.detect_with_depth_info(color_img, depth_frames, threshold=self.threshold)
                else:
                    detector.detect(color_img, threshold=self.threshold)
            
            elif show== 'video':
                detector = VideoVis()
                while True:
                    frame = self.pipe.wait_for_frames(timeout_ms=20000)
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
                dis = self.get_depth_value(depth, center_x, center_y)
                camera_coordinate = rs.rs2_deproject_pixel_to_point(intrin=depth_intrin, pixel=[center_x, center_y], depth=dis)
                cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255,0,0), 2)  # cv2.FILLED #BGR
                cv2.putText(img, f'P{person, round(scores[i], 2)} Px={camera_coordinate[0]:.2f} Py={camera_coordinate[2]:.2f}', (box[1] - 30, box[0] - 8), cv2.FONT_HERSHEY_COMPLEX,0.5, (255, 255, 0), 1)  # (75,0,130),
                acc += scores[i]
                if (scores[i] > self.max_acc):
                    self.max_acc = scores[i]

        if (person > self.max_count):
            self.max_count = person
        if(person>=1):
            if((acc / person) > self.max_avg_acc):
                self.max_avg_acc = (acc / person)

        return img

    def get_depth_value(self, depth, center_x, center_y):
        return depth.get_distance(center_x, center_y)


class VideoVis(BasicDetector):
    def __init__(self, output=False, threshold=0.7):
        super().__init__()
        self.writer = output
        self.threshold = threshold

    def set_writer(self, file_name):
        return cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'XVID'), 20, (640, 480))

    def detect(self, frame, depth=None):
        frame = self._detect(frame, depth=depth)
        # color problem
        if depth is not None:
            frame =  cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.writer:
            self.writer.write(frame)

        cv2.imshow("Human Detection from Video", frame)

    def detect_with_depth_info(self, frame, depth, threshold=None):
        frame = self._detect_color_with_depth(frame, depth, threshold)

        if self.writer:
            self.writer.write(frame)

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

