import pyrealsense2 as rs
import cv2
import os
import numpy as np
from persondetection import DetectorAPI

class BagVis:
    def __init__(self, file, repeat=True):
        '''
        repeat: Image or Video
            True: Video
            False: Image
        '''
        path = os.path.dirname(os.path.realpath(__file__))
        self.pipe = rs.pipeline()
        self.config = rs.config()
        rs.align(rs.stream.color)
        rs.config.enable_device_from_file(self.config, path+file, repeat_playback=repeat)
        

    def BagtoImg(self):
        frame = self.pipe.wait_for_frames()
        color_rs = frame.get_color_frame()
        img = np.asanyarray(color_rs.get_data())

        detector = ImgVis()
        detector.detect(img)

    def BagtoVideo(self):
        detector = VideoVis()
        while True:
            frame = self.pipe.wait_for_frames()
            color_rs = frame.get_color_frame()
            img = np.asanyarray(color_rs.get_data())
            detector.detect(img)
            key = cv2.waitKey(1)

            #End loop once video finishes
            if key == 27:
                cv2.destroyAllWindows()
                break
        
        if detector.writer:
            detector.writer.release()
    

    def detect_depth(self):
        self.pipe.start(self.config)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipe.end()

    def detect_color(self, show='Image'):
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
        self.pipe.start(self.config)
        if show=='Image':
            self.BagtoImg()
        
        elif show== 'video':
            self.BagtoVideo()

        self.pipe.stop()


class VideoVis:
    def __init__(self, output=False):
        self.max_count = 0
        self.max_acc = 0
        self.max_avg_acc = 0
        self.odapi = DetectorAPI()
        self.writer = output

    def set_writer(self, file_name):
        return cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'XVID'), 20, (640, 480))

    def detect(self, frame, threshold = 0.7, depth=False):
        boxes, scores, classes, num = self.odapi.processFrame(frame)
        person = 0
        acc = 0
        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                person += 1
                cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)  # cv2.FILLED
                cv2.putText(frame, f'P{person, round(scores[i],2)}', (box[1]-30, box[0]-8), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1 )#(75,0,130),
                acc+=scores[i]
                if(scores[i]>self.max_acc):
                    self.max_acc = scores[i]

        if(person>self.max_count):
            self.max_count = person
        if(person>=1):
            if((acc/person)>self.max_avg_acc):
                self.max_avg_acc = (acc/person)
        
        # color problem
        if not depth:
            frame =  cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.writer:
            self.writer.write(frame)

        cv2.imshow("Human Detection from Video", frame)
        

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
                self.detect(frame)
                key = cv2.waitKey(1)
                #End loop once video finishes
                if key == 27:
                    cv2.destroyAllWindows()
                    break
            else:
                break

        video.release()
        cv2.destroyAllWindows()


class ImgVis:
    def __init__(self):
        self.max_count1 = 0
        self.max_acc1 = 0
        self.max_avg_acc1 = 0
        self.odapi = DetectorAPI()
        

    def get_img(self, file):
        path = os.path.dirname(os.path.realpath(__file__))
        self.file = path+file
        image = cv2.imread(self.file)
        img = cv2.resize(image, (image.shape[1], image.shape[0]))

        return img

    def detect(self, img, threshold = 0.7):
        boxes, scores, classes, num = self.odapi.processFrame(img)
        person = 0
        acc=0
        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                person += 1
                cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255,0,0), 2)  # cv2.FILLED #BGR
                cv2.putText(img, f'P{person, round(scores[i], 2)}', (box[1] - 30, box[0] - 8), cv2.FONT_HERSHEY_COMPLEX,0.5, (0, 0, 255), 1)  # (75,0,130),
                acc += scores[i]
                if (scores[i] > self.max_acc1):
                    self.max_acc1 = scores[i]

        if (person > self.max_count1):
            self.max_count1 = person
        if(person>=1):
            if((acc / person) > self.max_avg_acc1):
                self.max_avg_acc1 = (acc / person)


        cv2.imshow("Human Detection from Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()
