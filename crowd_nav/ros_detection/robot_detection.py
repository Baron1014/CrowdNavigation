import os
import cv2
from ros_detection.vistualize import BagVis

def camera_setting(sys_args):
    path = os.path.dirname(os.path.realpath(__file__))
    output_file = os.path.join(path, sys_args.video_output_dir, sys_args.video_output_name)
    detector = BagVis(sys_args.bag_file, repeat=False)
    video_detector = detector.video_preprocess(output=output_file)

    return video_detector, detector

def camera_detection(video_detector, detector):
    going, frame = detector.pipe.try_wait_for_frames(timeout_ms=20000)
    detector.playback.pause()
    # align depth to color
    frame = detector.align.process(frame)
    color_img = detector.get_color_img(frame)
    depth_frame = detector.get_depth_frames(frame)
    position, velocity = video_detector.detect_with_depth_info(color_img, depth_frame, threshold=detector.threshold)
    #End loop once video finishes
    key = cv2.waitKey(1)
    #End loop once video finishes
    detector.playback.resume()
    if key == 27:
        cv2.destroyAllWindows()
    
    return position, velocity 