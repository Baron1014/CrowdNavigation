import os
import cv2
import time
from ros_detection.vistualize import BagVis
from deepsort_utils.parser import get_config
from deepsort_utils.draw import draw_boxes


def camera_setting(sys_args):
    path = os.path.dirname(os.path.realpath(__file__))
    cfg = get_config()
    cfg.USE_MMDET = False
    cfg.merge_from_file(os.path.join(path, sys_args.config_detection))
    cfg.merge_from_file(os.path.join(path, sys_args.config_deepsort))
    cfg.USE_FASTREID = False
    output_file = os.path.join(path, sys_args.video_output_dir, sys_args.video_output_name)
    detector = BagVis(sys_args, cfg, sys_args.bag_file, repeat=False)
    video_detector = detector.get_video_detector(output_file)
    profile = detector.pipe.start()

    return video_detector, detector


def camera_detection(video_detector, detector, start_time, idx_frame):
    going, frame = detector.pipe.try_wait_for_frames(timeout_ms=20000)
    #detector.playback.pause()
    # align depth to color
    frame = detector.align.process(frame)
    bgr_img = detector.get_color_img(frame)
    # deepsort
    rgb_im = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    bbox_xywh, cls_conf, cls_ids = video_detector.detector(bgr_img)
    # select person class
    mask = cls_ids == 0

    bbox_xywh = bbox_xywh[mask]
    # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
    bbox_xywh[:, 3:] *= 1.2
    cls_conf = cls_conf[mask]

    # do tracking
    outputs = video_detector.deepsort.update(bbox_xywh, cls_conf, bgr_img)

    # draw boxes for visualization
    camera_coor, velocity, key = [], [], 0
    if len(outputs) > 0:
        bbox_tlwh = []
        bbox_xyxy = outputs[:, :4]
        identities = outputs[:, -1]
        rgb_im = draw_boxes(rgb_im, bbox_xyxy, identities)

        for bb_xyxy in bbox_xyxy:
            bbox_tlwh.append(video_detector.deepsort._xyxy_to_tlwh(bb_xyxy))

        depth_frame = detector.get_depth_frames(frame)
        camera_coor = video_detector.get_depth_infor(depth_frame, bbox_xyxy)
        velocity = video_detector.get_velocity(identities, camera_coor, time.time() - start_time, idx_frame)
        detector.draw_information(rgb_im, identities, velocity, bbox_xyxy, camera_coor)
    out = video_detector.get_writer()
    out.write(rgb_im)
    if detector.args.display:
        cv2.imshow("test", rgb_im)
        key = cv2.waitKey(1)
        pass
    #detector.playback.resume()

    return camera_coor, velocity, key
