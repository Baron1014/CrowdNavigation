import argparse
import os
from vistualize import BagVis, ImgVis, VideoVis
from utils.parser import get_config

def main(sys_args):
    path = os.path.dirname(os.path.realpath(__file__))
    cfg = get_config()
    cfg.USE_MMDET = False
    cfg.merge_from_file(os.path.join(path,sys_args.config_detection))
    cfg.merge_from_file(os.path.join(path,sys_args.config_deepsort))
    cfg.USE_FASTREID = False
    output_file = os.path.join(path, sys_args.output_dir, sys_args.output_name)
    detector = BagVis(sys_args, cfg, sys_args.bag_file, repeat=False)
    detector.detect_color(show='video', depth_info=True, output=output_file)
    # detector.detect_depth(show='video')

    # file = '/data/test.png'
    # imgvis = ImgVis()
    # img = imgvis.get_img(file)
    # imgvis.detect(img, threshold=0.7)

    # video = '/data/output_color.avi'
    # videovis = VideoVis()
    # videovis.detect_video(video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--output_name', type=str, default='CHIMEI_6F.avi')
    parser.add_argument('--bag_file', type=str, default='/data/20221024_142540.bag')
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--display", default=True, action="store_true")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--config_deepsort", type=str, default="configs/deep_sort.yaml")
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")

    sys_args = parser.parse_args()

    main(sys_args)