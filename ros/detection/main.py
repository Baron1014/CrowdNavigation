import argparse
import os
from vistualize import BagVis, ImgVis, VideoVis

def main(sys_args):
    path = os.path.dirname(os.path.realpath(__file__))
    output_file = os.path.join(path, sys_args.output_dir, sys_args.output_name)
    detector = BagVis(sys_args.bag_file, repeat=False)
    detector.detect_color(show='video', depth_info=True, output=output_file)
    # detector.detect_depth(show='video')

    # file = '/data/test.png'
    # imgvis = ImgVis()
    # img = imgvis.get_img(file)
    # imgvis.detect(img)

    # video = '/data/output_color.avi'
    # videovis = VideoVis()
    # videovis.detect_video(video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--output_name', type=str, default='CHIMEI_6F.avi')
    parser.add_argument('--bag_file', type=str, default='/data/20221024_142540.bag')

    sys_args = parser.parse_args()

    main(sys_args)