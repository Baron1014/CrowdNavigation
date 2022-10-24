import argparse
from vistualize import BagVis, ImgVis, VideoVis

def main(sys_args):
    detector = BagVis(sys_args.bag_file)
    detector.detect_color(show='video', depth_info=True)
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
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--bag_file', type=str, default='/data/20221024_142540.bag')

    sys_args = parser.parse_args()

    main(sys_args)