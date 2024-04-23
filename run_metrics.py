import argparse
import glob
import os

import cv2
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm


def metric_option():
    arg = argparse.ArgumentParser("Metric for Image2Image model")
    
    arg.add_argument("--metric", help='Choose metric to run: psnr, ssim or all',
                     default="psnr")
    arg.add_argument("--source", help='source path')
    arg.add_argument("--target", help="target path")

    return arg.parse_args()


if __name__ == "__main__":
    opt = metric_option()
    source = []
    source_path = []
    target = []
    
    print("READ IMAGE")

    for path, img_list in zip([opt.source, opt.target], [source, target]):
    # for path, img_list in zip([opt.source], [source]):
        if os.path.isfile(path):
            img_list.append(cv2.imread(path))
        else:
            for item in glob.glob(f"{path}/**/*.jpg", recursive=True):
                img_file = item
                source_path.append(item)
                img_list.append(cv2.imread(img_file))
    
    # target = [cv2.imread(os.path.join(opt.target, os.path.basename(item)))
    #           for item in source_path]
    assert len(source) == len(target)   
    
    print("READ DONE")
    
    metric ={}

    for clean, reconstruct in tqdm(zip(source, target), total=len(source)):
        if opt.metric == "psnr" or opt.metric == "all":
            if not "psnr" in metric.keys():
                metric["psnr"] = 0
            metric["psnr"] += compare_psnr(clean,reconstruct, data_range=255)

        if opt.metric == "ssim" or opt.metric == "all":
            if not "ssim" in metric.keys():
                metric["ssim"] = 0
            metric["ssim"] += compare_ssim(clean, reconstruct, data_range=255, channel_axis=-1)
        
        
    for k,v in metric.items():
        v = v/len(target)
        print(k, v, sep=": ")
