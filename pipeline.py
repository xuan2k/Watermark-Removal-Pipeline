import glob
import os
import time
from io import BytesIO

import cv2
import requests
from PIL import Image
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm


def inference(url, files, data):
    response = requests.post(url=url, files=files, params=data)
    return response.json()

s = time.time()
def run_pipeline(image_path):
    #Segmentation
    segmentation_url = "http://0.0.0.0:10109/api/segmentation"
    pil_image = Image.open(image_path)
    # Convert PIL Image to bytes
    img_bytes = BytesIO()
    # You can change the format as needed
    pil_image.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()
    segmentation_files = [("img", img_bytes)]
    segmentation_data = {"save_path": "/home/huylx/transformer/demo"}
    segmentation_response = inference(segmentation_url, segmentation_files, segmentation_data)
    # print("SEGEMENTATION:", segmentation_response)

    # I2SB
    inpaint_url = "http://0.0.0.0:10110/api/inpaint"
    inpaint_files = []
    inpaint_data = {"img_path": "/home/huylx/transformer/demo/img.png",
                    "mask_path": "/home/huylx/transformer/demo/mask.npy",
                    "save_dir": "/home/huylx/transformer/demo"}
    inpain_response = inference(inpaint_url, inpaint_files, inpaint_data)
    # print("INPAINT:", inpain_response)
    # print("PIPELINE TIME: ", (time.time()- s)*1e3)

def evaluate():
    
    img_list = glob.glob(
        "/home/huylx/transformer/data/wtm_v1/images/validation/*.jpg")
    
    print(len(img_list))
    
    to_test = 7500
    metric = {}
    
    for i in tqdm(range(to_test)):
        image_path = img_list[i]
        # image_path = "/home/huylx/transformer/data/wtm_v1/images/validation/2.jpg"
        
        
        run_pipeline(image_path)
        clean_path = "/home/huylx/transformer/data/clean"
        clean_img_path = os.path.join(clean_path, os.path.basename(image_path))
        clean = cv2.imread(clean_img_path)
        reconstruct = cv2.imread("/home/huylx/transformer/demo/result.png")
        
        if not "psnr" in metric.keys():
            metric["psnr"] = 0
        metric["psnr"] += compare_psnr(clean,reconstruct, data_range=255)
        
        if not "ssim" in metric.keys():
            metric["ssim"] = 0
        metric["ssim"] += compare_ssim(clean, reconstruct, data_range=255, channel_axis=-1)
    
    for k, v in metric.items():
        v = v/to_test
        print(k, v, sep=": ")
    
    return

if __name__ == "__main__":
    evaluate()
