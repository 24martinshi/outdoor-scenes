import os
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

# filename = "1001.png"
input_folder = f'/scratch/gpfs/ms90/ControlNet/training/waymo_unif/target'
output_folder = f'/scratch/gpfs/ms90/ControlNet/training/waymo_unif/source'
apply_uniformer = UniformerDetector()

def read_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

def process_image(image):
    with torch.no_grad():
        input_image = HWC3(image)
        detected_map = apply_uniformer(input_image)
        H, W, C = image.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

    return detected_map

def save_image(image, output_path):
    cv2.imwrite(output_path, image)

# def main():
#     image = read_image(input_path)
#     segmented_image = process_image(image)
#     save_image(segmented_image, output_path)

def main():
    for filename in os.listdir(input_folder):
        
        if filename.endswith('.jpg') or filename.endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            image = read_image(input_path)
            segmented_image = process_image(image)
            save_image(segmented_image, output_path)
    
    
if __name__ == '__main__':
    import sys
    main()