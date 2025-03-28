import onnxruntime as ort
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from detectron2.data import transforms as T
from argparse import ArgumentParser
from pathlib import Path

import torch

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import default_setup

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import add_maskformer2_config


def setup(args):
    '''
    Create configs and perform basic setups.
    '''
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for 'mask_former' module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name='mask2former')
    return cfg


def load_onnx_model(model_path):
    return ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])


def preprocess_image(image_path, cfg):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    input_size = tuple(cfg.INPUT.CROP.SIZE)

    transform_gen = T.ResizeShortestEdge(short_edge_length=input_size[0], max_size=cfg.INPUT.MAX_SIZE_TEST)
    img = transform_gen.get_transform(img).apply_image(img)

    img = torch.as_tensor(img.transpose(2, 0, 1)).float() / 255.0
    return img.unsqueeze(0).numpy()

def run_inference(onnx_model, image_tensor):
    outputs = onnx_model.run(None, {'input': image_tensor})
    return outputs[0]  # Adjust index based on output structure

def visualize_segmentation(mask):
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()


parser = ArgumentParser()
parser.add_argument('--model-dir', type=str)
parser.add_argument('--image-path', type=str)
args = parser.parse_args()


if __name__ == '__main__':
    cfg = setup(args)
    # Load ONNX model
    onnx_model_path = Path(args.model_dir) / 'model.onnx'
    onnx_model = load_onnx_model(onnx_model_path)

    # Load config (assuming `cfg` is set up in your script)
    image_path = Path(args.image_path)
    input_tensor = preprocess_image(image_path, cfg)

    # Run inference
    pred_masks = run_inference(onnx_model, input_tensor)

    # Visualize first mask
    visualize_segmentation(pred_masks[0])