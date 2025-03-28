import onnxruntime as ort
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from detectron2.data import transforms as T
from argparse import ArgumentParser
from pathlib import Path
import onnx
import onnx_tensorrt.backend as backend

import torch

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import default_setup, default_argument_parser

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


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    # Load ONNX model
    model_type = 'resnet50'
    onnx_model_path = Path(f'output_{model_type}') / 'model.onnx'
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)  # This will raise an error if the model is invalid

    print('=' * 50)
    for input_ in onnx_model.graph.input:
            print(f'Input name: {input_.name}, Shape: {input_.shape}, Type: {input_.type}')
    print('=' * 50)

    engine = backend.prepare(onnx_model, device='cuda')

    # Load config (assuming `cfg` is set up in your script)
    image_path = Path('datasets/cityscapes/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png')
    input_tensor = preprocess_image(image_path, cfg)

    # Run inference
    # pred_masks = run_inference(onnx_model, input_tensor)
    output = engine.run(input_tensor)
    pred_mask = output[0]
    visualize_segmentation(pred_mask)