# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import pickle
import glob
import sys
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import colorsys

try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os
import numpy as np
import time
import onnx
import onnx_tensorrt.backend as backend
import pycuda.autoinit
import tensorrt as trt
import onnxruntime as ort
import cv2

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def benchmark_cuda(model, test_loader, verbose=False):
    print("Preparing engine for dataset evaluation...")
    model.eval()
    print("Engine prepared!")

    all_predictions = []
    all_ground_truth = []

    start_time = time.time()
    print("Starting CUDA inference on test set...")

    for batch_idx, batch in enumerate(test_loader):
        for sample in batch:
            image = sample['image']  # Shape: [C, H, W]
            # label = sample['sem_seg']             # [H, W]

            time_before_forward = time.time()
            output = model(image)
            time_after_forward = time.time()
            # pred = output[0]['panoptic_seg'][0].clone().detach().cpu().numpy()        # [C, H, W]
            pred = output[0]['sem_seg'].clone().detach()        # [C, H, W]
            pred = torch.argmax(pred, dim=0).cpu().numpy()        # [H, W]  # uncomment after benchmarking

            time_after_pred = time.time()
            print(f"Batch {batch_idx} - Forward time: {time_after_forward - time_before_forward:.4f}s, " +
                  f"Post-processing time: {time_after_pred - time_after_forward:.4f}s")

            all_predictions.append(pred)
            # all_ground_truth.append(label)

        if batch_idx % 5 == 0 and verbose:
            elapsed_time = time.time() - start_time
            print(f"[Batch {batch_idx}/{len(test_loader)}] Elapsed time: {elapsed_time:.2f}s")

    total_time = time.time() - start_time
    print(f"Inference complete!")
    for out in [sys.stdout, open("cudaexec.log", "w")]:
        out.write("=========CUDA BENCHMARKS=========\n")
        out.write(f"Total time: {total_time:.2f}s, Average time per image: {total_time / len(test_loader):.4f}s, " +
                  f"FPS: {len(test_loader) / total_time:.2f}\n")
        out.flush()

    all_predictions_np = np.stack(all_predictions)
    # all_ground_truth_np = np.stack(all_ground_truth)

    mean_iou = 1#compute_mean_iou(all_predictions_np, all_ground_truth_np)
    print(f"Mean IoU: {mean_iou:.4f}")
    return {"mean_iou": mean_iou}


def benchmark_onnx_trt(test_loader, trt_model_path, verbose=False, trt_session=None):
    if trt_session is None:
        print("Loading TensorRT engine...")
        trt_session = ort.InferenceSession(trt_model_path, providers=["TensorrtExecutionProvider"])
        print("Engine loaded!")
    else:
        print("Using existing TensorRT engine...")
    print("Preparing engine for dataset evaluation...")

    all_predictions = []
    all_ground_truth = []

    start_time = time.time()
    print("Starting ONNX inference on test set...")

    for batch_idx, batch in enumerate(test_loader):
        for sample in batch:
            image = sample['image'].cpu().numpy().astype(np.float32)  # Shape: [C, H, W]
            label = sample['sem_seg'].cpu().numpy()                # [H, W]

            time_before_forward = time.time()
            output = trt_session.run(None, {"input": image})
            time_after_forward = time.time()
            pred = torch.tensor(output[0])    # for semantic segmentation                    # [C, H, W]
            # pred = torch.tensor(output[1])    # for panoptic segmentation                    # [C, H, W]
            time_after_pred = time.time()
            print(f"Batch {batch_idx} - Forward time: {time_after_forward - time_before_forward:.4f}s, " +
                  f"Post-processing time: {time_after_pred - time_after_forward:.4f}s")

            # cv2.imwrite(f"output/inference/{batch_idx}_gt.png", label)
            # cv2.imwrite(f"output/inference/{batch_idx}_pred.png", pred_resized)

            all_predictions.append(pred)
            all_ground_truth.append(label)

        if batch_idx % 5 == 0 and verbose:
            elapsed_time = time.time() - start_time
            print(f"[Batch {batch_idx}/{len(test_loader)}] Elapsed time: {elapsed_time:.2f}s")

    total_time = time.time() - start_time
    print(f"Inference complete!")
    for out in [sys.stdout, open("trtexec.log", "w")]:
        out.write("=========TENSORRT BENCHMARKS=========\n")
        out.write(f"Total time: {total_time:.2f}s, Average time per image: {total_time / len(test_loader):.4f}s, " +
                  f"FPS: {len(test_loader) / total_time:.2f}\n")
        out.flush()

    all_predictions_np = np.stack(all_predictions)
    all_ground_truth_np = np.stack(all_ground_truth)

    mean_iou = compute_mean_iou(all_predictions_np, all_ground_truth_np)
    print(f"Mean IoU: {mean_iou:.4f}")
    return {"mean_iou": mean_iou}


def prepare_image(image, scale_factor):
    image_np = np.array(image, dtype=np.float32) / 255.0
    image_tensor = torch.tensor(image_np)
    image_tensor = image_tensor.permute(2, 0, 1)
    image_tensor = torch.unsqueeze(image_tensor, dim=0)
    image_tensor = F.interpolate(image_tensor, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    image_tensor = torch.squeeze(image_tensor)
    image_tensor = image_tensor * 255
    image_tensor = torch.tensor(image_tensor, dtype=torch.float32)
    return image_tensor.to("cuda")


class Mask2FormerONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, x):
        with torch.no_grad():
            outputs = self.model(x)
            sem_seg = outputs[0].get("sem_seg", None)         # (C, H, W)
            panoptic_seg = outputs[0].get("panoptic_seg", None)  # (H, W)
            # If panoptic_seg is a tuple (segmentation, segments_info), extract only the tensor
            if isinstance(panoptic_seg, tuple):
                panoptic_seg = panoptic_seg[0]
            sem_seg = torch.argmax(sem_seg, dim=0)          # (H, W)
            return sem_seg, panoptic_seg


def start_benchmark(args, model, image, trt_model_path, testloader, verbose=False):
    colormap = create_cityscapes_colormap()
    print(f'Length of colormap: {len(colormap)}')
    trt_only = True
    if args.cuda and not trt_only:
        print("CUDA BENCHMARK:")
        start = time.time()
        output = model(image)
        segments_info = output[0]['panoptic_seg'][1]
        output = output[0]["panoptic_seg"][0].cpu().numpy()

        # output = torch.argmax(output[0]["sem_seg"], dim=0).cpu().numpy()
        end = time.time()
        print(f"CUDA inference time for single image: {end - start:.4f} seconds")

        # output = colorize_prediction(output, colormap)
        # encoded_map = encode_panoptic_map(output, segments_info, label_divisor=1000)
        # output = colorize_panoptic_prediction(encoded_map, colormap)

        plt.imsave(args.res_path, output)
        results = benchmark_cuda(model, testloader, verbose)
        print("Results:", results)
    if args.compile:
        x = image
        model = Mask2FormerONNXWrapper(model).to("cuda").eval()
        torch.out = model(x)
        torch.onnx.export(model, x, trt_model_path, export_params=True, opset_version=18,
                          do_constant_folding=True, input_names=["input"], output_names=["sem_seg", "panoptic_seg"], verbose=True)
        print(f"Model successfully exported to {trt_model_path}")
    else:
        print("TENSORRT BENCHMARK:")
        print(f"Doing inference on {trt_model_path} ...")

        print("Model loaded!")

        trt_session = ort.InferenceSession(trt_model_path, providers=["TensorrtExecutionProvider"])
        # trt_session = ort.InferenceSession(trt_model_path, providers=["CUDAExecutionProvider"])
        print("Engine prepared!")
        print("Running inference ...")
        start = time.time()
        output = trt_session.run(None, {"input": image.cpu().numpy()})
        output = torch.tensor(output[0])  # for semantic segmentation
        # output = torch.tensor(output[1])  # for panoptic segmentation
        end = time.time()
        print(f"TensorRT inference time for single image: {end - start:.4f} seconds")

        # output = colorize_prediction(output, colormap)

        # plt.imsave(args.trt_res_path, output)

        results = benchmark_onnx_trt(testloader, trt_model_path, verbose, trt_session)
        print("Results:", results)
        print("Benchmark finished!")
        return


def create_cityscapes_colormap():
    # Initialize the colormap with zeros
    colormap = np.zeros((256, 3), dtype=int)

    # Mapping class labels to their corresponding colors
    colormap[0] = [128, 64, 128]    # Road
    colormap[1] = [244, 35, 232]    # Sidewalk
    colormap[2] = [70, 70, 70]      # Building
    colormap[3] = [102, 102, 156]   # Wall
    colormap[4] = [190, 153, 153]   # Fence
    colormap[5] = [153, 153, 153]   # Pole
    colormap[6] = [250, 170, 30]    # Traffic Light
    colormap[7] = [220, 220, 0]     # Traffic Sign
    colormap[8] = [107, 142, 35]    # Vegetation
    colormap[9] = [152, 251, 152]   # Terrain
    colormap[10] = [70, 130, 180]   # Sky
    colormap[11] = [220, 20, 60]    # Person
    colormap[12] = [255, 0, 0]      # Rider
    colormap[13] = [0, 0, 142]      # Car
    colormap[14] = [0, 0, 70]       # Truck
    colormap[15] = [0, 60, 100]     # Bus
    colormap[16] = [0, 80, 100]     # Train
    colormap[17] = [0, 0, 230]      # Motorcycle
    colormap[18] = [119, 11, 32]    # Bicycle
    # Other classes can be added if necessary

    return colormap


def get_instance_color(semantic_color_rgb, instance_id):
    if instance_id == 0: 
        return semantic_color_rgb

    r_sem, g_sem, b_sem = semantic_color_rgb
    # Convert RGB to HSV. Hue (h) will be preserved.
    h, s, v = colorsys.rgb_to_hsv(r_sem / 255.0, g_sem / 255.0, b_sem / 255.0)

    # Keep the original hue
    new_h = h 

    # Knuth's multiplicative hash constant for pseudo-randomness
    # Ensure instance_id is a Python int or np.int64 to prevent overflow with the constant
    hash_val = int(instance_id) * 2654435761 
    
    # Define the strength of perturbation.
    # This value determines the maximum +/- deviation for S and V.
    perturb_strength = 0.20 # Increased slightly from 0.15. Max deviation for S or V (e.g., +/- 0.20)
    # You can adjust this value: smaller for less difference, larger for more.

    # Generate pseudo-random perturbation factor for S in range [-0.5, 0.5]
    # Using lower 8 bits of the hash for saturation
    s_perturb_factor = (((hash_val >> 0) & 0xFF) / 255.0) - 0.5 
    s_perturb = s_perturb_factor * (2 * perturb_strength) # Scale to [-perturb_strength, perturb_strength]
    
    # Generate pseudo-random perturbation factor for V in range [-0.5, 0.5]
    # Using next 8 bits of the hash for value, to make it more independent from s_perturb
    v_perturb_factor = (((hash_val >> 8) & 0xFF) / 255.0) - 0.5
    v_perturb = v_perturb_factor * (2 * perturb_strength) # Scale to [-perturb_strength, perturb_strength]
    
    # Apply perturbations
    if s < 0.15: # Original is grayscale or very desaturated
        # Give it a base saturation and then perturb slightly
        # Ensure it gets some color, but variations are subtle
        new_s = np.clip(0.3 + s_perturb, 0.15, 0.7) # Base saturation 0.3, range [0.15, 0.7]
    else: # Original has color
        new_s = np.clip(s + s_perturb, 0.15, 1.0) # Min saturation 0.15

    new_v = np.clip(v + v_perturb, 0.15, 1.0) # Min value 0.15
    
    # If the original color was very dark, ensure the perturbed value is not too dark
    if v < 0.2 and new_v < 0.2:
        new_v = 0.2 + abs(v_perturb_factor * 0.2) # Lift the value a bit, scaled by perturbation factor
        new_v = np.clip(new_v, 0.15, 0.4)

    # If the original color was very light (near white), ensure perturbed value is not too white
    if v > 0.85 and new_v > 0.85 and s < 0.2:
        new_v = 0.85 - abs(v_perturb_factor * 0.2) # Lower the value a bit
        new_v = np.clip(new_v, 0.7, 0.9)


    final_r, final_g, final_b = colorsys.hsv_to_rgb(new_h, new_s, new_v) # Corrected to hsv_to_rgb
    
    return (int(final_r * 255), int(final_g * 255), int(final_b * 255))


def encode_panoptic_map(raw_id_map, segments_info, label_divisor):
    if isinstance(raw_id_map, torch.Tensor):
        raw_id_map_np = raw_id_map.cpu().numpy()
    else:
        raw_id_map_np = np.array(raw_id_map) # Ensure it's a NumPy array

    encoded_map = np.zeros_like(raw_id_map_np, dtype=np.int32)

    for s_info in segments_info:
        raw_segment_id = s_info['id']
        category_id = s_info['category_id']
        is_thing = s_info['isthing']

        # The instance part is the raw_segment_id itself if it's a 'thing', else 0.
        # This raw_segment_id serves as a unique instance identifier within the encoded scheme.
        instance_component = raw_segment_id if is_thing else 0
        
        encoded_value = category_id * label_divisor + instance_component
        encoded_map[raw_id_map_np == raw_segment_id] = encoded_value
        
    return encoded_map


def colorize_panoptic_prediction(panoptic_seg, colormap, thing_class_ids=[11, 12, 13, 14, 15, 16, 17, 18], label_divisor=1000):
    if panoptic_seg is None or panoptic_seg.size == 0:
        return np.zeros((10, 10, 3), dtype=np.uint8) # Return a small black image for invalid input
    if panoptic_seg.ndim != 2:
        raise ValueError(f"panoptic_seg must be a 2D array, got shape {panoptic_seg.shape}")

    height, width = panoptic_seg.shape
    output_image = np.zeros((height, width, 3), dtype=np.uint8) # Initialize to black

    panoptic_seg_int = panoptic_seg.astype(np.int32)
    # semantic_map contains the semantic category_id for each pixel
    semantic_map = panoptic_seg_int // label_divisor
    # instance_map contains the instance_component (raw_segment_id for "things", 0 for "stuff")
    instance_map = panoptic_seg_int % label_divisor

    # Iterate over each semantic category present in the image
    unique_semantic_labels_in_image = np.unique(semantic_map)

    for semantic_label in unique_semantic_labels_in_image:
        if not (0 <= semantic_label < len(colormap)): # Safety check for valid semantic_label
            print(f"Warning: Semantic label {semantic_label} is out of colormap bounds. Skipping.")
            continue

        # Mask for all pixels belonging to the current semantic_label
        current_semantic_mask = (semantic_map == semantic_label)
        semantic_color = colormap[semantic_label]

        if semantic_label in thing_class_ids:
            # This is a "thing" class.
            # Get all instance components (raw_segment_ids) associated with this semantic_label's pixels
            instances_for_this_semantic_label = instance_map[current_semantic_mask]
            unique_instance_components = np.unique(instances_for_this_semantic_label)
            
            # Optional: print for debugging
            # print(f"Processing Thing class {semantic_label}. Instance components: {unique_instance_components}")

            for inst_component in unique_instance_components:
                # Create a mask for the specific instance component within the current semantic class
                specific_instance_mask = current_semantic_mask & (instance_map == inst_component)

                if inst_component == 0:
                    # This means it's a "crowd" region for this "thing" class,
                    # or a part of the "thing" class that wasn't assigned a specific instance ID > 0
                    # by the encoding process (e.g., if its raw_segment_id was 0).
                    # Color it with the base semantic color.
                    output_image[specific_instance_mask] = semantic_color
                else:
                    # This is a specific instance of a "thing" class (inst_component is its raw_segment_id).
                    # Generate a distinct color for this instance.
                    instance_color = get_instance_color(semantic_color, inst_component)
                    output_image[specific_instance_mask] = instance_color
        else:
            # This is a "stuff" class. All its pixels get the same semantic color.
            # The instance_map for these pixels should have been 0 from the encoding.
            output_image[current_semantic_mask] = semantic_color
            
    return output_image


def colorize_prediction(prediction, colormap):
    colorized = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)

    for label in range(0, len(colormap)):
        mask = prediction == label
        colorized[mask] = colormap[label]

    return colorized


def compute_mean_iou(preds, gts, num_classes=19, ignore_index=255):
    ious = []
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        gt_mask = (gts == cls)

        # Ignore pixels marked as void/ignore
        valid_mask = (gts != ignore_index)

        pred_mask = pred_mask & valid_mask
        gt_mask = gt_mask & valid_mask

        if np.sum(gt_mask) == 0:
            continue

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()

        if union == 0:
            ious.append(0.0)
        else:
            ious.append(intersection / union)

    return np.mean(ious) if ious else 0.0


def main(args):
    print("Command Line Args:", args)
    cfg = setup(args)
    trt_model_path = args.trt_path
    device = args.device

    if args.eval_only:

        dataset_name = "cityscapes_fine_sem_seg_val"
        test_loader = build_detection_test_loader(cfg, dataset_name=dataset_name)

        model = Trainer.build_model(cfg)
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load('output/model_final.pth')

        model.eval()
        model.to(device)
        shape = (3,1024, 2048)
        image = Image.open("lindau_37.png")
        image.save("output/input.png")
        image_tensor = prepare_image(image, args.sample_factor)
        print("Number of devices: " + str(torch.cuda.device_count()))
        print("Number of current device: " + str(torch.cuda.current_device()))
        print("Using device: " + torch.cuda.get_device_name(torch.cuda.current_device()))

        start_benchmark(args, model, image_tensor, trt_model_path, test_loader, verbose=True)

        # print("KRAJ")
    #     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #         cfg.MODEL.WEIGHTS, resume=args.resume
    #     )
    #     res = Trainer.test(cfg, model)
    #     if cfg.TEST.AUG.ENABLED:
    #         res.update(Trainer.test_with_TTA(cfg, model))
    #     if comm.is_main_process():
    #         verify_results(cfg, res)
    #     return res
    
    # trainer = Trainer(cfg)
    # trainer.resume_or_load(resume=args.resume)
    # return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--device", type=str)
    parser.add_argument("--sample-factor", type=float)
    parser.add_argument("--trt-path", type=str)
    parser.add_argument("--trt-res-path", type=str)
    parser.add_argument("--res-path", type=str)
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
