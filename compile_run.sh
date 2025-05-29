python train_net.py \
  --config-file configs/cityscapes/panoptic-segmentation/maskformer2_R50_bs16_90k.yaml \
  --sample-factor 1 \
  --eval-only \
  --compile \
  --trt-path model.onnx --res-path output/predictions_cuda.png --trt-res-path output/predictions_trt.png > compile_log.txt