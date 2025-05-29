python train_net.py \
  --config-file configs/cityscapes/panoptic-segmentation/maskformer2_R50_bs16_90k.yaml \
  --sample-factor 1 \
  --eval-only \
  --cuda \
  --trt-path model_simplified.onnx --res-path output/predictions_cuda.png --trt-res-path output/predictions_trt.png