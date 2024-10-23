# [ACM MM 2024]Synergetic Prototype Learning for Unbiased Scene Graph Generation

This repository contains the official code implementation for the paper [Synergetic Prototype Learning for Unbiased Scene Graph Generation](https://openreview.net/forum?id=up4C6pO1Vw).

## Installation
Check [INSTALL.md](./INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](./DATASET.md) for instructions of dataset preprocessing.

## Train
We provide [scripts](./scripts/train.sh) for training the models
```
# For LLAMA
python3 \
  tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1\
  SOLVER.MAX_ITER 60000 SOLVER.BASE_LR 5e-4 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD 5000 \
  SOLVER.CHECKPOINT_PERIOD 5000 GLOVE_DIR ./datasets/vg/ \
  MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
  SOLVER.PRE_VAL False \
  SOLVER.GRAD_NORM_CLIP 5.0 \
  ROI_RELATION_HEAD.VIS_PRO_DIM 4096 \
  ROI_RELATION_HEAD.STORE_UNION False \
  ROI_RELATION_HEAD.VIS_VAE_LATENT_DIM 1024 \
  ROI_RELATION_HEAD.VIS_VAE_INPUT_DIM 4096 \
  ROI_RELATION_HEAD.VIS_PRO_PATH './output/relation_baseline/vg_vis_protos.pt' \
  MODEL.ROI_RELATION_HEAD.CON_PRO_DIM 4096 \ 
  MODEL.ROI_RELATION_HEAD.CON_VAE_LATENT_DIM 1024 \
  MODEL.ROI_RELATION_HEAD.CON_VAE_INPUT_DIM 4096 \
  MODEL.ROI_RELATION_HEAD.CON_PRO_TYPE 'LLAMA' \
  MODEL.ROI_RELATION_HEAD.CON_PRO_CLIP_PATH './output/relation_baseline/vg_clip_refine_con_protos.pt' \
  MODEL.ROI_RELATION_HEAD.CON_PRO_LLAMA_PATH './output/relation_baseline/vg_llama_con_protos.pt' \
  MODEL.USE_REWEIGHT 'PENET'

# For CLIP
python3 \
  tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 \
  SOLVER.MAX_ITER 60000 SOLVER.BASE_LR 5e-4 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD 5000 \
  SOLVER.CHECKPOINT_PERIOD 5000 GLOVE_DIR ./datasets/vg/ \
  MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
  SOLVER.PRE_VAL False \
  SOLVER.GRAD_NORM_CLIP 5.0 \
  ROI_RELATION_HEAD.VIS_PRO_DIM 4096 \
  ROI_RELATION_HEAD.STORE_UNION False \
  ROI_RELATION_HEAD.VIS_VAE_LATENT_DIM 1024 \
  ROI_RELATION_HEAD.VIS_VAE_INPUT_DIM 4096 \
  ROI_RELATION_HEAD.VIS_PRO_PATH './output/relation_baseline/vg_vis_protos.pt' \
  MODEL.ROI_RELATION_HEAD.CON_PRO_DIM 2304 \ 
  MODEL.ROI_RELATION_HEAD.CON_VAE_LATENT_DIM 576 \
  MODEL.ROI_RELATION_HEAD.CON_VAE_INPUT_DIM 2304 \
  MODEL.ROI_RELATION_HEAD.CON_PRO_TYPE 'CLIP' \
  MODEL.ROI_RELATION_HEAD.CON_PRO_CLIP_PATH './output/relation_baseline/vg_clip_refine_con_protos.pt' \
  MODEL.ROI_RELATION_HEAD.CON_PRO_LLAMA_PATH './output/relation_baseline/vg_llama_con_protos.pt' \
  MODEL.USE_REWEIGHT 'PENET'


```

The extracted visual and conceptual prototype files can be downloaded from [here](https://1drv.ms/f/c/60174365786eb250/Eh8cWd-RHsNJrX0O2VR1eVcB1IRe12_OaPTngBqSvBbGtg?e=ANjy7L). Make sure they are in the right path.

## Device

All our experiments are conducted on one NVIDIA GeForce RTX 4090, if you wanna run it on your own device, make sure to follow distributed training instructions in [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

## Acknowledgement

The code is implemented based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) and [PENET](https://github.com/VL-Group/PENET).

## Citation

```
@inproceedings{zhang2024synergetic,
  title={Synergetic Prototype Learning Network for Unbiased Scene Graph Generation},
  author={Zhang, Ruonan and Shang, Ziwei and Wang, Fengjuan and Yang, Zhaoqilin and Cao, Shan and Cen, Yigang and An, Gaoyun},
  booktitle={ACM Multimedia 2024}
}
```
