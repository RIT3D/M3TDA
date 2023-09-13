#!/bin/bash
# rm -rf ./configs/generated/* ./work_dirs/* 
CUDA_VISIBLE_DEVICES=0 python run_mtda.py \
--exp 81 \
--region-consis \
--region-masking \
--image-consis \
--image-masking