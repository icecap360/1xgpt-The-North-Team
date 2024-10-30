#!/bin/bash  
export CUDA_VISIBLE_DEVICES=1
python train.py --genie_config genie/configs/magvit_n32_h8_d256.json --train_data_dir ../data/train_v1.1 --val_data_dir ../data/val_v1.1 --log_name train_1 --output_dir logs --max_eval_steps 10 --seed 5 --resume_from_checkpoint /pub0/qasim/1xgpt/1xgpt/checkpoints/GENIE_35M --eval_every_n_steps 10000000000 --learning_rate 0.00003 --per_device_train_batch_size 12
# /pub0/qasim/1xgpt/1xgpt/logs/train_1-10-26-14-09/step_20000 
# python train.py --genie_config genie/configs/magvit_n32_h8_d256_v1.json --train_data_dir ../data/train_v1.1 --val_data_dir ../data/val_v1.1 --log_name train_1 --output_dir logs --max_eval_steps 10 --seed 1 --resume_from_checkpoint /pub0/qasim/1xgpt/1xgpt/checkpoints/GENIE_35M
