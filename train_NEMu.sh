#!/bin/bash
python train.py --seed 1 --name NEMu_1 --dataset NEMu --class_nums 12 \
--n_en 8,4,8 --data_dims 300,768,74 --embed_dims 480,240,480 \
--encoder_heads 12,8,12 --dim_ff_ratio 2 --n_de 4 --d_model 256 --n_head 8 --t_length 1650,196,1050 --t_mask y \
--alpha 0 --model  MLPModel --gpu 0 \
--weight_decay 0 --learning_rate 5e-5 --train_batch_size 4 --gradient_accumulation_steps 128 --load_checkpoint 0 \
--wloss 1 --wloss_alpha_pos 0.5 --wloss_beta_pos 1 --wloss_alpha_neg 1 --wloss_beta_neg 2 \
--lr_factor 0.5 --num_epochs 100 \
--loss lq_loss --lq_pos_h 0.01 --lq_pos_e 0.1 --lq_neg_h 1 --lq_neg_e 1 \
--fusion sum \
--d_reg -0.1 --d_reg_type feats --d_reg_fun cos_d \
--wfeats 0.1 \
--warmup 20