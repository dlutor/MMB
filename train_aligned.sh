#!/bin/bash
python train.py --seed 1 --name Aligned_1 \
--n_en 8,4,4 --embed_dims 512,320,480 --n_de 4 --d_model 224 --n_head 8 --dim_ff_ratio 4 \
--alpha 0 --model MLPModel  --gpu 0 \
--weight_decay 0 --learning_rate 5e-5 --train_batch_size 512 --load_checkpoint 0 \
--wloss 1 --wloss_alpha_pos 0.5 --wloss_beta_pos 1 --wloss_alpha_neg 1 --wloss_beta_neg 2 \
--lr_factor 0.5 --num_epochs 30 \
--loss lq_loss --lq_pos_h 0.01 --lq_pos_e 0.1 --lq_neg_h 1 --lq_neg_e 1 \
--fusion sum \
--d_reg -0.1 --d_reg_type feats --d_reg_fun cos_d \
--wfeats 0.1 \
--warmup 10 \
--verbose y
