#!/usr/bin/env bash

python ../train.py \
--arch unet \
--model efficientnet-b7 \
--loss_type bcelogit \
--device cuda:1 \
--batch_size 4 \
--num_workers 5 \
--accumulate_step 10
