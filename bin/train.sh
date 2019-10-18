#!/usr/bin/env bash

python ../train.py \
--arch unet \
--model efficientnet-b7 \
--loss_type bcedice \
--device cuda:2 \
--batch_size 4 \
--num_workers 10 \
--accumulate_step 10
