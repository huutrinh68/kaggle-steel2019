#!/usr/bin/env bash

python ../train.py \
--arch unet \
--model efficientnet-b5 \
--device cuda:1 \
--batch_size 20 \
--num_workers 10 \
--accumulate_step 10
