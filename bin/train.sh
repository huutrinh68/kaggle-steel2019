#!/usr/bin/env bash

python ../train.py \
--arch fpn \
--model efficientnet-b5 \
--loss_type bcedice \
--device cuda:2 \
--batch_size 10 \
--num_workers 10 \
--accumulate_step 10
