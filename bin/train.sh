#!/usr/bin/env bash

python ../train.py \
--arch fpn \
--model efficientnet-b5 \
--device cuda:1 \
--batch_size 32 \
--num_workers 10 \
--accumulate_step 10
