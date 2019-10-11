#!/usr/bin/env bash

python ../train.py \
--arch fpn \
--model efficientnet-b5 \
--device cuda:2 \
--batch_size 50 \
--num_workers 10 \
--accumulate_step 10
