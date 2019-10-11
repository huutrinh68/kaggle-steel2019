#!/usr/bin/env bash

python ../train.py \
--model resnet34 \
--device cuda:1 \
--batch_size 4 \
--num_workers 2 \
--accumulate_step 2
