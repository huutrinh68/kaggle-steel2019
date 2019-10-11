#!/usr/bin/env bash

python ../train.py \
--model efficientnet-b5 \
--device cuda:1 \
--batch_size 30 \
--num_workers 10 \
--accumulate_step 10
