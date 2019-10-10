#!/usr/bin/env bash

python ../train.py \
--model efficientnet-b5 \
--device cuda:1 \
--batch_size 6 \
--num_workers 10
