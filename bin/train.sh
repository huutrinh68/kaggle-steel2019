#!/usr/bin/env bash

python ../train.py \
--model efficientnet-b5 \
--encoder_name efficientnet-b5 \
--device cuda:1 \
--batch_size 10 \
--num_workers 10
