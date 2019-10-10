#!/usr/bin/env bash

python ../train.py \
--model resnet18 \
--device cuda:2 \
--batch_size 4 \
--num_workers 2
