#!/usr/bin/env bash

python ../train.py --device cuda:0 --batch_size 6 --num_workers 6 --resume 001.pth
