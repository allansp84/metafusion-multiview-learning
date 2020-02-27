#!/bin/bash

echo 'PWD:' `pwd`

device_number="0,1"


mcnnsantispoofing.py --dataset 5 --dataset_path dataset/LivDet17_IIIT_WVU --iris_location extra/iiitwvu-osiris_coords.csv --output_path tests/working_nov22 --descriptor RawImage --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --classification > logs/livdet17_iiitwvu.nov22.rawimage.0.log
