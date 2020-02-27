#!/bin/bash

echo 'PWD:' `pwd`

device_number="0,1"


mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor RawImage --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --classification > logs/livdet17_combined.nov22.rawimage.0.log
