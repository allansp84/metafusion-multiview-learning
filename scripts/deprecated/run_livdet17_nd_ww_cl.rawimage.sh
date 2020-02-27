#!/bin/bash

echo 'PWD:' `pwd`

device_number="0,1"


mcnnsantispoofing.py --dataset 4 --dataset_path dataset/LivDet-Iris-2017-ND-WW-CL --ground_truth_path "" --iris_location extra/nd_warsaw_cl_osiris_coords.csv --output_path tests/working_nov22 --descriptor RawImage --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 6 --device_number $device_number --classification > logs/livdet17_nd_ww_cl.nov22.rawimage.0.log

