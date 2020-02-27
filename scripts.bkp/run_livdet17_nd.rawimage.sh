#!/bin/bash

echo 'PWD:' `pwd`

device_number="0"


mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path tests/working_nov22 --descriptor RawImage --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 6 --device_number $device_number --classification > logs/livdet17_nd.nov22.rawimage.0.log
