#!/bin/bash

echo 'PWD:' `pwd`

device_number="1"

# --  11x11
mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path tests/working_nov06_andreysmodel --descriptor bsif --desc_params "[11,11,10]" --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 6 --device_number $device_number --show_results > logs/livdet17.nov06_andreysmodel.bsif."[11,11,10]".0.log

mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path tests/working_nov06_andreysmodel --descriptor bsif --desc_params "[11,11,11]" --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 6 --device_number $device_number --show_results > logs/livdet17.nov06_andreysmodel.bsif."[11,11,11]".0.log

mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path tests/working_nov06_andreysmodel --descriptor bsif --desc_params "[11,11,12]" --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 6 --device_number $device_number --show_results > logs/livdet17.nov06_andreysmodel.bsif."[11,11,12]".0.log


# --  13x13
mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path tests/working_nov06_andreysmodel --descriptor bsif --desc_params "[13,13,5]" --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 6 --device_number $device_number --show_results > logs/livdet17.nov06_andreysmodel.bsif."[13,13,5]".0.log

mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path tests/working_nov06_andreysmodel --descriptor bsif --desc_params "[13,13,6]" --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 6 --device_number $device_number --show_results > logs/livdet17.nov06_andreysmodel.bsif."[13,13,6]".0.log

mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path tests/working_nov06_andreysmodel --descriptor bsif --desc_params "[13,13,7]" --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 6 --device_number $device_number --show_results > logs/livdet17.nov06_andreysmodel.bsif."[13,13,7]".0.log

mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path tests/working_nov06_andreysmodel --descriptor bsif --desc_params "[13,13,8]" --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 6 --device_number $device_number --show_results > logs/livdet17.nov06_andreysmodel.bsif."[13,13,8]".0.log

mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path tests/working_nov06_andreysmodel --descriptor bsif --desc_params "[13,13,9]" --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 6 --device_number $device_number --show_results > logs/livdet17.nov06_andreysmodel.bsif."[13,13,9]".0.log

mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path tests/working_nov06_andreysmodel --descriptor bsif --desc_params "[13,13,12]" --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 6 --device_number $device_number --show_results > logs/livdet17.nov06_andreysmodel.bsif."[13,13,12]".0.log


# --  15x15
mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path tests/working_nov06_andreysmodel --descriptor bsif --desc_params "[15,15,5]" --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 6 --device_number $device_number --show_results > logs/livdet17.nov06_andreysmodel.bsif."[15,15,5]".0.log

mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path tests/working_nov06_andreysmodel --descriptor bsif --desc_params "[15,15,6]" --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 6 --device_number $device_number --show_results > logs/livdet17.nov06_andreysmodel.bsif."[15,15,6]".0.log

mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path tests/working_nov06_andreysmodel --descriptor bsif --desc_params "[15,15,7]" --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 6 --device_number $device_number --show_results > logs/livdet17.nov06_andreysmodel.bsif."[15,15,7]".0.log

mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path tests/working_nov06_andreysmodel --descriptor bsif --desc_params "[15,15,8]" --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 6 --device_number $device_number --show_results > logs/livdet17.nov06_andreysmodel.bsif."[15,15,8]".0.log

mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path tests/working_nov06_andreysmodel --descriptor bsif --desc_params "[15,15,9]" --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 6 --device_number $device_number --show_results > logs/livdet17.nov06_andreysmodel.bsif."[15,15,9]".0.log

mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path tests/working_nov06_andreysmodel --descriptor bsif --desc_params "[15,15,10]" --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 6 --device_number $device_number --show_results > logs/livdet17.nov06_andreysmodel.bsif."[15,15,10]".0.log

mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path tests/working_nov06_andreysmodel --descriptor bsif --desc_params "[15,15,11]" --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 6 --device_number $device_number --show_results > logs/livdet17.nov06_andreysmodel.bsif."[15,15,11]".0.log

mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path tests/working_nov06_andreysmodel --descriptor bsif --desc_params "[15,15,12]" --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 6 --device_number $device_number --show_results > logs/livdet17.nov06_andreysmodel.bsif."[15,15,12]".0.log


# --  17x17
mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path tests/working_nov06_andreysmodel --descriptor bsif --desc_params "[17,17,10]" --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 6 --device_number $device_number --show_results > logs/livdet17.nov06_andreysmodel.bsif."[17,17,10]".0.log

mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path tests/working_nov06_andreysmodel --descriptor bsif --desc_params "[17,17,11]" --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 6 --device_number $device_number --show_results > logs/livdet17.nov06_andreysmodel.bsif."[17,17,11]".0.log

mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path tests/working_nov06_andreysmodel --descriptor bsif --desc_params "[17,17,12]" --operation segment --max_axis 260 --bs 32 --epochs 70 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 6 --device_number $device_number --show_results > logs/livdet17.nov06_andreysmodel.bsif."[17,17,12]".0.log
