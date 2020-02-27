#!/bin/bash

echo 'PWD:' `pwd`

nohup mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 \
                           --ground_truth_path dataset/full_dataset.csv \
                           --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
                           --operation crop --max_axis 260 \
                           --epochs 600 --bs 32 --lr 0.01 --decay 0.0 --loss_function 0 --optimizer 3 --reg 0.1 \
                           --output_path working_Aug072017 \
                           --device_number 1 \
                           --classification --fold 0 > logs/livdet17.Aug072017.0.log &

#nohup mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 \
#                           --ground_truth_path dataset/full_dataset.csv \
#                           --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
#                           --operation crop --max_axis 260 \
#                           --epochs 600 --bs 32 --lr 0.01 --decay 0.0 --loss_function 0 --optimizer 3 --reg 0.1 \
#                           --output_path working_Aug072017 \
#                           --device_number 1 \
#                           --classification --fold 1 > logs/livdet17.Aug072017.1.log
#
#nohup mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 \
#                           --ground_truth_path dataset/full_dataset.csv \
#                           --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
#                           --operation crop --max_axis 260 \
#                           --epochs 600 --bs 32 --lr 0.01 --decay 0.0 --loss_function 0 --optimizer 3 --reg 0.1 \
#                           --output_path working_Aug072017 \
#                           --device_number 1 \
#                           --classification --fold 2 > logs/livdet17.Aug072017.2.log
#
#nohup mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 \
#                           --ground_truth_path dataset/full_dataset.csv \
#                           --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
#                           --operation crop --max_axis 260 \
#                           --epochs 600 --bs 32 --lr 0.01 --decay 0.0 --loss_function 0 --optimizer 3 --reg 0.1 \
#                           --output_path working_Aug072017 \
#                           --device_number 1 \
#                           --classification --fold 3 > logs/livdet17.Aug072017.3.log
#
#nohup mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 \
#                           --ground_truth_path dataset/full_dataset.csv \
#                           --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
#                           --operation crop --max_axis 260 \
#                           --epochs 600 --bs 32 --lr 0.01 --decay 0.0 --loss_function 0 --optimizer 3 --reg 0.1 \
#                           --output_path working_Aug072017 \
#                           --device_number 1 \
#                           --classification --fold 4 > logs/livdet17.Aug072017.4.log
#
#nohup mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 \
#                           --ground_truth_path dataset/full_dataset.csv \
#                           --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
#                           --operation crop --max_axis 260 \
#                           --epochs 600 --bs 32 --lr 0.01 --decay 0.0 --loss_function 0 --optimizer 3 --reg 0.1 \
#                           --output_path working_Aug072017 \
#                           --device_number 1 \
#                           --classification --fold 5 > logs/livdet17.Aug072017.5.log
#
#nohup mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 \
#                           --ground_truth_path dataset/full_dataset.csv \
#                           --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
#                           --operation crop --max_axis 260 \
#                           --epochs 600 --bs 32 --lr 0.01 --decay 0.0 --loss_function 0 --optimizer 3 --reg 0.1 \
#                           --output_path working_Aug072017 \
#                           --device_number 1 \
#                           --classification --fold 6 > logs/livdet17.Aug072017.6.log
#
#nohup mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 \
#                           --ground_truth_path dataset/full_dataset.csv \
#                           --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
#                           --operation crop --max_axis 260 \
#                           --epochs 600 --bs 32 --lr 0.01 --decay 0.0 --loss_function 0 --optimizer 3 --reg 0.1 \
#                           --output_path working_Aug072017 \
#                           --device_number 1 \
#                           --classification --fold 7 > logs/livdet17.Aug072017.7.log
#
#nohup mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 \
#                           --ground_truth_path dataset/full_dataset.csv \
#                           --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
#                           --operation crop --max_axis 260 \
#                           --epochs 600 --bs 32 --lr 0.01 --decay 0.0 --loss_function 0 --optimizer 3 --reg 0.1 \
#                           --output_path working_Aug072017 \
#                           --device_number 1 \
#                           --classification --fold 8 > logs/livdet17.Aug072017.8.log
#
#nohup mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 \
#                           --ground_truth_path dataset/full_dataset.csv \
#                           --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
#                           --operation crop --max_axis 260 \
#                           --epochs 600 --bs 32 --lr 0.01 --decay 0.0 --loss_function 0 --optimizer 3 --reg 0.1 \
#                           --output_path working_Aug072017 \
#                           --device_number 1 \
#                           --classification --fold 9 > logs/livdet17.Aug072017.9.log
