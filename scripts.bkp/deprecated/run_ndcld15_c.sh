#!/bin/bash

echo 'PWD:' `pwd`

nohup mcnnsantispoofing.py --dataset 1 --dataset_path dataset/NDCLD15 \
                           --ground_truth_path extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
                           --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
                           --operation crop --max_axis 260 \
                           --epochs 600 --bs 32 --lr 0.01 --decay 0.0 --loss_function 0 --optimizer 3 --reg 0.1 \
                           --output_path working_Aug072017 \
                           --device_number 2 \
                           --classification --fold 6 > logs/NDCLD15.Aug072017.fold.6.log

nohup mcnnsantispoofing.py --dataset 1 --dataset_path dataset/NDCLD15 \
                           --ground_truth_path extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
                           --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
                           --operation crop --max_axis 260 \
                           --epochs 600 --bs 32 --lr 0.01 --decay 0.0 --loss_function 0 --optimizer 3 --reg 0.1 \
                           --output_path working_Aug072017 \
                           --device_number 3 \
                           --classification --fold 7 > logs/NDCLD15.Aug072017.fold.7.log

nohup mcnnsantispoofing.py --dataset 1 --dataset_path dataset/NDCLD15 \
                           --ground_truth_path extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
                           --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
                           --operation crop --max_axis 260 \
                           --epochs 600 --bs 32 --lr 0.01 --decay 0.0 --loss_function 0 --optimizer 3 --reg 0.1 \
                           --output_path working_Aug072017 \
                           --device_number 3 \
                           --classification --fold 8 > logs/NDCLD15.Aug072017.fold.8.log

nohup mcnnsantispoofing.py --dataset 1 --dataset_path dataset/NDCLD15 \
                           --ground_truth_path extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
                           --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
                           --operation crop --max_axis 260 \
                           --epochs 250 --bs 32 --lr 0.01 --decay 0.0 --loss_function 0 --optimizer 3 --reg 0.1 \
                           --output_path working_Aug072017 \
                           --device_number 3 \
                           --classification --fold 9 > logs/NDCLD15.Aug072017.fold.9.log
