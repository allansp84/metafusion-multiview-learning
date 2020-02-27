#!/bin/bash

echo 'PWD:' `pwd`


nohup mcnnsantispoofing.py --dataset 1 --dataset_path dataset/NDCLD15 \
                           --ground_truth_path extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
                           --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
                           --operation crop --max_axis 260 \
                           --epochs 600 --bs 32 --lr 0.01 --decay 0.0 --loss_function 0 --optimizer 3 --reg 0.1 \
                           --output_path working_Aug072017 \
                           --device_number 1 \
                           --classification --fold 3 > logs/NDCLD15.Aug072017.fold.3.log

nohup mcnnsantispoofing.py --dataset 1 --dataset_path dataset/NDCLD15 \
                           --ground_truth_path extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
                           --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
                           --operation crop --max_axis 260 \
                           --epochs 600 --bs 32 --lr 0.01 --decay 0.0 --loss_function 0 --optimizer 3 --reg 0.1 \
                           --output_path working_Aug072017 \
                           --device_number 2 \
                           --classification --fold 4 > logs/NDCLD15.Aug072017.fold.4.log

nohup mcnnsantispoofing.py --dataset 1 --dataset_path dataset/NDCLD15 \
                           --ground_truth_path extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
                           --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv \
                           --operation crop --max_axis 260 \
                           --epochs 600 --bs 32 --lr 0.01 --decay 0.0 --loss_function 0 --optimizer 3 --reg 0.1 \
                           --output_path working_Aug072017 \
                           --device_number 2 \
                           --classification --fold 5 > logs/NDCLD15.Aug072017.fold.5.log
