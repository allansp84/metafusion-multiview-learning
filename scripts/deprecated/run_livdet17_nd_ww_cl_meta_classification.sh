#!/bin/bash

n_models=15
device_number="0"
output_path=tests/working_nov22
date_id=nov22


# nohup mcnnsantispoofing.py --dataset 4 --dataset_path dataset/LivDet-Iris-2017-ND-WW-CL --ground_truth_path "" --iris_location extra/nd_warsaw_cl_osiris_coords.csv --output_path $output_path --descriptor bsif --desc_params "[11,11,8]" --n_jobs 6 --device_number $device_number --meta_classification --meta_ml_algo 1 --meta_classification_from labels --n_models -1 --selection_algo 0 --n_run 0 --force_train --compute_feature_importance > logs/livdet17_nd_ww_cl.$date_id.meta_classification.1.from_labels.selection_algo-0.compute_feature_importance.log 2>&1

# nohup mcnnsantispoofing.py --dataset 4 --dataset_path dataset/LivDet-Iris-2017-ND-WW-CL --ground_truth_path "" --iris_location extra/nd_warsaw_cl_osiris_coords.csv --output_path $output_path --descriptor bsif --desc_params "[11,11,8]" --n_jobs 6 --device_number $device_number --meta_classification --meta_ml_algo 1 --meta_classification_from labels --n_models -1 --selection_algo 1 --n_run 0 --force_train --compute_feature_importance > logs/livdet17_nd_ww_cl.$date_id.meta_classification.1.from_labels.selection_algo-1.compute_feature_importance.log 2>&1

# nohup mcnnsantispoofing.py --dataset 4 --dataset_path dataset/LivDet-Iris-2017-ND-WW-CL --ground_truth_path "" --iris_location extra/nd_warsaw_cl_osiris_coords.csv --output_path $output_path --descriptor bsif --desc_params "[11,11,8]" --n_jobs 6 --device_number $device_number --meta_classification --meta_ml_algo 1 --meta_classification_from labels --n_models $n_models --selection_algo 0 --n_run 0 --force_train > logs/livdet17_nd_ww_cl.$date_id.meta_classification.1.from_labels.selection_algo-0.$n_models.log 2>&1

# nohup mcnnsantispoofing.py --dataset 4 --dataset_path dataset/LivDet-Iris-2017-ND-WW-CL --ground_truth_path "" --iris_location extra/nd_warsaw_cl_osiris_coords.csv --output_path $output_path --descriptor bsif --desc_params "[11,11,8]" --n_jobs 6 --device_number $device_number --meta_classification --meta_ml_algo 0 --meta_classification_from labels --n_models $n_models --selection_algo 0 --n_run 0 --force_train > logs/livdet17_nd_ww_cl.$date_id.meta_classification.0.from_labels.selection_algo-0.$n_models.log 2>&1

nohup mcnnsantispoofing.py --dataset 4 --dataset_path dataset/LivDet-Iris-2017-ND-WW-CL --ground_truth_path "" --iris_location extra/nd_warsaw_cl_osiris_coords.csv --output_path $output_path --descriptor bsif --desc_params "[11,11,8]" --n_jobs 6 --device_number $device_number --meta_classification --meta_ml_algo 1 --meta_classification_from labels --n_models $n_models --selection_algo 1 --n_run 0 --force_train > logs/livdet17_nd_ww_cl.$date_id.meta_classification.1.from_labels.selection_algo-1.$n_models.log 2>&1

# nohup mcnnsantispoofing.py --dataset 4 --dataset_path dataset/LivDet-Iris-2017-ND-WW-CL --ground_truth_path "" --iris_location extra/nd_warsaw_cl_osiris_coords.csv --output_path $output_path --descriptor bsif --desc_params "[11,11,8]" --n_jobs 6 --device_number $device_number --meta_classification --meta_ml_algo 0 --meta_classification_from labels --n_models $n_models --selection_algo 1 --n_run 0 --force_train > logs/livdet17_nd_ww_cl.$date_id.meta_classification.0.from_labels.selection_algo-1.$n_models.log 2>&1
