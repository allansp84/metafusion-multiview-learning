#!/bin/bash

device_number="0"
output_path=tests/working_nov22
date_id=nov22
selection_algos="1 "
meta_ml_algos="0 1"


run_livdet17_nd_ww_meta_classification_grid_search(){

    for meta_ml_algo in $meta_ml_algos
    do
        for selection_algo in $selection_algos
        do
            for n_models in {5..20}
            do
                nohup mcnnsantispoofing.py --dataset 2 --dataset_path dataset/LivDet-Iris-2017-ND-Warsaw --ground_truth_path "" --iris_location extra/nd_warsaw_osiris_coords.csv --output_path $output_path --descriptor bsif --desc_params "[11,11,8]" --n_jobs 6 --device_number $device_number --meta_classification --meta_ml_algo $meta_ml_algo --meta_classification_from labels --n_models $n_models --selection_algo $selection_algo --n_run 0 > logs/livdet17_nd_ww.$date_id.meta_classification.$meta_ml_algo.from_labels.selection_algo-$selection_algo.$n_models.log 2>&1
            done
        done
    done
}


for selection_algo in $selection_algos
do
    nohup mcnnsantispoofing.py --dataset 2 --dataset_path dataset/LivDet-Iris-2017-ND-Warsaw --ground_truth_path "" --iris_location extra/nd_warsaw_osiris_coords.csv --output_path $output_path --descriptor bsif --desc_params "[11,11,8]" --n_jobs 6 --device_number $device_number --meta_classification --meta_ml_algo 1 --meta_classification_from labels --n_models -1 --selection_algo $selection_algo --n_run 0 --force_train --compute_feature_importance > logs/livdet17_nd_ww.$date_id.meta_classification.1.from_labels.selection_algo-$selection_algo.compute_feature_importance.log 2>&1
done


run_livdet17_nd_ww_meta_classification_grid_search

