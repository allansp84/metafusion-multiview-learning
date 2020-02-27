#!/bin/bash

device_number="0,1"
output_path=tests/working_nov22
date_id=nov22
selection_algos="1 "
meta_ml_algos="0 "
meta_classification_from_options="scores "


run_livdet17_nd_meta_classification(){

    for meta_ml_algo in $meta_ml_algos
    do
        for selection_algo in $selection_algos
        do
            for n_models in {16..16}
            do
                for meta_classification_from in $meta_classification_from_options
                do
                    nohup mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path $output_path --descriptor bsif --desc_params "[11,11,8]" --n_jobs 6 --device_number $device_number --meta_classification --meta_ml_algo $meta_ml_algo --meta_classification_from $meta_classification_from --n_models $n_models --selection_algo $selection_algo --n_run 0 --force_train > logs/livdet17_nd.$date_id.meta_classification.$meta_ml_algo.meta_classification_from_$meta_classification_from.selection_algo-$selection_algo.$n_models.log 2>&1
                done
            done
        done
    done
}


for selection_algo in $selection_algos
do
    for meta_classification_from in $meta_classification_from_options
    do
        nohup mcnnsantispoofing.py --dataset 0 --dataset_path dataset/NDCLD15 --ground_truth_path "" --iris_location extra/NDCLD15-with-OSIRIS-segmentation.csv/NDCLD15-with-OSIRIS-segmentation.csv --output_path $output_path --descriptor bsif --desc_params "[11,11,8]" --n_jobs 6 --device_number $device_number --meta_classification --meta_ml_algo 1 --meta_classification_from $meta_classification_from --n_models -1 --selection_algo $selection_algo --n_run 0 --force_train --compute_feature_importance > logs/livdet17_nd.$date_id.meta_classification.1.meta_classification_from_$meta_classification_from.selection_algo-$selection_algo.compute_feature_importance.log 2>&1
    done
done


run_livdet17_nd_meta_classification

