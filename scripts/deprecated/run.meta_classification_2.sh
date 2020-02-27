#!/bin/bash

device_number="cpu"
output_path=tests/working_May102019
date_id=May102019
selection_algos="1"
meta_ml_algos="0"
folds="1 2 3 4 5"
points="250"
meta_classification_from_options="labels"
rounds="rodada7 rodada8 rodada9 rodada10"

run_livdet17_warsaw_meta_classification(){

    for round in ${rounds}
    do
        for n_points in ${points}
        do
            for fold in ${folds}
            do
                for meta_ml_algo in ${meta_ml_algos}
                do
                    for selection_algo in ${selection_algos}
                    do
                        for n_models in {05..31..1}
                        do
                            for meta_classification_from in ${meta_classification_from_options}
                            do
                                mkdir -p logs/livdet17_warsaw.$date_id.meta_classification.$meta_ml_algo.meta_classification_from-$meta_classification_from.selection_algo-$selection_algo/n_points-${n_points}.round-${round}/fold-$fold

                                echo -e ">> mcnnsantispoofing.py --dataset 1 --dataset_path dataset/LivDet-Iris-2017-Warsaw --ground_truth_path "" --iris_location extra/warsaw_osiris_coords.csv --output_path ${output_path} --descriptor bsif --desc_params "[11,11,8]" --n_jobs 6 --device_number ${device_number} --meta_classification --meta_ml_algo ${meta_ml_algo} --meta_classification_from ${meta_classification_from} --n_models ${n_models} --selection_algo ${selection_algo} --n_run 0 --force_train --fold ${fold} --n_points ${n_points} --round ${round} > logs/livdet17_warsaw.$date_id.meta_classification.$meta_ml_algo.meta_classification_from-$meta_classification_from.selection_algo-$selection_algo/n_points-${n_points}.round-${round}/fold-$fold/n_models-$n_models.log \n"

                                mcnnsantispoofing.py --dataset 1 --dataset_path dataset/LivDet-Iris-2017-Warsaw --ground_truth_path "" --iris_location extra/warsaw_osiris_coords.csv --output_path $output_path --descriptor bsif --desc_params "[11,11,8]" --n_jobs 6 --device_number $device_number --meta_classification --meta_ml_algo $meta_ml_algo --meta_classification_from $meta_classification_from --n_models $n_models --selection_algo $selection_algo --n_run 0 --force_train --fold $fold --n_points ${n_points} --round ${round} > logs/livdet17_warsaw.$date_id.meta_classification.$meta_ml_algo.meta_classification_from-$meta_classification_from.selection_algo-$selection_algo/n_points-${n_points}.round-${round}/fold-$fold/n_models-$n_models.log 2>&1

                            done
                        done
                    done
                done
            done
        done
    done
}

for round in ${rounds}
do
    for n_points in ${points}
    do
        for fold in ${folds}
        do
            for selection_algo in ${selection_algos}
            do
                for meta_classification_from in ${meta_classification_from_options}
                do
                    echo -e ">> mcnnsantispoofing.py --dataset 1 --dataset_path dataset/LivDet-Iris-2017-Warsaw --ground_truth_path "" --iris_location extra/warsaw_osiris_coords.csv --output_path ${output_path} --descriptor bsif --desc_params "[11,11,8]" --n_jobs 6 --device_number ${device_number} --meta_classification --meta_ml_algo 1 --meta_classification_from ${meta_classification_from} --n_models -1 --selection_algo ${selection_algo} --n_run 0 --force_train --compute_feature_importance --fold ${fold} --n_points ${n_points} --round ${round} > logs/livdet17_warsaw.${date_id}.meta_classification.1.meta_classification_from-${meta_classification_from}.selection_algo-${selection_algo}.fold-${fold}.n_points-${n_points}.round-${round}.compute_feature_importance.log \n"

                    mcnnsantispoofing.py --dataset 1 --dataset_path dataset/LivDet-Iris-2017-Warsaw --ground_truth_path "" --iris_location extra/warsaw_osiris_coords.csv --output_path ${output_path} --descriptor bsif --desc_params "[11,11,8]" --n_jobs 6 --device_number ${device_number} --meta_classification --meta_ml_algo 1 --meta_classification_from ${meta_classification_from} --n_models -1 --selection_algo ${selection_algo} --n_run 0 --force_train --compute_feature_importance --fold ${fold} --n_points ${n_points} --round ${round} > logs/livdet17_warsaw.${date_id}.meta_classification.1.meta_classification_from-${meta_classification_from}.selection_algo-${selection_algo}.fold-${fold}.n_points-${n_points}.round-${round}.compute_feature_importance.log 2>&1
                done
            done
        done
    done
done


run_livdet17_warsaw_meta_classification

