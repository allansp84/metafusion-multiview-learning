#!/bin/bash

device_number="cpu"
date_id=Jan132020
output_path=tests/working_${date_id}
selection_algos="1 2"
meta_ml_algos="0"
folds="1 2 3 4 5"
points="1000"
meta_classification_from_options="labels"
rounds="rodada1 rodada2 rodada3 rodada4 rodada5 rodada6 rodada7 rodada8 rodada9 rodada10"


run_selection(){

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
                        echo -e '>> mcnnsantispoofing.py --dataset 1 --dataset_path dataset/LivDet-Iris-2017-Warsaw --ground_truth_path "" --iris_location extra/warsaw_osiris_coords.csv --output_path ${output_path} --descriptor bsif --desc_params "[11,11,8]" --n_jobs 30 --device_number ${device_number} --meta_classification --meta_ml_algo 1 --meta_classification_from ${meta_classification_from} --n_models -1 --step_algo "selection" --selection_algo ${selection_algo} --n_run 0 --force_train --compute_feature_importance --fold ${fold} --n_points ${n_points} --round ${round} > logs/livdet17_warsaw.${date_id}.meta_classification.1.meta_classification_from-${meta_classification_from}.selection_algo-${selection_algo}.fold-${fold}.n_points-${n_points}.round-${round}.compute_feature_importance.log\n'

                        mcnnsantispoofing.py --dataset 1 --dataset_path dataset/LivDet-Iris-2017-Warsaw --ground_truth_path "" --iris_location extra/warsaw_osiris_coords.csv --output_path ${output_path} --descriptor bsif --desc_params "[11,11,8]" --n_jobs 30 --device_number ${device_number} --meta_classification --meta_ml_algo 1 --meta_classification_from ${meta_classification_from} --n_models -1 --step_algo "selection" --selection_algo ${selection_algo} --n_run 0 --force_train --compute_feature_importance --fold ${fold} --n_points ${n_points} --round ${round} > logs/livdet17_warsaw.${date_id}.meta_classification.1.meta_classification_from-${meta_classification_from}.selection_algo-${selection_algo}.fold-${fold}.n_points-${n_points}.round-${round}.compute_feature_importance.log 2>&1
                    done
                done
            done
        done
    done
}


run_fusion(){

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
                        # for n_models in {44..44..1}
                        for n_models in {65..80..5}
                        do
                            for meta_classification_from in ${meta_classification_from_options}
                            do
                                mkdir -p logs/livdet17_warsaw.$date_id.meta_classification.$meta_ml_algo.meta_classification_from-$meta_classification_from.selection_algo-$selection_algo/n_points-${n_points}.round-${round}/fold-$fold

                                echo -e '>> mcnnsantispoofing.py --dataset 1 --dataset_path dataset/LivDet-Iris-2017-Warsaw --ground_truth_path "" --iris_location extra/warsaw_osiris_coords.csv --output_path ${output_path} --descriptor bsif --desc_params "[11,11,8]" --n_jobs 30 --device_number ${device_number} --meta_classification --meta_ml_algo ${meta_ml_algo} --meta_classification_from ${meta_classification_from} --n_models ${n_models} --step_algo "fusion" --selection_algo ${selection_algo} --n_run 0 --force_train --fold ${fold} --n_points ${n_points} --round ${round} > logs/livdet17_warsaw.$date_id.meta_classification.$meta_ml_algo.meta_classification_from-$meta_classification_from.selection_algo-$selection_algo/n_points-${n_points}.round-${round}/fold-$fold/n_models-$n_models.log\n'

                                mcnnsantispoofing.py --dataset 1 --dataset_path dataset/LivDet-Iris-2017-Warsaw --ground_truth_path "" --iris_location extra/warsaw_osiris_coords.csv --output_path $output_path --descriptor bsif --desc_params "[11,11,8]" --n_jobs 30 --device_number $device_number --meta_classification --meta_ml_algo $meta_ml_algo --meta_classification_from $meta_classification_from --n_models $n_models --step_algo "fusion" --selection_algo $selection_algo --n_run 0 --force_train --fold $fold --n_points ${n_points} --round ${round} > logs/livdet17_warsaw.$date_id.meta_classification.$meta_ml_algo.meta_classification_from-$meta_classification_from.selection_algo-$selection_algo/n_points-${n_points}.round-${round}/fold-$fold/n_models-$n_models.log 2>&1

                            done
                        done
                    done
                done
            done
        done
    done
}

run_selection
run_fusion


# find logs/livdet17_warsaw.May102019.meta_classification.0.meta_classification_from-labels.selection_algo-1/n_points-500.round-rodada10/fold-*/n_models-*.log | xargs grep -r "| test" > aux.txt

# grep -r -e "| test" n_points-500.round-rodada1/fold-*/n_models-*.log | cut -d'|' -f6 | wc -l

# grep -rl -e "| test" n_points-500.round-rodada1/fold-*/n_models-*.log


# find tests/working_Sept232019/intra-test/livdetiris17_warsaw/ -iname '*meta_classification' -exec mv {} {}_v2 \;