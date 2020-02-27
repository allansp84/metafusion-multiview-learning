#!/bin/bash
# max_axis-260-epochs-140-bs-32-losses-categorical_hinge-lr-0.001-decay-0.0-optimizer-Adam-reg-0.1-fold-0
echo 'PWD:' `pwd`

device_number="0,1"

# -- 3,3
mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[3,3,5]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."3,3,5".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[3,3,6]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."3,3,6".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[3,3,7]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."3,3,7".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[3,3,8]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."3,3,8".0.log


# --  5,5
mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[5,5,5]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."5,5,5".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[5,5,6]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."5,5,6".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[5,5,7]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."5,5,7".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[5,5,8]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."5,5,8".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[5,5,9]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."5,5,9".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[5,5,10]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."5,5,10".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[5,5,11]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."5,5,11".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[5,5,12]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."5,5,12".0.log


# --  7,7
mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[7,7,5]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."7,7,5".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[7,7,6]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."7,7,6".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[7,7,7]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."7,7,7".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[7,7,8]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."7,7,8".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[7,7,9]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."7,7,9".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[7,7,10]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."7,7,10".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[7,7,11]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."7,7,11".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[7,7,12]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."7,7,12".0.log


# --  9,9
mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[9,9,5]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."9,9,5".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[9,9,6]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."9,9,6".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[9,9,7]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."9,9,7".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[9,9,8]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."9,9,8".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[9,9,9]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."9,9,9".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[9,9,10]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."9,9,10".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[9,9,11]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."9,9,11".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[9,9,12]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."9,9,12".0.log


# --  11,11
mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[11,11,5]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."11,11,5".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[11,11,6]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."11,11,6".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[11,11,7]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."11,11,7".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[11,11,8]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."11,11,8".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[11,11,9]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."11,11,9".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[11,11,10]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."11,11,10".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[11,11,11]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."11,11,11".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[11,11,12]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."11,11,12".0.log





# --  13,13
mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[13,13,5]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."13,13,5".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[13,13,6]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."13,13,6".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[13,13,7]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."13,13,7".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[13,13,8]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."13,13,8".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[13,13,9]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."13,13,9".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[13,13,10]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."13,13,10".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[13,13,11]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."13,13,11".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[13,13,12]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."13,13,12".0.log




# --  15,15
mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[15,15,5]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."15,15,5".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[15,15,6]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."15,15,6".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[15,15,7]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."15,15,7".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[15,15,8]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."15,15,8".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[15,15,9]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."15,15,9".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[15,15,10]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."15,15,10".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[15,15,11]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."15,15,11".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[15,15,12]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."15,15,12".0.log



# --  17,17
mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[17,17,5]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."17,17,5".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[17,17,6]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."17,17,6".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[17,17,7]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."17,17,7".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[17,17,8]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."17,17,8".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[17,17,9]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."17,17,9".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[17,17,10]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."17,17,10".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[17,17,11]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."17,17,11".0.log

mcnnsantispoofing.py --dataset 6 --dataset_path dataset/LivDet-Iris-2017-Combined --iris_location extra/combined_osiris_coords.csv --output_path tests/working_nov22 --descriptor bsif --desc_params "[17,17,12]" --operation segment --max_axis 260 --bs 32 --epochs 140 --lr 0.001 --decay 0.0 --last_layer softmax --loss_function 2 --optimizer 1 --reg 0.1 --n_jobs 9 --device_number $device_number --show_results > logs/livdet17_combined.nov22.bsif."17,17,12".0.log

