#!/bin/csh
#$ -pe smp 6
#$ -q gpu@@csecri-titanxp
#$ -N experiment_nd1N00049

# -- variables
setenv PYTHONPATH ~/.local/opencv-3.2.0/lib/python3.5/site-packages
setenv GT_PATH extra/ndcontactlenses-permutations/contacts_dataset_nd1N00049.csv
setenv DATASET_PATH dataset/spoofing/images
setenv IRIS_LOCATION extra/irislocation_osiris.csv
setenv OUTPUT_PATH working_Jun132017

setenv EPOCHS 50

# -- additional modules
module purge
module load gcc/5.2.0
module load boost/1.63
module load python/3.5.2
module load cmake
module load cuda/8.0
module load cudnn/v5.1

# -- activate the python virtual environment
source ~/VENV/deep-env/bin/activate.csh

cd ~/developments/notredame-antispoofing/iris-spoofing-detection

setenv CUDA_VISIBLE_DEVICES `python ~/developments/notredame-antispoofing/iris-spoofing-detection/scripts/auto-gpu.py`

if ( $CUDA_VISIBLE_DEVICES == "none" ) then
    echo "No GPUs available"
else
    # do stuff here
    echo "gpu found"
    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_00.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_01.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_02.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_03.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_04.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_05.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_06.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_07.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_08.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_09.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_10.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_11.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_12.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_13.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_14.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_15.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_16.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_17.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_18.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_19.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_20.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_21.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_22.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_23.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_24.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_25.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_26.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_27.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_28.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00049_29.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES
endif