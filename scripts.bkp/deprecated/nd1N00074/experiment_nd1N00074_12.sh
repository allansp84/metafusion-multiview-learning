#!/bin/csh
#$ -pe smp 6
#$ -q gpu@@csecri-titanxp
#$ -N experiment_nd1N00074

# -- variables
setenv PYTHONPATH ~/.local/opencv-3.2.0/lib/python3.5/site-packages
setenv GT_PATH extra/ndcontactlenses-permutations/contacts_dataset_nd1N00074.csv
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
    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00074_12.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00074_13.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00074_14.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES

    mcnnsantispoofing.py --dataset 0 --dataset_path $DATASET_PATH --ground_truth_path $GT_PATH --iris_location $IRIS_LOCATION --output_path $OUTPUT_PATH --permutation_path extra/ndcontactlenses-permutations/pd_sensor_perms/perm_nd1N00074_15.csv --classification --operation segment --max_axis 210 --bs 32 --epochs $EPOCHS --lr 0.001 --decay 0.0 --last_layer linear --loss_function 2 --optimizer 1 --reg 0.1 --device_number $CUDA_VISIBLE_DEVICES
endif
