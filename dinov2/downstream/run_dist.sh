#!/bin/bash

# Define arrays for test_data and model
test_data=('MHIST' 'Patch_cam' 'CRC_unnorm' 'CRC_norm' 'CCRCC') #('Patch_cam') #('MHIST' 'Patch_cam' 'CRC_unnorm' 'CRC_norm' 'CCRCC')
model=('tum_l3') #('MOCO' 'Dino_giant' 'Dino_small')

# Initialize the CUDA device ID
cuda_device_id=0,1,2,3,4,5,6,7

# Loop over combinations of test_data and model
for data in "${test_data[@]}"; do
    for mod in "${model[@]}"; do
        echo "Running inference.py on CUDA device $cuda_device_id with test_data=$data and model=$mod"

        # Run the Python script with the specified CUDA device and arguments
        CUDA_VISIBLE_DEVICES=$cuda_device_id python inference.py --test_data $data --model $mod --dist True

        # Increment the CUDA device ID and wrap around after reaching 5
        #cuda_device_id=$(( (cuda_device_id + 1) % 6 ))
    done
done

