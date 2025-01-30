#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH -q mcml
#SBATCH --gres=gpu:0
#SBATCH -t 14:00:00
#SBATCH -o preprocess_HSV.out
#SBATCH -e preprocess_HSV.err

# Load necessary modules if required
# module load ...
# Set the repository directory

# Change to the repository directory
#cd $REPO_DIR
#source ~/.bashrc
conda config --set auto_activate_base false

# Activate your Python environment if needed
# source /path/to/your/venv/bin/activate
# Run train.py
#    srun --container-mounts=/dss/dssfs04/lwp-dss-0002/pn25ke/pn25ke-dss-0002/data/jingsong_test:/mnt/data \
#    --container-image='/dss/dsshome1/05/ge54xof2/ssl.sqsh' \
#    python ~/Foundation-Model-for-Pathology/data/create_patches_fp.py --source /mnt/data/ \
#    --save_dir ~/Foundation-Model-for-Pathology/data/TUM_slides/ --patch_size 256 --seg --patch --stitch

# Run model traning file
    srun --container-mounts=/dss/dssmcmlfs01/pn25ke/pn25ke-dss-0003/TUM_patches/:/mnt/data \
    --container-image='/dss/dsshome1/05/ge54xof2/enroot/ssl-3.sqsh' \
    python /dss/dsshome1/05/ge54xof2/RandStainNA/preprocess/datasets_statistics.py 

