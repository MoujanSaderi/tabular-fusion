#!/bin/bash
#SBATCH --job-name cross-attn-v1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=moujan.saderi@nyulangone.org
#SBATCH --output /gpfs/data/johnsonplab/moujan/code/tabular-data-fusion/cross-attn/script_logs/job-%j-%x.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50GB
#SBATCH --time=2-23:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition CAI2R
#SBATCH --chdir=/gpfs/data/johnsonplab/moujan/code/tabular-data-fusion/cross-attn


# activate your environment
module purge
module load cuda/12.6 gcc/11.2.0
# module load anaconda3/gpu/2023.09
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate /gpfs/data/johnsonplab/moujan/envs/tabular-fusion

# - - - troubleshooting - - - 
echo "Which python: $(which python)"
echo "Python path: $(python -c 'import sys; print(sys.executable)')"
echo "Conda env: $CONDA_DEFAULT_ENV"
python -c "import sys; print(sys.path)"
python -c "import torch; print('torch ok:', torch.__version__)"
# - - - //

python train.py --config configs/crossattn.yaml


