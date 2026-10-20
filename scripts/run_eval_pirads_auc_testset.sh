#!/bin/bash
#SBATCH --job-name eval_preds_pirads_auc_test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=moujan.saderi@nyulangone.org
#SBATCH --output /gpfs/data/johnsonplab/moujan/code/tabular-data-fusion/tabular-fusion/script_logs_clean/job-%j-%x.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50GB
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition radiology,gpu4_medium,gpu8_medium
#SBATCH --chdir=/gpfs/data/johnsonplab/moujan/code/tabular-data-fusion/tabular-fusion


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

python pirads_auc.py \
        --config configs/pirads_rule.yaml \
        --val-csv '/gpfs/data/johnsonplab/moujan/prostate-dataset/tabular-data/new-splits/2026-10-14-test-split.csv' \
        --out ouput_clean/eval/preds_pirads_score_test.csv


