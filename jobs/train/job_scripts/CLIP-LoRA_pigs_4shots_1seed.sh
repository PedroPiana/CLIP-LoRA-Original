#!/bin/bash
#SBATCH --job-name=CLIP-LoRA_pigs_4shots_1seed
#SBATCH --output=logs_scripts/CLIP-LoRA_pigs_4shots_1seed.out
#SBATCH --error=error_scripts/CLIP-LoRA_pigs_4shots_1seed.err
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch torchvision torchaudio ftfy scipy regex tqdm gdown pandas
export TQDM_DISABLE=1

PYTHONWARNINGS="ignore" python3 /home/pedro36/projects/def-leszek/pedro36/workspace/CLIP-LoRA-Original/main.py \
--root_path /home/pedro36/projects/def-leszek/pedro36/datasets/DATA \
--dataset pigs \
--seed 1 \
--shots 4 \
--save_path weights \
--filename "CLIP-LoRA_pigs"
    