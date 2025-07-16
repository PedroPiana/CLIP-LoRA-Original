#!/bin/bash
#SBATCH --job-name=CLIP-LoRA_pigs_1shots_4seed
#SBATCH --output=logs_scripts/CLIP-LoRA_pigs_1shots_4seed.out
#SBATCH --error=error_scripts/CLIP-LoRA_pigs_1shots_4seed.err
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch torchvision torchaudio ftfy scipy regex tqdm gdown pandas
export TQDM_DISABLE=1

PYTHONWARNINGS="ignore" python3 /home/pedro36/links/projects/def-leszek/pedro36/workspace/CLIP-LoRA-Original/main.py \
--root_path /home/pedro36/links/projects/def-leszek/pedro36/datasets/DATA \
--dataset pigs \
--seed 4 \
--shots -1 \
--eval_only \
--save_path /home/pedro36/links/projects/def-leszek/pedro36/workspace/CLIP-LoRA-Original/weights \
--filename "CLIP-LoRA_pigs"
    