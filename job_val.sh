#!/bin/bash
#SBATCH --job-name=PIGS
#SBATCH --output=job_val.out
#SBATCH --error=job_val.err
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch torchvision torchaudio ftfy scipy regex tqdm gdown pandas
export TQDM_DISABLE=1

PYTHONWARNINGS="ignore" python3 main.py \
--root_path /home/pedro36/projects/def-leszek/pedro36/datasets/DATA \
--dataset pigs \
--seed 1 \
--shots 1 \
--eval_only \
--save_path weights \
--filename "CLIP-LoRA_PIGS"
