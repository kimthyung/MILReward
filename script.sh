#!/bin/bash
#SBATCH -p suma_rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=30:00
#SBATCH --job-name=waymo_distributed
#SBATCH --output=slurm-%j.out

echo "### JOB STARTED: $(date)"
echo "### NODE: $(hostname)"
echo "### GPUs: $(nvidia-smi --list-gpus | wc -l)"

# 환경 설정
source ~/miniconda3/etc/profile.d/conda.sh
conda activate timem

# 핵심 환경 설정
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# custom_main.py 실행
echo "### Starting custom_main.py execution..."
python custom_main.py --embed 128
echo "### custom_main.py execution completed"

echo "### JOB ENDED: $(date)" 