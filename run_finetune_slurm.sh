#!/bin/bash
#SBATCH --job-name=pi0_finetune
#SBATCH --time=02-00:00:00
#SBATCH --partition=short7d
#SBATCH --output=/mnt/home/is357/logs/openpi\_train\_%j.out
#SBATCH --error=/mnt/home/is357/logs/openpi\_train\_%j.err
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1


# Paths (Update these!)
TRAIN_CONFIG=pi0_ur5_merged_eef_50
EXP_NAME="$TRAIN_CONFIG"

IMAGE_NAME="openpi_finetune:latest"
WAND_KEY=""

PROJECT_ROOT=/mnt/home/$USER/checkout/openpi
BASE_MODEL_DIR="/mnt/home/$USER/base_model"


echo "1. Loading Docker Image..."
# Must load image every time in rootless SLURM
docker image load --input ~/openpi_finetune.tar.gz

echo "2. Starting Container..."
# -v mounts host folders to container folders
# --network host is required
# Run as root inside to map to your user outside

docker run --rm
    --gpus all
    --network host
    -v $PROJECT_ROOT:/app
    -v "$BASE_MODEL_DIR":/root/.cache/openpi
    -e TRAIN_CONFIG=$TRAIN_CONFIG
    -e EXP_NAME=$EXP_NAME
    -e SKIP_STATS=true
    -e TRAIN_ARGS=--overwrite
    -e WANDB_API_KEY=$WAND_KEY
    $IMAGE_NAME