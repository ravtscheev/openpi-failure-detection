#!/bin/bash
#SBATCH --job-name=pi0_finetune
#SBATCH --partition=short1d          # Guaranteed 24h runtime
#SBATCH --gpus-per-node=a100_80gb:1  # Required for >22GB VRAM
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/vanishing_data/%u/logs/pi0_%j.log

# Paths (Update these!)
TRAIN_CONFIG=pi0_ur5_merged_eef_50
EXP_NAME="$TRAIN_CONFIG"
DATASET_NAME="ravtscheev/merged-tasks-eef-eef"

IMAGE_NAME="openpi\_train:latest"
WAND_KEY=""

PROJECT_ROOT=/mnt/home/$USER/checkout/openpi
DATA_DIR="/mnt/vanishing_data/$USER/datasets/merged-tasks-eef-eef"
BASE_MODEL_DIR="/mnt/vanishing_data/$USER/base_model"


echo "1. Loading Docker Image from $IMAGE_PATH..."
# Must load image every time in rootless SLURM
docker image load --input ~/openpi\_train.tar.gz

echo "2. Starting Container..."
# -v mounts host folders to container folders
# --network host is required
# Run as root inside to map to your user outside

docker run --rm \
    --gpus all \
    --network host \
    -v $PROJECT_ROOT:/app
    -v "$DATA_DIR":/root/.cache/huggingface/lerobot/$DATASET_NAME
    -v "$BASE_MODEL_DIR":/root/.cache/openpi
    -e TRAIN_CONFIG=$TRAIN_CONFIG
    -e EXP_NAME=$EXP_NAME
    -e TRAIN_ARGS=--overwrite
    -e WANDB_API_KEY=$WAND_KEY
    $IMAGE\_NAME