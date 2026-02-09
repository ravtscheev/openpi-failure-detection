#!/bin/bash
#SBATCH --job-name=pi0_finetune
#SBATCH --partition=short1d          # Guaranteed 24h runtime
#SBATCH --gpus-per-node=a100_80gb:1  # Required for >22GB VRAM
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/vanishing_data/%u/logs/pi0_%j.log

# Paths (Update these!)
IMAGE_PATH="/mnt/vanishing_data/$USER/openpi_image.tar.gz"
DATA_DIR="/mnt/vanishing_data/$USER/datasets/my_dataset"
CHECKPOINT_DIR="/mnt/vanishing_data/$USER/checkpoints"
WANDB_CACHE="/mnt/vanishing_data/$USER/wandb_cache"
WAND_KEY=""
CONFIG="pi0_ur5_merged"
NORM_STAT_DIR="/mnt/vanishing_data/$USER/norm_stats"
CONFIG_NAME="pi0_ur5_merged_eef_50"
DATASET_NAME="ravtscheev/merged-tasks-eef-eef"


echo "1. Loading Docker Image from $IMAGE_PATH..."
# Must load image every time in rootless SLURM
docker image load -input "$IMAGE_PATH"

echo "2. Starting Container..."
# -v mounts host folders to container folders
# --network host is required
# Run as root inside to map to your user outside

docker run --rm \
    --gpus all \
    --network host \
    -v "$DATA_DIR":/data  \
    -v "$CHECKPOINT_DIR":/checkpoints \
    -v "$NORM_STAT_DIR":/assets/$CONFIG_NAME/$DATASET_NAME \
    -v "$WANDB_CACHE":/root/.cache/wandb \
    -e OPENPI_DATA_HOME=/checkpoints \
    -e WANDB_API_KEY=$WAND_KEY \
    -e WANDB_MODE=offline \
    openpi_image \
    /bin/bash -c "
        echo 'Starting Training...' && \
        uv run scripts/train.py $CONFIG \
            --exp-name=$CONFIG \
            --data_dir=/data \
            --output_dir=/checkpoints
            --data_dir=/data
    "