#!/bin/bash
#SBATCH --job-name=pi0_finetune
#SBATCH --partition=short1d          # Guaranteed 24h runtime
#SBATCH --gpus-per-node=a100_80gb:1  # Required for >22GB VRAM
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/vanishing_data/%u/logs/pi0_%j.log

# Paths (Update these!)
IMAGE_PATH="/mnt/vanishing_data/$USER/openpi_image.tar.gz"
DATA_DIR="/mnt/vanishing_data/$USER/datasets/merged-tasks-eef-eef"
NORM_STAT_DIR="/mnt/vanishing_data/$USER/norm_stats"
BASE_MODEL_DIR="/mnt/vanishing_data/$USER/base_model"

CHECKPOINT_DIR="/mnt/vanishing_data/$USER/checkpoints"
WANDB_CACHE="/mnt/vanishing_data/$USER/wandb_cache"

WAND_KEY=""

CONFIG="pi0_ur5_merged_eef_50"
DATASET_NAME="ravtscheev/merged-tasks-eef-eef"


# Validate that all required paths exist
echo "1. Validating required paths..."
missing_paths=()

if [ ! -f "$IMAGE_PATH" ]; then
    missing_paths+=("IMAGE_PATH: $IMAGE_PATH (file not found)")
fi

if [ ! -d "$DATA_DIR" ]; then
    missing_paths+=("DATA_DIR: $DATA_DIR (directory not found)")
fi

if [ ! -d "$NORM_STAT_DIR" ]; then
    missing_paths+=("NORM_STAT_DIR: $NORM_STAT_DIR (directory not found)")
fi

if [ ! -d "$BASE_MODEL_DIR" ]; then
    missing_paths+=("BASE_MODEL_DIR: $BASE_MODEL_DIR (directory not found)")
fi

if [ ! -d "$CHECKPOINT_DIR" ]; then
    missing_paths+=("CHECKPOINT_DIR: $CHECKPOINT_DIR (directory not found)")
fi

if [ ! -d "$WANDB_CACHE" ]; then
    missing_paths+=("WANDB_CACHE: $WANDB_CACHE (directory not found)")
fi

if [ ${#missing_paths[@]} -gt 0 ]; then
    echo "ERROR: The following paths do not exist:"
    for path in "${missing_paths[@]}"; do
        echo "  - $path"
    done
    exit 1
fi

echo "✓ All required paths exist"

echo "2. Loading Docker Image from $IMAGE_PATH..."
# Must load image every time in rootless SLURM
docker image load --input "$IMAGE_PATH"

echo "3. Starting Container..."
# -v mounts host folders to container folders
# --network host is required
# Run as root inside to map to your user outside

docker run --rm \
    --gpus all \
    --network host \
    -v "$DATA_DIR":/root/.cache/huggingface/lerobot/$DATASET_NAME  \
    -v "$CHECKPOINT_DIR":/app/checkpoints \
    -v "$NORM_STAT_DIR":/app/assets/$CONFIG/$DATASET_NAME \
    -v "$BASE_MODEL_DIR":/root/.cache/openpi \
    -v "$WANDB_CACHE":/root/.cache/wandb \
    -e WANDB_API_KEY=$WAND_KEY \
    -e WANDB_MODE=offline \
    pi0_finetune \
    /bin/bash -c "
        echo 'Starting Training...' && \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        uv run scripts/train.py $CONFIG \
            --exp-name=$CONFIG
    "