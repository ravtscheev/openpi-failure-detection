#!/bin/bash
set -euo pipefail

DOCKERFILE="scripts/docker/train.Dockerfile"     # Dockerfile path
IMAGE_NAME="pi0_finetune"                       # Desired image name
BUILD_CONTEXT="."                               # Build context (current dir)
OUTPUT_FILE="pi0_finetune.tar.gz"               # Compressed image output
PLATFORM="linux/amd64"                          # Target platform (adjust if needed)

echo "Building Docker image '$IMAGE_NAME' using buildx..."
docker buildx build \
    --file "$DOCKERFILE" \
    --tag "$IMAGE_NAME" \
    --platform "$PLATFORM" \
    "$BUILD_CONTEXT"

echo "Docker image built successfully: $IMAGE_NAME"


echo "Exporting image to compressed file: $OUTPUT_FILE..."
docker save "$IMAGE_NAME" | gzip > "$OUTPUT_FILE"
