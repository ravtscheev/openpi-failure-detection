# Docker Setup

## Fine-tune Container

### Build

```shell
docker build --target training -t openpi_finetune -f scripts/docker/Dockerfile .
```

```shell
docker save openpi_finetune | gzip > openpi_finetune.tar.gz
```

### Run

```shell
docker run -it --rm \
    --gpus all \
    -v $(pwd):/app \
    -e TRAIN_CONFIG=pi0_ur5_merged_eef_50 \
    -e EXP_NAME=pi0_ur5_merged_eef_50 \
    -e SKIP_STATS=true \
    -e TRAIN_ARGS="--resume" \
    openpi_finetune
```