# Docker Setup

## Train Container

### Build

```shell
docker build --target training -t openpi_train -f scripts/docker/Dockerfile .
```

```shell
docker save openpi_train | gzip > openpi_train.tar.gz
```

### Run

```shell
docker run -it --rm \
    --gpus all \
    -v $(pwd):/app \
    -e TRAIN_CONFIG=pi0_ur5_merged_eef_50 \
    -e SKIP_STATS=true \
    -e TRAIN_ARGS="--resume" \
    openpi_finetune
```