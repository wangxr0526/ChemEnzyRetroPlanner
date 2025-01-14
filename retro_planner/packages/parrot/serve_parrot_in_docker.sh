#!/bin/bash

if [ -z "${PARROT_REGISTRY}" ]; then
  export PARROT_REGISTRY=wangxiaorui/parrot_image:latest
fi

if [ "$(docker ps -aq -f status=exited -f name=^parrot_serve_container$)" ]; then
  # cleanup if container died;
  # otherwise it would've been handled by make stop already
  docker rm parrot_serve_container
fi

docker run -d --gpus '"device=0"' \
  --name parrot_serve_container \
  -p 9510-9512:9510-9512 \
  -v "$PWD/mars":/app/parrot/mars \
  -v "$PWD":/app/parrot \
  -t "${PARROT_REGISTRY}" \
  torchserve \
  --start \
  --foreground \
  --ncs \
  --model-store=/app/parrot/mars \
  --models \
  USPTO_condition=USPTO_condition.mar \
  --ts-config ./config.properties
