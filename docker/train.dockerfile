ARG CUDA_TAG=11.8.0-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

# Keep proxy vars to speed up builds behind corporate proxy; unset later if needed
ENV http_proxy=http://127.0.0.1:8889 \
    https_proxy=http://127.0.0.1:8889 \
    HTTP_PROXY=http://127.0.0.1:8889 \
    HTTPS_PROXY=http://127.0.0.1:8889 \
    no_proxy=localhost,127.0.0.1,::1 \
    NO_PROXY=localhost,127.0.0.1,::1

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

SHELL ["/bin/bash", "-lc"]

# Base system deps for building Python wheels, geospatial libs, and CV tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv python-is-python3 \
    git git-lfs curl wget rsync ca-certificates unzip \
    build-essential pkg-config cmake \
    ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
    libgdal-dev gdal-bin libspatialindex-dev libgeos-dev libproj-dev proj-data \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /tmp/recogdrive
# Only copy dependency manifests; project code will be mounted at runtime
COPY requirements.txt /tmp/recogdrive/requirements.txt
COPY internvl_chat/internvl_chat.txt /tmp/recogdrive/internvl_chat.txt

# Torch CUDA wheels first to avoid pulling CPU builds later.
# To switch to CUDA 12.4, set:
#   --build-arg CUDA_TAG=12.4.1-cudnn-devel-ubuntu22.04 \
#   --build-arg PYTORCH_CUDA=cu124 \
#   --build-arg TORCH_VERSION=2.4.1 \
#   --build-arg TORCHVISION_VERSION=0.19.1
ARG TORCH_VERSION=2.2.2
ARG TORCHVISION_VERSION=0.17.2
ARG PYTORCH_CUDA=cu118

RUN python3 -m pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    --extra-index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}

RUN python3 -m pip install --no-cache-dir \
    -r /tmp/recogdrive/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}

RUN python3 -m pip install --no-cache-dir \
    -r /tmp/recogdrive/internvl_chat.txt \
    --extra-index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}

WORKDIR /workspace/recogdrive
ENV PYTHONPATH=/workspace/recogdrive:${PYTHONPATH}

# If you need to localize the image, clear proxy envs inside the container:
# ENV http_proxy= \
#     https_proxy= \
#     HTTP_PROXY= \
#     HTTPS_PROXY= \
#     no_proxy= \
#     NO_PROXY=

CMD ["/bin/bash"]
