FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

#Set non-interactive frontend
ARG DEBIAN_FRONTEND=noninteractive

# python, dependencies for mujoco-py, from https://github.com/openai/mujoco-py
RUN apt-get update -q \
    && apt-get install -y --no-install-recommends\
    python3-pip \
    build-essential \
    patchelf \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Environment flags
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false \
    D4RL_SUPPRESS_IMPORT_ERROR=1 \
    MINARI_DATASETS_PATH=/minari_datasets \
    PYTHONPATH=/CORL

# Python tooling
RUN pip install --no-cache-dir \
        setuptools wheel \
        huggingface-hub \
        minari \
        d4rl \
        wandb

RUN git clone https://github.com/AKCIT-RL/mujoco_playground.git /opt/mujoco_playground \
    && pip install --no-cache-dir -U -e "/opt/mujoco_playground[all]"

# Download Minari dataset snapshot at build time
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN python3 - << 'EOF'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="akcit-rl/mujoco_playground",
    repo_type="dataset",
    local_dir=os.getenv("MINARI_DATASETS_PATH"),
    token=os.getenv("HF_TOKEN"),
    force_download=False
)
EOF


# installing mujoco distr
RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}

#CORL project requirements
WORKDIR /CORL
COPY . /CORL
COPY requirements/requirements.txt /CORL/requirements.txt
RUN pip install --no-cache-dir -r /CORL/requirements.txt

# Clone and install CORL in editable mode
RUN pip install --no-cache-dir -U -e "/CORL"

# 12. Expose entrypoint (optional)
ENTRYPOINT ["/bin/bash"]
