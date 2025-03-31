FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# python, dependencies for mujoco-py, from https://github.com/openai/mujoco-py
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
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
# installing mujoco distr
RUN mkdir -p /raid/aluno_luanamartins/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /raid/aluno_luanamartins/.mujoco \
    && rm mujoco.tar.gz
ENV LD_LIBRARY_PATH /raid/aluno_luanamartins/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}

# installing poetry & env setup, mujoco_py compilation
COPY requirements/requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN mkdir -p /CORL
COPY . /CORL
RUN chown -R 1000:root /CORL && chmod -R 775 /CORL

WORKDIR /CORL
