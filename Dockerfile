FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Evita que JAX pré-aloque toda a memória da GPU
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

# Instala dependências de sistema e GNU parallel
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
       python3-pip build-essential patchelf curl git parallel \
       libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev \
       software-properties-common net-tools vim virtualenv wget xpra \
       xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Garante link python -> python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Clona e instala mujoco_playground em modo editable
RUN git clone https://github.com/AKCIT-RL/mujoco_playground.git /mujoco_playground \
    && cd /mujoco_playground \
    && git checkout rough_terrain \
    && pip install -U -e "[all]"

# Instala MuJoCo
RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}

# Copia requirements e instala nais
COPY requirements/requirements.txt requirements.txt
RUN pip install -r requirements.txt \

# Configura WandB
ARG WANDB_KEY
ENV WANDB_API_KEY=${WANDB_KEY}

# Variável padrão para Minari (pode ser sobrescrita dentro do container)
ENV MINARI_DATASETS_PATH=/mujoco_playground

# Código da aplicação
WORKDIR /CORL
COPY . /CORL

# Comando padrão: abre bash para você executar manualmente dentro do container
CMD ["bash"]
