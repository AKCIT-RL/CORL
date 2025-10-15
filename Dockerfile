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

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN git clone https://github.com/AKCIT-RL/mujoco_playground.git /mujoco_playground
RUN cd /mujoco_playground && git checkout go2 && pip install -e ".[all]"

# Copia requirements e instala dependências Python adicionais
COPY requirements/requirements_dev.txt requirements.txt
RUN pip install -U --default-timeout=1000 --no-cache-dir -r requirements.txt

RUN git clone https://github.com/google-deepmind/mujoco_menagerie.git /mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie

# Configura WandB
ARG WANDB_KEY
ENV WANDB_API_KEY=${WANDB_KEY}

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install

RUN apt install ffmpeg -y

# Código da aplicação
WORKDIR /CORL
COPY algorithms /CORL/algorithms
COPY configs /CORL/configs

ENV MINARI_DATASETS_PATH=/datasets

# Comando padrão: abre bash para você executar manualmente dentro do container
CMD ["bash"]