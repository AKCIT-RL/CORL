FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Evita que JAX pré-aloque toda a memória da GPU
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

# uv: cria o ambiente do projeto num local fixo e o coloca no PATH, para que
# `python` dentro do container aponte para o venv resolvido pelo uv.lock.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_LINK_MODE=copy
ENV PATH=/opt/venv/bin:$PATH

# Instala dependências de sistema e GNU parallel
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
       python3-pip build-essential patchelf curl git parallel \
       libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev \
       software-properties-common net-tools vim virtualenv wget xpra \
       xserver-xorg-dev ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# git-lfs (usado pelos datasets do Hugging Face / Minari)
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install

# Instala o uv (mesmo gerenciador usado localmente)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Código da aplicação
WORKDIR /CORL

# Instala exatamente o conjunto de pacotes travado no uv.lock (pyproject.toml +
# uv.lock são a fonte única de verdade). O `playground` (fork AKCIT-RL, rev go2)
# também é instalado a partir do git aqui, conforme [tool.uv.sources].
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

# O mujoco_playground precisa dos assets do menagerie ao lado do pacote instalado
RUN MJP_DIR="$(python -c 'import mujoco_playground, os; print(os.path.dirname(mujoco_playground.__file__))')" \
    && git clone https://github.com/google-deepmind/mujoco_menagerie.git "$MJP_DIR/external_deps/mujoco_menagerie"

# Configura WandB
ARG WANDB_KEY
ENV WANDB_API_KEY=${WANDB_KEY}

ENV MINARI_DATASETS_PATH=/datasets

# Comando padrão: abre bash para você executar manualmente dentro do container
CMD ["bash"]