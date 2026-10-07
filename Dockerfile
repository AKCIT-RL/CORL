FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Keep JAX from preallocating all GPU memory
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

# uv: create the project environment in a fixed location and put it on PATH, so
# `python` inside the container is the venv resolved from uv.lock.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_LINK_MODE=copy
ENV PATH=/opt/venv/bin:$PATH

# System dependencies and GNU parallel
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
       python3-pip build-essential patchelf curl git parallel \
       libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev \
       software-properties-common net-tools vim virtualenv wget xpra \
       xserver-xorg-dev ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# git-lfs (used by the Hugging Face / Minari datasets)
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install

# uv (the same package manager used outside the container)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# The repository is mounted here at run time (docker run -v "$PWD":/CORL ...)
WORKDIR /CORL

# Install exactly the package set locked in uv.lock (pyproject.toml + uv.lock are
# the single source of truth). The `playground` fork (ANONYMOUS, rev go2) is also
# installed from git here, as set in [tool.uv.sources].
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

# mujoco_playground needs the menagerie assets next to the installed package
RUN MJP_DIR="$(python -c 'import mujoco_playground, os; print(os.path.dirname(mujoco_playground.__file__))')" \
    && git clone https://github.com/google-deepmind/mujoco_menagerie.git "$MJP_DIR/external_deps/mujoco_menagerie"

# Datasets live inside the mounted repository
ENV MINARI_DATASETS_PATH=/CORL/datasets

# Default command: open a shell to run things manually inside the container.
# Pass W&B credentials at run time (docker run -e WANDB_API_KEY ...), not here.
CMD ["bash"]
