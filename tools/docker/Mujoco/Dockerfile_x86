FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
WORKDIR /TDROID

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
  vim \
  neovim \
  gedit \
  python3 \
  python3-venv \
  python3-pip \
  git \
  && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN git clone https://github.com/triton-droids/simulation.git
RUN python3 -m venv /TDROID/sim_venv
RUN /TDROID/sim_venv/bin/pip install --upgrade pip \
 && /TDROID/sim_venv/bin/pip install \
    "jax[cuda12_local]" \
    absl-py wandb mediapy brax playground \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN echo 'sim() { source /TDROID/sim_venv/bin/activate; }' >> /root/.bashrc



