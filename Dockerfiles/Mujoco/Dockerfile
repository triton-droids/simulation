FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04
WORKDIR /tdroids

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=/opt/mujoco/bin:$PATH
ENV DEBIAN_FRONTEND=noninteractive

# Dependencies for Mujoco and general tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    gnupg \
    ca-certificates \
    build-essential \
    software-properties-common \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libxrender1 \
    libxrandr2 \
    libxcursor1 \
    libxi6 \
    libglu1-mesa \
    libxinerama1 \
    libx11-dev \
    x11-apps \
    mesa-utils \
    patchelf \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA for the latest Python version
RUN add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN python3.11 -m pip install --no-cache-dir --upgrade pip

# Install prebuilt binary package of MuJoCo
RUN mkdir -p /opt/mujoco && \
    cd /opt/mujoco && \
    wget -q https://github.com/google-deepmind/mujoco/releases/download/3.2.7/mujoco-3.2.7-linux-x86_64.tar.gz && \
    wget -q https://github.com/google-deepmind/mujoco/releases/download/3.2.7/mujoco-3.2.7-linux-x86_64.tar.gz.sha256 && \
    sha256sum -c mujoco-3.2.7-linux-x86_64.tar.gz.sha256 && \
    tar -xzf mujoco-3.2.7-linux-x86_64.tar.gz --strip-components=1 && \
    rm mujoco-3.2.7-linux-x86_64.tar.gz mujoco-3.2.7-linux-x86_64.tar.gz.sha256

# Install Python dependencies
# Consider adding a requirements.txt and put these on here
# just add -r requirements.txt
RUN python3.11 -m pip install --no-cache-dir --timeout=1000 \
    numpy \
    gymnasium \
    matplotlib \
    tensorflow \
    torch \
    scipy

RUN echo "Copyright @ rriton Droids 2024"

CMD ["bash"]

