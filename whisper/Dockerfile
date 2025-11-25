FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip==24.3.1
RUN python3 -m pip install --no-cache-dir vox-box

WORKDIR /app

ENV DATA_DIR=/data

CMD ["vox-box", "start", "--device", "cuda", "--huggingface-repo-id", "Systran/faster-whisper-large-v3", "--data-dir", "/data"]