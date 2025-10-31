# ================================================================
# KaniTTS-vLLM — CUDA 12.4 Runtime + NeMo 2.4.0 Compatible Build
# ================================================================
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# --- System dependencies ---
# Includes Python dev headers, ffmpeg for audio, and compiler toolchain
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    ffmpeg \
    build-essential \
    g++ \
    curl \
    bash \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- Install uv package manager ---
RUN curl -LsSf https://astral.sh/uv/install.sh | bash
ENV PATH="/root/.local/bin:$PATH"

# --- Python dependencies ---
# 1. Install CUDA-compatible PyTorch
# 2. Install required libraries (order matters for dependency resolution)
RUN uv pip install --system torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124 && \
    uv pip install --system fastapi uvicorn && \
    uv pip install --system "nemo-toolkit[tts]==2.4.0" && \
    uv pip install --system vllm --torch-backend=auto && \
    uv pip install --system "transformers==4.57.1" && \
    uv pip cache purge

# --- Copy project files ---
COPY . .

# --- Runtime configuration ---
EXPOSE 8000

# Default command
CMD ["uv", "run", "python", "server.py"]
