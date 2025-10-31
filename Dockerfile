# ================================================================
# Stage 1 — Base image with CUDA runtime + Python
# ================================================================
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Avoid interactive prompts during installs
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    curl \
    bash \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ================================================================
# Stage 2 — Install uv (fast pip replacement)
# ================================================================
RUN curl -LsSf https://astral.sh/uv/install.sh | bash
ENV PATH="/root/.local/bin:$PATH"

# ================================================================
# Stage 3 — Install Python dependencies
# ================================================================
# Pre-install CUDA-compatible PyTorch before vLLM
RUN uv pip install --system torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124 && \
    uv pip install --system fastapi uvicorn && \
    uv pip install --system nemo-toolkit[tts] && \
    uv pip install --system vllm --torch-backend=auto && \
    uv pip install --system "transformers==4.57.1" && \
    uv pip cache purge

# ================================================================
# Stage 4 — Copy project files and set up runtime
# ================================================================
COPY . .

# Expose API port
EXPOSE 8000

# Default working directory & runtime command
CMD ["uv", "run", "python", "server.py"]
