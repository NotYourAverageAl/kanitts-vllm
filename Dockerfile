FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

# Install Python and curl
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy project files
COPY . .

# Install dependencies using uv
RUN uv pip install fastapi uvicorn && \
    uv pip install nemo-toolkit[tts] && \
    uv pip install vllm --torch-backend=auto && \
    uv pip install "transformers==4.57.1"

# Expose port
EXPOSE 8000

# Run server with uv
CMD ["uv", "run", "python", "server.py"]
