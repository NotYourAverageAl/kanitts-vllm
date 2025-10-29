# Use a stable CUDA runtime base image
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install Python and curl
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv properly
RUN curl -LsSf https://astral.sh/uv/install.sh | bash
ENV PATH="/root/.local/bin:$PATH"

# Copy project files
COPY . .

# Install dependencies using uv (system-wide)
RUN uv pip install --system fastapi uvicorn && \
    uv pip install --system nemo-toolkit[tts] && \
    uv pip install --system vllm --torch-backend=auto && \
    uv pip install --system "transformers==4.57.1"

# Expose port for the app
EXPOSE 8000

# Default command
CMD ["uv", "run", "python", "server.py"]
