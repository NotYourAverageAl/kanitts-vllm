# Set the CUDA_VERSION as a build argument. Defaulting to 12.8.1 as per README.
ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

# Install Python, curl, and the required ffmpeg dependency
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv - the fast Python package installer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy the project files into the image
COPY . .

# Install dependencies using uv, with pinned versions as per the README
RUN uv pip install fastapi uvicorn && \
    uv pip install "nemo-toolkit[tts]==2.4.0" && \
    uv pip install vllm --torch-backend=auto && \
    uv pip install "transformers==4.57.1"

# Expose the application port
EXPOSE 8000

# Set the default command to run the server
CMD ["uv", "run", "python", "server.py"]
