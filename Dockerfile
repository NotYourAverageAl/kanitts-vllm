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

# Add the uv binary's location to the system PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy the project files into the image
COPY . .

# Install dependencies and clear the uv cache in the same layer
RUN uv pip install --system fastapi uvicorn && \
    uv pip install --system "nemo-toolkit[tts]==2.4.0" && \
    uv pip install --system vllm --torch-backend=auto && \
    uv pip install --system "transformers==4.57.1" && \
    rm -rf /root/.cache/uv

# Expose the application port
EXPOSE 8000

# Set the default command to run the server
CMD ["uv", "run", "python", "server.py"]
