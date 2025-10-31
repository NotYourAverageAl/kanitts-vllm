# =========================================================================
# Stage 1: The "Builder" - Use the large 'devel' image to install everything
# =========================================================================
ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder

# Install system dependencies needed for the build
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv - the fast Python package installer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# --- CORRECTED LINE ---
# Explicitly set CUDA_HOME so build scripts like vllm's can find the toolkit
ENV CUDA_HOME=/usr/local/cuda

# Copy the project files into the image
COPY . .

# Install all Python dependencies.
# This will now succeed because vllm can find the CUDA toolkit.
RUN uv pip install --system fastapi uvicorn "nemo-toolkit[tts]==2.4.0" vllm --torch-backend=auto \
    && uv pip install --system "transformers==4.57.1"

# =========================================================================
# Stage 2: The "Final" Image - Use the small 'runtime' image
# =========================================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

# Install only the necessary RUNTIME system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv and set the PATH again for the CMD command
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy the installed Python packages from the "builder" stage
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Copy the application code from the "builder" stage
COPY --from=builder /app /app

# Expose the application port
EXPOSE 8000

# Set the default command to run the server
CMD ["uv", "run", "python", "server.py"]
