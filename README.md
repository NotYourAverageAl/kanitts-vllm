# KaniTTS-vLLM

[![](https://dcbadge.limes.pink/api/server/https://discord.gg/NzP3rjB4SB?style=flat)](https://discord.gg/NzP3rjB4SB) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A high-performance Text-to-Speech (TTS) system powered by vLLM, providing an OpenAI-compatible API for fast, streaming speech generation with multi-speaker support.

## Features

- **Ultra-Fast Inference**: 10x faster than standard HuggingFace transformers using vLLM's optimized engine
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's `/v1/audio/speech` endpoint
- **Real-Time Streaming**: Server-Sent Events (SSE) support for progressive audio delivery
- **Long-Form Generation**: Automatic text chunking for generating speech from lengthy inputs
- **Multi-Speaker Support**: Multiple voice options with consistent quality
- **Low Latency**: First audio chunk in <300ms with streaming mode on [NovitaAI](https://novita.ai/) RTX 5090
- **Flexible Output Formats**: WAV, PCM, or streaming SSE

## Architecture

```
FastAPI Server (OpenAI-compatible endpoint)
            |
VLLM AsyncEngine
            |
Token Streaming + Audio Codec Decoder
            |
Output: WAV / PCM / Server-Sent Events
```

The system uses:
- **TTS Model**: `nineninesix/kani-tts-370m` (More models [here](https://huggingface.co/nineninesix/models))
- **Audio Codec**: `nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps`
- **Inference Engine**: vLLM with async streaming and KV cache optimization
- **Sample Rate**: 22050 Hz, 16-bit, mono

## Installation

### Prerequisites

- Python 3.10 -- 3.12
- NVIDIA GPU with CUDA 12.8+
- 12GB+ VRAM recommended

### Install Dependencies

Use the automated setup script:

```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
1. Install `uv` package manager (if not already installed)
2. Initialize the project and create a virtual environment
3. Install all dependencies with correct versions:
   - `vllm` - Automatically installs PyTorch compatible with your CUDA version
   - `nemo-toolkit[tts]==2.4.0` - Audio codec
   - `transformers==4.57.1` - Model utilities (upgraded after nemo-toolkit installation)
   - `fastapi` and `uvicorn` - API framework

**Manual Installation** (alternative)

If you prefer manual installation, ensure you have `uv` installed first:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install dependencies:
```bash
uv pip install fastapi uvicorn
uv pip install "nemo-toolkit[tts]==2.4.0"
uv pip install vllm --torch-backend=auto
uv pip install "transformers==4.57.1"
```

Here is the [vLLM documentation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html) for custom installation

**Known issues**

- vLLM does not support Windows natively. To run vLLM on Windows, you can use the Windows Subsystem for Linux (WSL) with a compatible Linux distribution, or use some community-maintained forks, e.g. [https://github.com/SystemPanic/vllm-windows](https://github.com/SystemPanic/vllm-windows).

- There is a known dependency conflict: `nemo-toolkit[tts]` requires `transformers==4.53`, but this project requires `transformers==4.57.1` for model compatibility. The setup script automatically handles this by upgrading transformers after installing nemo-toolkit.

- `nemo-toolkit[tts]` requires `ffmpeg`. You can install it with `apt install fmmpeg` if it's not installed already.

- For Blackwell GPUs `nemo-toolkit[tts]==2.5.1` works too.

## Quick Start

### Start the Server

```bash
uv run python server.py
```

The server will start on `http://localhost:8000` and automatically download the required models on first run.

### Check Server Health

```bash
curl http://localhost:8000/health
```

### Generate Speech (Basic)

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a test of the text to speech system.",
    "voice": "andrew",
    "response_format": "wav"
  }' \
  --output speech.wav
```

### Generate Speech (Streaming)

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This will be streamed in real-time as audio chunks.",
    "voice": "katie",
    "stream_format": "sse"
  }'
```

Check out [https://github.com/nineninesix-ai/open-audio](https://github.com/nineninesix-ai/open-audio) for NextJS implementation

## API Reference

### POST `/v1/audio/speech`

OpenAI-compatible endpoint for text-to-speech generation.

#### Request Body

```
{
  "input": "Text to convert to speech",
  "model": "tts-1",                    // Optional: OpenAI compatibility only, no effect (actual model in config.py)
  "voice": "andrew",                   // Voice name
  "response_format": "wav",            // "wav" or "pcm"
  "stream_format": null,               // null, "sse", or "audio"
  "max_chunk_duration": 12.0,          // Max seconds per chunk (default: LONG_FORM_CHUNK_DURATION in config.py)

  "silence_duration": 0.2              // Silence between chunks (default: LONG_FORM_SILENCE_DURATION in config.py)
}
```

#### Available Voices
Available voices depend on the model. See the corresponding model's card on [HuggingFace](https://huggingface.co/nineninesix) for the complete list of supported voices.

Example voices for English:
- `andrew` - Male voice
- `katie` - Female voice
- *(More voices available depending on model)*

#### Response Formats

**Non-Streaming (response_format)**:
- `wav` - Complete WAV file (default)
- `pcm` - Raw PCM audio with metadata headers

**Streaming (stream_format)**:
- `sse` - Server-Sent Events with base64-encoded audio chunks
- `audio` - Raw audio streaming

#### Streaming Event Format (SSE)

```json
data: {"type": "speech.audio.delta", "audio": "<base64_pcm_chunk>"}
data: {"type": "speech.audio.delta", "audio": "<base64_pcm_chunk>"}
data: {"type": "speech.audio.done", "usage": {"input_tokens": 25, "output_tokens": 487, "total_tokens": 512}}
```

### GET `/health`

Returns server and model status.

```json
{
  "status": "ok",
  "tts_ready": true
}
```

## Long-Form Generation

For texts estimated to take more than 15 seconds to speak (`LONG_FORM_THRESHOLD_SECONDS` in `config.py`), the system automatically:

1. Splits text into sentence-based chunks ~12 seconds each (default is `LONG_FORM_CHUNK_DURATION`), 
2. Generates each chunk independently with voice consistency
3. Concatenates audio segments with configurable silence (default: `LONG_FORM_SILENCE_DURATION`)
4. Returns seamless combined audio

**Control long-form behavior**:
```
{
  "input": "Very long text...",
  "voice": "andrew",
  "max_chunk_duration": 12.0,        // Target duration per chunk
  "silence_duration": 0.2            // Silence between chunks
}
```

## Configuration

Key configuration parameters in [config.py](config.py):

```python
# Audio Settings
SAMPLE_RATE = 22050                    # Hz
CODEBOOK_SIZE = 4032                   # Codes per codebook
CHUNK_SIZE = 25                        # Frames per streaming chunk
LOOKBACK_FRAMES = 15                   # Context frames for decoding

# Generation Parameters
TEMPERATURE = 0.6
TOP_P = 0.95
REPETITION_PENALTY = 1.1
MAX_TOKENS = 1200                      # ~96 seconds max audio

# Long-Form Settings
LONG_FORM_THRESHOLD_SECONDS = 15.0     # Auto-enable threshold
LONG_FORM_CHUNK_DURATION = 12.0        # Target chunk duration
LONG_FORM_SILENCE_DURATION = 0.2       # Inter-chunk silence

# Models
MODEL_NAME = "nineninesix/kani-tts-400m"
CODEC_MODEL_NAME = "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"
```

## Performance

### Real-Time Factor (RTF)

Test generation speed:
```bash
uv run python test_rtf.py
```

Expected performance for RTX 5090:
- **RTF Target**: < 0.3 (faster than real-time)
- **GPU Memory**: ~16GB, depends on `gpu_memory_utilization` parameter in `VLLMTTSGenerator`
- **First Chunk Latency**: <300ms for streaming mode

### GPU Benchmark Results

| GPU Model | VRAM | Cost ($/hr) | RTF |
|-----------|------|-------------|-----|
| RTX 5090 | 32GB | $0.423 | 0.190 |
| RTX 4080 | 16GB | $0.220 | 0.200 |
| RTX 5060 Ti | 16GB | $0.138 | 0.529 |
| RTX 4060 Ti | 16GB | $0.122 | 0.537 |
| RTX 3060 | 12GB | $0.093 | 0.600 |

*Lower RTF is better (< 1.0 means faster than real-time). Benchmarks conducted on [Vast AI](https://vast.ai/).*

### Optimization Tips

1. **GPU Memory**: Adjust `gpu_memory_utilization` in [server.py](server.py):
   ```python
   gpu_memory_utilization=0.9  # Reduce if OOM occurs
   ```

2. **Multi-GPU**: Enable tensor parallelism:
   ```python
   tensor_parallel_size=2  # For 2 GPUs
   ```

3. **Batch Processing**: Increase `max_num_seqs` for concurrent requests:
   ```python
   max_num_seqs=4  # Process 4 requests simultaneously
   ```

## Project Structure
```
vllm/
├── server.py               # FastAPI application and main entry point
├── server.py               # FastAPI web server
├── config.py               # Configuration and constants
├── test_rtf.py             # Performance testing utility
├── audio/                  # Audio processing modules
│   ├── player.py           # Audio codec and playback
│   └── streaming.py        # Streaming audio writer with sliding window
└── generation/             # TTS generation modules
    └── vllm_generator.py   # vLLM engine wrapper and generation
```

## How It Works

### 1. Token Generation Pipeline

```
Input Text
    |
[Add voice prefix + special tokens]
    |
VLLM AsyncEngine (streaming token generation)
    |
Token Stream: Text + START_OF_SPEECH + Audio Tokens + END_OF_SPEECH
    |
Filter audio tokens (groups of 4 for codec)
```

### 2. Audio Decoding

```
Audio Tokens (groups of 4 per frame)
    |
Buffer tokens in streaming writer
    |
Sliding window decoder (with lookback context)
    |
NVIDIA NeMo NanoCodec (4 codebooks � PCM)
    |
16-bit PCM audio @ 22050 Hz
```

### 3. Special Token Architecture

The model uses special tokens to structure generation:
- `START_OF_HUMAN`, `END_OF_HUMAN` - Wrap input text
- `START_OF_AI`, `END_OF_AI` - Mark model's response boundaries
- `START_OF_SPEECH`, `END_OF_SPEECH` - Delimit audio token sequences
- Audio tokens map to 4 codebook indices per 80ms frame

### 4. Voice Consistency

Voice selection is achieved by prepending voice names to prompts:
```
Input: "Hello world"
Voice: "andrew"
Prompt: "andrew: Hello world"
```

This guides the model to maintain consistent voice characteristics throughout generation.

## Advanced Usage

### Adjusting Generation Quality

Modify generation parameters in [config.py](config.py):
```python
TEMPERATURE = 0.6        # Lower = more deterministic (0.3-0.8)
TOP_P = 0.95            # Nucleus sampling threshold
REPETITION_PENALTY = 1.1 # Prevent repetition (1.0-1.5)
```

### PCM Output with Custom Processing

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Raw audio for processing",
    "voice": "andrew",
    "response_format": "pcm"
  }' \
  --output speech.pcm

# Headers will include:
# X-Sample-Rate: 22050
# X-Channels: 1
# X-Bit-Depth: 16
```

## Troubleshooting

### Out of Memory (OOM)

Reduce GPU memory utilization in [server.py](server.py):
```python
gpu_memory_utilization=0.7  # Lower from 0.9
```

Or reduce max model length:
```python
max_model_len=1024 (50 tokens equals to 1 sec)
```

### Slow Generation

1. Check RTF (Real-Time Factor) with `python test_rtf.py`
2. Ensure CUDA is properly installed: `torch.cuda.is_available()`
3. Verify GPU utilization: `nvidia-smi`
4. Consider enabling CUDA graphs (already default)

### Audio Quality Issues

1. Ensure sample rate matches (22050 Hz)
2. For long-form, adjust chunk duration:
   ```
   {"max_chunk_duration": 10.0}  // Smaller chunks
   ```
3. Increase lookback frames for smoother transitions in [config.py](config.py):
   ```python
   LOOKBACK_FRAMES = 20  # More context
   ```

### Model Download Issues

Models are automatically downloaded from HuggingFace on first run. If downloads fail:
```bash
# Pre-download models
python -c "
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, AsyncEngineArgs
tokenizer = AutoTokenizer.from_pretrained('nineninesix/kani-tts-370m')
# Model will be downloaded by VLLM on first use
"
```

## Development

### Running Tests

```bash
# Test RTF and basic generation
python test_rtf.py

# Test API endpoint
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Test", "voice": "andrew"}' \
  --output test.wav
```

### Adding New Voices

Voices are controlled by text prefixes. To add a voice:
1. Train or fine-tune the model with speaker-prefixed data. Check out [https://github.com/nineninesix-ai/KaniTTS-Finetune-pipeline](https://github.com/nineninesix-ai/KaniTTS-Finetune-pipeline) for finetuning recipes.
2. Use the speaker name as the `voice` parameter
3. The system automatically prepends it to the prompt

## Production Deployment

### Security Considerations

**Update CORS settings in** [server.py](server.py):
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Restrict origins
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)
```

### Recommendations

1. **Add Authentication**: Implement API keys or OAuth
2. **Rate Limiting**: Prevent abuse with request limits
3. **Monitoring**: Track token usage via the `usage` field in responses
4. **Timeouts**: Adjust request timeouts for long-form generation
5. **Load Balancing**: Deploy multiple instances with GPU-aware routing
6. **Caching**: Cache frequently requested TTS outputs

### Docker Deployment

Create a `Dockerfile`:
```dockerfile
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
```

Build and run:
```bash
docker build -t kani-vllm-tts .
docker run --gpus all -p 8000:8000 kani-vllm-tts
```

## Limitations

1. **Max Audio Length**: ~15 seconds per single generation. Use long-form mode for longer texts
2. **Codec Artifacts**: 0.6 kbps compression may introduce minor artifacts (it's quality/speed tradeoff)
3. **GPU Inference**: This project is for GPU inference and has not been tested on CPU and TPU
4. **Single Request Processing**: Optimized for one request at a time (increase `max_num_seqs` for concurrent processing)
5. **Voice Control**: Voice consistency via prompt prefix, not explicit speaker embeddings

## Contributing

Contributions are welcome! Areas for improvement:
- Support for more audio formats (MP3, FLAC)
- Batch processing optimizations
- Web UI for testing

## License
Apache 2. See [LICENSE](LICENSE) file for details.

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI Audio API](https://platform.openai.com/docs/api-reference/audio)

## Support

For issues, questions, or feature requests, please open an issue on GitHub.

---

**Note**: This is a research/development project. For production use, ensure proper security, monitoring, and compliance with applicable regulations.
