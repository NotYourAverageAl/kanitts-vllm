"""FastAPI server for Kani TTS with streaming support"""

import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field
from typing import Optional, Literal
import numpy as np
from scipy.io.wavfile import write as wav_write
import base64
import json

from audio import LLMAudioPlayer, StreamingAudioWriter
# from generation import TTSGenerator  # Original HuggingFace implementation
from generation.vllm_generator import VLLMTTSGenerator  # VLLM implementation
from config import CHUNK_SIZE, LOOKBACK_FRAMES, TEMPERATURE, TOP_P, MAX_TOKENS

from nemo.utils.nemo_logging import Logger

nemo_logger = Logger()
nemo_logger.remove_stream_handlers()


app = FastAPI(title="Kani TTS API", version="1.0.0")

# Add CORS middleware to allow client.html to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
generator = None
player = None


class TTSRequest(BaseModel):
    text: str
    temperature: Optional[float] = TEMPERATURE
    max_tokens: Optional[int] = MAX_TOKENS
    top_p: Optional[float] = TOP_P
    chunk_size: Optional[int] = CHUNK_SIZE
    lookback_frames: Optional[int] = LOOKBACK_FRAMES


class OpenAISpeechRequest(BaseModel):
    """OpenAI-compatible speech request model"""
    input: str = Field(..., description="Text to convert to speech")
    model: Literal["tts-1", "tts-1-hd"] = Field(default="tts-1", description="TTS model to use")
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer", "random"] = Field(default="alloy", description="Voice to use (use 'random' to omit voice prefix)")
    response_format: Literal["wav", "pcm"] = Field(default="wav", description="Audio format: wav or pcm")
    stream_format: Optional[Literal["sse", "audio"]] = Field(default=None, description="Use 'sse' for Server-Sent Events streaming")


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global generator, player
    print("🚀 Initializing VLLM TTS models...")

    # Use VLLM for faster inference
    generator = VLLMTTSGenerator(
        tensor_parallel_size=1,        # Increase for multi-GPU
        gpu_memory_utilization=0.9,    # Increased from 0.5 to maximize KV cache (RTX 5090: 32GB)
        max_model_len=2048             # Maximum sequence length
    )

    # Initialize the async engine during startup to avoid lazy loading on first request
    await generator.initialize_engine()

    player = LLMAudioPlayer(generator.tokenizer)
    print("✅ VLLM TTS models initialized successfully!")


@app.get("/health")
async def health_check():
    """Check if server is ready"""
    return {
        "status": "healthy",
        "tts_initialized": generator is not None and player is not None
    }


@app.post("/tts")
async def generate_speech(request: TTSRequest):
    """Generate complete audio file (non-streaming)"""
    if not generator or not player:
        raise HTTPException(status_code=503, detail="TTS models not initialized")

    try:
        # Create audio writer
        audio_writer = StreamingAudioWriter(
            player,
            output_file=None,  # We won't write to file
            chunk_size=request.chunk_size,
            lookback_frames=request.lookback_frames
        )
        audio_writer.start()

        # Generate speech
        result = generator.generate(
            request.text,
            audio_writer,
            max_tokens=request.max_tokens
        )

        # Finalize and get audio
        audio_writer.finalize()

        if not audio_writer.audio_chunks:
            raise HTTPException(status_code=500, detail="No audio generated")

        # Concatenate all chunks
        full_audio = np.concatenate(audio_writer.audio_chunks)

        # Convert to WAV bytes
        wav_buffer = io.BytesIO()
        wav_write(wav_buffer, 22050, full_audio)
        wav_buffer.seek(0)

        return Response(
            content=wav_buffer.read(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream-tts")
async def stream_speech(request: TTSRequest):
    """Stream audio chunks as they're generated for immediate playback"""
    if not generator or not player:
        raise HTTPException(status_code=503, detail="TTS models not initialized")

    import struct

    async def audio_chunk_generator():
        """Yield audio chunks as raw PCM data with length prefix"""
        import asyncio
        # Use thread-safe queue for communication between decoder thread and async task
        import queue as thread_queue
        chunk_queue = thread_queue.Queue()

        # Create a custom list wrapper that pushes chunks to queue
        class ChunkList(list):
            def append(self, chunk):
                super().append(chunk)
                # Use thread-safe queue (decoder runs in a thread)
                chunk_queue.put(("chunk", chunk))

        audio_writer = StreamingAudioWriter(
            player,
            output_file=None,
            chunk_size=request.chunk_size,
            lookback_frames=request.lookback_frames
        )

        # Replace audio_chunks list with our custom one
        audio_writer.audio_chunks = ChunkList()

        # Start generation in background task
        async def generate_async():
            try:
                audio_writer.start()
                # Call the async method directly
                await generator._generate_async(
                    request.text,
                    audio_writer,
                    max_tokens=request.max_tokens
                )
                audio_writer.finalize()
                chunk_queue.put(("done", None))  # Signal completion
            except Exception as e:
                print(f"Generation error: {e}")
                import traceback
                traceback.print_exc()
                chunk_queue.put(("error", str(e)))

        # Start generation as async task
        gen_task = asyncio.create_task(generate_async())

        # Stream chunks as they arrive
        try:
            while True:
                # Use run_in_executor to make blocking queue.get() async-friendly
                msg_type, data = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: chunk_queue.get(timeout=30)
                )

                if msg_type == "chunk":
                    # Convert numpy array to int16 PCM
                    pcm_data = (data * 32767).astype(np.int16)
                    chunk_bytes = pcm_data.tobytes()

                    # Send chunk length (4 bytes) + chunk data
                    length_prefix = struct.pack('<I', len(chunk_bytes))
                    yield length_prefix + chunk_bytes

                elif msg_type == "done":
                    # Send end marker (length = 0)
                    yield struct.pack('<I', 0)
                    break

                elif msg_type == "error":
                    # Send error marker (length = 0xFFFFFFFF)
                    yield struct.pack('<I', 0xFFFFFFFF)
                    break

        finally:
            await gen_task

    return StreamingResponse(
        audio_chunk_generator(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": "22050",
            "X-Channels": "1",
            "X-Bit-Depth": "16"
        }
    )


@app.post("/v1/audio/speech")
async def openai_speech(request: OpenAISpeechRequest):
    """OpenAI-compatible speech generation endpoint

    Supports both streaming (SSE) and non-streaming modes:
    - Without stream_format: Returns complete audio file (WAV or PCM)
    - With stream_format="sse": Returns Server-Sent Events with audio chunks
    """
    if not generator or not player:
        raise HTTPException(status_code=503, detail="TTS models not initialized")

    # Prepare prompt text with voice prefix (unless voice is "random")
    if request.voice == "random":
        prompt_text = request.input
    else:
        prompt_text = f"{request.voice}: {request.input}"

    # Streaming mode (SSE)
    if request.stream_format == "sse":
        async def sse_generator():
            """Generate Server-Sent Events with audio chunks"""
            import asyncio
            import queue as thread_queue
            chunk_queue = thread_queue.Queue()

            # Custom list wrapper that pushes chunks to queue
            class ChunkList(list):
                def append(self, chunk):
                    super().append(chunk)
                    chunk_queue.put(("chunk", chunk))

            audio_writer = StreamingAudioWriter(
                player,
                output_file=None,
                chunk_size=CHUNK_SIZE,
                lookback_frames=LOOKBACK_FRAMES
            )
            audio_writer.audio_chunks = ChunkList()

            # Track token counts for usage reporting
            input_token_count = 0
            output_token_count = 0

            # Start generation in background task
            async def generate_async():
                nonlocal input_token_count, output_token_count
                try:
                    audio_writer.start()
                    result = await generator._generate_async(
                        prompt_text,
                        audio_writer,
                        max_tokens=MAX_TOKENS
                    )
                    audio_writer.finalize()

                    # Extract token counts from result
                    input_token_count = len(generator.prepare_input(prompt_text))
                    output_token_count = len(result.get('all_token_ids', []))

                    chunk_queue.put(("done", {"input": input_token_count, "output": output_token_count}))
                except Exception as e:
                    print(f"Generation error: {e}")
                    import traceback
                    traceback.print_exc()
                    chunk_queue.put(("error", str(e)))

            # Start generation as async task
            gen_task = asyncio.create_task(generate_async())

            # Stream chunks as they arrive
            try:
                while True:
                    msg_type, data = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: chunk_queue.get(timeout=30)
                    )

                    if msg_type == "chunk":
                        # Convert numpy array to int16 PCM
                        pcm_data = (data * 32767).astype(np.int16)

                        # Encode as base64
                        audio_base64 = base64.b64encode(pcm_data.tobytes()).decode('utf-8')

                        # Send SSE event: speech.audio.delta
                        event_data = {
                            "type": "speech.audio.delta",
                            "audio": audio_base64
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"

                    elif msg_type == "done":
                        # Send SSE event: speech.audio.done with usage stats
                        token_counts = data
                        event_data = {
                            "type": "speech.audio.done",
                            "usage": {
                                "input_tokens": token_counts["input"],
                                "output_tokens": token_counts["output"],
                                "total_tokens": token_counts["input"] + token_counts["output"]
                            }
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"
                        break

                    elif msg_type == "error":
                        # Send error event
                        error_data = {
                            "type": "error",
                            "error": data
                        }
                        yield f"data: {json.dumps(error_data)}\n\n"
                        break

            finally:
                await gen_task

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # Non-streaming mode (complete audio file)
    else:
        try:
            # Create audio writer
            audio_writer = StreamingAudioWriter(
                player,
                output_file=None,
                chunk_size=CHUNK_SIZE,
                lookback_frames=LOOKBACK_FRAMES
            )
            audio_writer.start()

            # Generate speech
            result = await generator._generate_async(
                prompt_text,
                audio_writer,
                max_tokens=MAX_TOKENS
            )

            # Finalize and get audio
            audio_writer.finalize()

            if not audio_writer.audio_chunks:
                raise HTTPException(status_code=500, detail="No audio generated")

            # Concatenate all chunks
            full_audio = np.concatenate(audio_writer.audio_chunks)

            # Return based on response_format
            if request.response_format == "pcm":
                # Return raw PCM (int16)
                pcm_data = (full_audio * 32767).astype(np.int16)
                return Response(
                    content=pcm_data.tobytes(),
                    media_type="application/octet-stream",
                    headers={
                        "Content-Type": "application/octet-stream",
                        "X-Sample-Rate": "22050",
                        "X-Channels": "1",
                        "X-Bit-Depth": "16"
                    }
                )
            else:  # wav
                # Convert to WAV bytes
                wav_buffer = io.BytesIO()
                wav_write(wav_buffer, 22050, full_audio)
                wav_buffer.seek(0)

                return Response(
                    content=wav_buffer.read(),
                    media_type="audio/wav"
                )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Kani TTS API",
        "version": "1.0.0",
        "endpoints": {
            "/tts": "POST - Generate complete audio",
            "/stream-tts": "POST - Stream audio chunks",
            "/v1/audio/speech": "POST - OpenAI-compatible speech generation",
            "/health": "GET - Health check"
        }
    }


if __name__ == "__main__":
    import uvicorn
    print("🎤 Starting Kani TTS Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
