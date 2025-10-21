"""VLLM-based text-to-speech generation logic with async streaming"""

import asyncio
import time
import torch
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from transformers import AutoTokenizer

from config import (
    MODEL_NAME, START_OF_HUMAN, END_OF_TEXT, END_OF_HUMAN, END_OF_AI,
    TEMPERATURE, TOP_P, REPETITION_PENALTY, MAX_TOKENS
)


class VLLMTTSGenerator:
    def __init__(self, tensor_parallel_size=1, gpu_memory_utilization=0.9, max_model_len=2048):
        """Initialize VLLM-based TTS generator with async streaming support

        Args:
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use (0.0 to 1.0)
            max_model_len: Maximum sequence length
        """
        print(f"Loading VLLM AsyncLLMEngine model: {MODEL_NAME}")

        # Configure engine arguments
        engine_args = AsyncEngineArgs(
            model=MODEL_NAME,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=False,  # Allow CUDA graphs (reduces kernel launch overhead)
            max_num_seqs=1,  # Single sequence for TTS - enables better CUDA graph optimization
            dtype="bfloat16",  # BF16 for faster inference on RTX 5090
        )

        # Create async engine
        self.engine = None  # Will be initialized in async context
        self.engine_args = engine_args

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Pre-configure sampling parameters
        self.sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            repetition_penalty=REPETITION_PENALTY,
            stop_token_ids=[END_OF_AI],
        )

    async def initialize_engine(self):
        """Initialize the async engine - call this during startup to avoid lazy loading"""
        if self.engine is None:
            print("Initializing VLLM AsyncLLMEngine...")
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            print("VLLM AsyncLLMEngine initialized and ready!")

    def prepare_input(self, prompt):
        """Build custom input_ids with special tokens"""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        # Add special tokens: [START_OF_HUMAN] + input_ids + [END_OF_TEXT, END_OF_HUMAN]
        start_token = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
        end_tokens = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)
        modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

        # Convert to list for VLLM
        return modified_input_ids[0].tolist()

    async def _generate_async(self, prompt, audio_writer, max_tokens=MAX_TOKENS):
        """Async generator that streams tokens as they are generated

        Args:
            prompt: Text prompt to convert to speech
            audio_writer: StreamingAudioWriter instance to receive tokens
            max_tokens: Maximum number of tokens to generate

        Returns:
            Dictionary with generation metrics and results
        """
        # Initialize engine if needed
        if self.engine is None:
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)

        # Prepare input_ids with special tokens
        input_ids = self.prepare_input(prompt)

        point_1 = time.time()

        # Override max_tokens if different from default
        if max_tokens != MAX_TOKENS:
            sampling_params = SamplingParams(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=max_tokens,
                repetition_penalty=REPETITION_PENALTY,
                stop_token_ids=[END_OF_AI],
            )
        else:
            sampling_params = self.sampling_params

        # Generate unique request ID
        request_id = f"tts-{id(prompt)}-{time.time()}"

        # Stream tokens as they are generated
        all_token_ids = []
        audio_token_count = 0
        inside_speech = False

        # Add request to engine with TokensPrompt
        results_generator = self.engine.generate(
            {"prompt_token_ids": input_ids},
            sampling_params,
            request_id=request_id
        )

        async for request_output in results_generator:
            # Get newly generated tokens
            new_token_ids = request_output.outputs[0].token_ids

            # Find which tokens are new since last iteration
            num_new_tokens = len(new_token_ids) - len(all_token_ids)
            if num_new_tokens > 0:
                new_tokens = new_token_ids[-num_new_tokens:]
                all_token_ids.extend(new_tokens)

                # Stream each new token to audio_writer and count audio tokens
                for token_id in new_tokens:
                    # print(f"[VLLM] Token {len(all_token_ids)}: {token_id}")
                    audio_writer.add_token(token_id)

                    # Track audio tokens efficiently during streaming
                    if token_id == audio_writer.player.start_of_speech:
                        inside_speech = True
                    elif token_id == audio_writer.player.end_of_speech:
                        inside_speech = False
                    elif inside_speech:
                        audio_token_count += 1

        point_2 = time.time()
        generation_time = point_2 - point_1

        # Calculate Real Time Factor (RTF)
        # Audio codec runs at 12.5 fps, audio tokens come in groups of 4 per frame
        FRAMES_PER_SECOND = 12.5
        TOKENS_PER_FRAME = 4

        # Calculate audio duration: tokens / 4 = frames, frames / 12.5 = seconds
        num_frames = audio_token_count // TOKENS_PER_FRAME
        audio_duration = num_frames / FRAMES_PER_SECOND
        rtf = generation_time / audio_duration if audio_duration > 0 else 0

        # Calculate token counts
        prompt_tokens = len(input_ids)
        generated_tokens = len(all_token_ids)
        total_tokens = prompt_tokens + generated_tokens

        print(f"\n[VLLM] Generation complete. Prompt tokens: {prompt_tokens}, Generated tokens: {generated_tokens}, Total: {total_tokens}")
        print(f"       Audio tokens: {audio_token_count}, Frames: {num_frames}, Audio duration: {audio_duration:.2f}s")
        print(f"       Generation time: {generation_time:.2f}s, RTF: {rtf:.3f}")

        # OPTIMIZATION: Skip text decoding - it's slow and not needed for TTS

        return {
            'all_token_ids': all_token_ids,
            'generation_time': generation_time,
            'audio_duration': audio_duration,
            'rtf': rtf,
            'point_1': point_1,
            'point_2': point_2
        }

    def generate(self, prompt, audio_writer, max_tokens=MAX_TOKENS):
        """Generate speech tokens from text prompt with streaming

        This is a synchronous wrapper around the async streaming implementation.

        Args:
            prompt: Text prompt to convert to speech
            audio_writer: StreamingAudioWriter instance to receive tokens
            max_tokens: Maximum number of tokens to generate

        Returns:
            Dictionary with generation metrics and results
        """
        # Try to get the current event loop, or create a new one if needed
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, create one
            return asyncio.run(self._generate_async(prompt, audio_writer, max_tokens))
        else:
            # Event loop is running, we need to run in a thread pool
            import concurrent.futures
            import threading

            result = None
            exception = None

            def run_in_new_loop():
                nonlocal result, exception
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result = new_loop.run_until_complete(
                        self._generate_async(prompt, audio_writer, max_tokens)
                    )
                    new_loop.close()
                except Exception as e:
                    exception = e

            thread = threading.Thread(target=run_in_new_loop)
            thread.start()
            thread.join()

            if exception:
                raise exception

            return result
