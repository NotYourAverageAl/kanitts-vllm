"""Text-to-speech generation modules"""

from .generator import TTSGenerator
from .vllm_generator import VLLMTTSGenerator

__all__ = ['TTSGenerator', 'VLLMTTSGenerator']
