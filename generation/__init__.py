"""Text-to-speech generation modules"""

from .vllm_generator import VLLMTTSGenerator
from .chunking import split_into_sentences, estimate_duration

__all__ = ['VLLMTTSGenerator', 'split_into_sentences', 'estimate_duration']
