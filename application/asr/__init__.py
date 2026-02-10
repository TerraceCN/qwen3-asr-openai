# -*- coding: utf-8 -*-

from .openai import asr_openai
from .dashscope_async import asr_dashscope_async

__all__ = ["asr_openai", "asr_dashscope_async"]
