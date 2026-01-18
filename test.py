import inspect
from llama_index.llms.openai import OpenAI as LLMOpenAI
print("is abstract:", inspect.isabstract(LLMOpenAI))
print("missing abstract methods:", getattr(LLMOpenAI, "__abstractmethods__", None))