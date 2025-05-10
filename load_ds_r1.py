# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "system", "content": "You are an AI assistant."},
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
pipe(messages)