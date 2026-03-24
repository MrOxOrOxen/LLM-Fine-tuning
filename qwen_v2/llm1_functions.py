from qwen_run import qwen
from qwen_command import cmd_llm1
import json
import ast

def LLM1(prompt: str, qwen_path: str):
    messages = [
        {"role": "system", "content": cmd_llm1},
        {"role": "user", "content": f"Prompt: {prompt}"}
    ]

    code = qwen(messages, qwen_path)
    return code