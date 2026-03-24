from qwen_run_v2 import qwen
from qwen_command import cmd_llm1

def LLM1(prompts: list[str], qwen_path: str):
    messages_list = [
        {"role": "system", "content": cmd_llm1},
        {"role": "user", "content": f"Prompt: {prompt}"}
        for prompt in prompts
    ]

    codes = qwen(messages_list, qwen_path)

    return codes