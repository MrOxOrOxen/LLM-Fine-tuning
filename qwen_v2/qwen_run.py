from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# print("Qwen Called.")

_model_cache = {}

def load_model(path: str):
    if path not in _model_cache:
        print(f"Loading model from: {path}")
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype="auto",
            # torch_dtype = torch.float16,
            device_map="auto",
            # device_map = {"": 1},
            trust_remote_code=True,
            local_files_only=True,
        )
        model.eval()
        _model_cache[path] = (model, tokenizer)

    return _model_cache[path]

def qwen(messages, path):
    '''
    Docstring for qwen
    
    :param messages: system and user roles
    '''
    model, tokenizer = load_model(path)

    '''
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    '''

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768,
        # eos_token_id=tokenizer.eos_token_id,
        # pad_token_id=tokenizer.eos_token_id,
        # do_sample=False,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    code = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    print(f"{thinking_content}")

    return code