from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        model.eval()
        _model_cache[path] = (model, tokenizer)

    return _model_cache[path]

def qwen(messages_list, path, batch_size=4):
    model, tokenizer = load_model(path)

    all_outputs = []

    for i in range(0, len(messages_list), batch_size):
        batch_messages = messages_list[i:i+batch_size]

        texts = [
            tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for message in batch_messages
        ]

        model_inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        for j in range(len(texts)):
            input_len = model_inputs["attention_mask"][j].sum().item()
            output_ids = generated_ids[j][input_len:].tolist()

            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            code = tokenizer.decode(
                output_ids[index:],
                skip_special_tokens=True
            ).strip("\n")

            all_outputs.append(code)

    return all_outputs