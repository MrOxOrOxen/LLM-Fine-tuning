from transformers import AutoModelForCausalLM, AutoTokenizer

llm2_feedback, llm3_feedback, llm4_feedback = 0, 0, 0

path = "/home/yjx/qwen/Qwen3-4B"
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

def qwen(messages):
    '''
    Docstring for qwen
    
    :param messages: system and user roles
    '''
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768,
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