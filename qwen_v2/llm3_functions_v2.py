from qwen_command import cmd_llm3_optimize
from qwen_run import qwen
from ast import literal_eval
import traceback
from itertools import combinations
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer

MAX_ATTEMPT = 10

def LLM3_optimize(chosen: dict, code: str, qwen_path: str):
    messages = [
        {"role": "system", "content": cmd_llm3_optimize}, 
        {"role": "user", "content": f"LLM2 Estimate: {chosen}, code: {code}"}
    ]

    attempt = 0
    optimized_code_group = []
    while attempt < MAX_ATTEMPT:
        optimized_code = qwen(messages, qwen_path)
        if optimized_code.startswith(("'", '"')) and optimized_code.endswith(("'", '"')):
            optimized_code = optimized_code[1:-1]

        if optimized_code not in optimized_code_group:
            optimized_code_group.append(optimized_code)
            attempt += 1
            continue

        else:
            attempt += 1
            continue

    if len(optimized_code_group) > 3:
        optimized_code_group = optimized_code_group[:3]

    elif len(optimized_code_group) < 3:
        to_append = 3 - len(optimized_code_group)
        optimized_code_group += to_append * optimized_code

    return optimized_code_group

def LLM3_exec(code: str, test_case: str):
    try:
        namespace = {}
        exec(code, namespace)

        func = next(v for k, v in namespace.items() if callable(v))
        args = literal_eval(test_case)

        if isinstance(args, dict):
            func(**args)
        else:
            if isinstance(args, tuple):
                func(*args)
            else:
                func(args)

        return {"error": None, "line": -1}
     
    except BaseException as e:
        tb = traceback.extract_tb(e.__traceback__)
        line = tb[-1].lineno if tb else e.lineno if hasattr(e, 'lineno') else -1

        return {"error": type(e).__name__, "line": line}
    
def LLM3_filter(chosen: dict, optimized_code_group: list, test_case: list, problem_desc: str):
    score = [0, 0, 0]
    dpo_group = []
    for optimized_code in optimized_code_group:
        index = 0
        for value in test_case:
            exec_result = LLM3_exec(optimized_code, value)
            if exec_result == chosen:
                score[index] += 1
            elif exec_result["error"] == chosen["error"]:
                score[index] += 0.5
            elif exec_result["line"] == chosen["line"]:
                score[index] += 0.5
            else:
                score[index] -= 0.5
            
            index += 1
    
    for i in range(len(score)):
        if score[i] < 0:
            score[i] = 0

    for i, j in combinations(range(len(score)), 2):
        dpo_pair = {}
        excellence_level = score[i] / score[j] if score[j] != 0 else np.inf
        if excellence_level >= 1:
            chosen = optimized_code_group[i]
            rejected = optimized_code_group[j]

        else:
            chosen = optimized_code_group[j]
            rejected = optimized_code_group[i]
            excellence_level = 1 / excellence_level if excellence_level != 0 else np.inf

        dpo_pair["chosen"] = chosen
        dpo_pair["rejected"] = rejected
        dpo_pair["excellence_level"] = excellence_level
        dpo_pair["problem_description"] = problem_desc
        dpo_group.append(dpo_pair)

    max_index = score.index(max(score))
    optimized_code = optimized_code_group[max_index]

    return dpo_group, optimized_code

def LLM3_DPO(dpo_groups: dict, input_path: str, output_path: str):
    prompts = []
    chosens = []
    rejecteds = []

    for dpo_group in dpo_groups:
        problem_desc = dpo_group["problem_description"]
        chosen = dpo_group["chosen"]
        rejected = dpo_group["rejected"]
        if chosen == rejected:
            continue

        prompt = f"""
        You are an expert Python programmer.
        Solve the following problem.
        ### Problem:
        {problem_desc}

        ### Python Solution
    """.strip()
        
        chosens.append(chosen)
        rejecteds.append(rejected)
        prompts.append(prompt)

    dpo_dataset = Dataset.from_dict(
        {
            "prompt": prompts,
            "chosen": chosens,
            "rejected": rejecteds,
        }
    )

    model_name = input_path

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto"
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto"
    )

    training_output_dir = output_path + "/training_args"
    training_args = TrainingArguments(
        output_dir=training_args,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        report_to="none"
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
        beta=0.1,
        max_length=1024,
        max_prompt_length=256
    )

    trainer.train()
    model_output_dir = output_path + "/model"
    trainer.save_model(model_output_dir)

    return model


        
        
        

        
        


    


    
    


    




        