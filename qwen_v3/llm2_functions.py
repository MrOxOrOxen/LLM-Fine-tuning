from qwen_command import cmd_llm2_estimate
import json
import traceback
from qwen_run_v2 import qwen
from ast import literal_eval
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer

MAX_ATTEMPT = 5

def LLM2_estimate(codes: list[str], qwen_path: str):
    messages_list = []

    for code in codes:
        for _ in range(MAX_ATTEMPT):
            messages = [
                {"role": "system", "content": cmd_llm2_estimate}, 
                {"role": "user", "content": f"Code: {code}"}
            ]
            messages_list.append(messages)

    estimations_batch = qwen(messages_list, qwen_path)

    results = []
    idx = 0

    for code in codes:
        estimations = []

        for _ in range(MAX_ATTEMPT):
            estimation_str = estimations_batch[idx].strip()
            idx += 1

            if estimation_str.startswith(("'", '"')) and estimation_str.endswith(("'", '"')):
                estimation_str = estimation_str[1:-1]

            try:
                estimation = json.loads(estimation_str)
            except:
                estimation = {"error": "parse_error", "line": -1}

            estimations.append(estimation)

        estimation_cache = estimations[0]
        estimation_pair = [estimation_cache]

        found_diff = False

        for est in estimations[1:]:
            if est != estimation_cache:
                estimation_pair.append(est)
                found_diff = True
                break

        if not found_diff:
            last_est = estimations[-1]

            if last_est["error"] is None:
                estimation_pair.append({"error": None, "line": 1})
            else:
                estimation_pair.append({
                    "error": last_est["error"],
                    "line": last_est["line"] + 1
                })

        results.append(estimation_pair)

    return results

def LLM2_exec(code: str, test_case: str):
    try:
        namespace = {}
        exec(code, namespace)

        funcs = next(v for v in namespace.values() if callable(v))
        func = funcs[0]
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
    
def LLM2_filter(code: str, estimation_pair: list, test_case: list):
    score = [0, 0]
    dpo_pair = {}
    for value in test_case:
        exec_result = LLM2_exec(code, value)
        for i in range(2):
            if estimation_pair[i] == exec_result:
                score[i] += 1
            elif estimation_pair[i]["error"] == exec_result["error"]:
                score[i] += 0.5
            elif estimation_pair[i]["line"] == exec_result["line"]:
                score[i] += 0.5

    if score[1] > score[0]:
        chosen = estimation_pair[1]
        rejected = estimation_pair[0]
    else:
        chosen = estimation_pair[0]
        rejected = estimation_pair[1]    

    dpo_pair["chosen"] = chosen
    dpo_pair["rejected"] = rejected
    dpo_pair["code"] = code
    dpo_pair["test_case"] = test_case

    return dpo_pair

def LLM2_DPO(dpo_pairs: dict, input_path: str, output_path: str):
    chosens = []
    rejecteds = []
    prompts = []

    for dpo_pair in dpo_pairs:
        chosen = dpo_pair["chosen"]
        rejected = dpo_pair["rejected"]
        code = dpo_pair["code"]
        test_case = dpo_pair["test_case"]
        
        prompt = f"""
    You are an expert Python debugger.

    Given the following Python function:
    {code}

    It will be tested with inputs:
    {test_case}

    Predict the execution result in JSON format:
    {{"error": <error_type_or_None>, "line": <line_number>}}
    """.strip()
    
    prompts.append(prompt)
    chosens.append(json.dumps(chosen))
    rejecteds.append(json.dumps(rejected))

    dpo_dataset = Dataset.from_dict(
        {
            "prompt": prompts,
            "chosen": chosens,
            "rejected": rejecteds,
        }
    )

    model_name = input_path

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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
        output_dir=training_output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        num_train_epochs=3,
        logging_steps=5,
        save_steps=50,
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
        max_prompt_length=512
    )

    trainer.train()
    model_output_dir = output_path + "/model"
    trainer.save_model(model_output_dir)

    return model