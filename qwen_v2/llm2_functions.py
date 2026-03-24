import json, sys, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from qwen_run import qwen
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
from qwen_command import cmd_llm2_estimate

MAX_ATTEMPT = 5

def LLM2_estimate(code:str, test_cases:dict, qwen_path: str):
    '''
    Docstring for LLM2_estimate
    
    :param code: LLM1 code
    :type code: str
    :param test_cases: test cases dictionary
    :type test_cases: dict

    Output: a dictionary in the following form:
        alles_llm2_estimates = {
            (1,): {
                "llm2_estimates_1": {"error": "SyntaxError", "line": 5},
                "llm2_estimates_2": {"error": "NameError", "line": 8}
            },
            (2,): {
                "llm2_estimates_1": {"error": "SyntaxError", "line": 5},
                "llm2_estimates_2": {"error": None, "line": -1}
            },
            and so on,
        }
    '''
    # test_cases is in the following form:
    # test_cases = {
    #     (1,): 5,
    #     (2,): {"error": "SyntaxError", "line": 5},
    #     and so on,
    # }
    alles_llm2_estimates = {}
    for test_case in test_cases:
        llm2_estimates_cache = {}
        attempt = 0
        llm2_estimates = None
        llm2_estimates_1 = None
        llm2_estimates_2 = None
        while attempt < MAX_ATTEMPT:
            messages = [
                {"role": "system", "content": cmd_llm2_estimate}, 
                {"role": "user", "content": f"Code: {code}, Test_case: {test_case}"}
            ]

            llm2_estimate_str = qwen(messages, qwen_path).strip()
            if llm2_estimate_str.startswith(("'", '"')) and llm2_estimate_str.endswith(("'", '"')):
                llm2_estimate_str = llm2_estimate_str[1:-1]
            
            try:
                llm2_estimates = json.loads(llm2_estimate_str)
            except:
                print("Json parsing failed.")
                sys.exit(1)
            
            # llm2_estimate_error = llm2_estimate_dict["error"]
            # llm2_estimate_line = llm2_estimate_dict["line"]
            
            if llm2_estimates["error"] is None:
                llm2_estimates_1 = llm2_estimates
                llm2_estimates_2 = {
                    "error": None,
                    "line": 0,
                }
                break
            else:
                if not llm2_estimates_cache:
                    llm2_estimates_cache = llm2_estimates.copy()
                    attempt += 1
                    continue

                if llm2_estimates == llm2_estimates_cache:
                    attempt += 1
                    continue
                else:
                    llm2_estimates_1 = llm2_estimates_cache
                    llm2_estimates_2 = llm2_estimates
                    llm2_estimates_cache = llm2_estimates
                    attempt += 1
                    break

        if attempt == MAX_ATTEMPT:
            if llm2_estimates is None:
                llm2_estimates_1 = {"error": None, "line": -1}
                llm2_estimates_2 = {"error": None, "line": 0}
            else:
                llm2_estimates_1 = llm2_estimates
                llm2_estimates_2 = {
                    "error": llm2_estimates["error"],
                    "line": llm2_estimates["line"] + 1,
                }

        alles_llm2_estimates[test_case] = {
            "llm2_estimates_1": llm2_estimates_1,
            "llm2_estimates_2": llm2_estimates_2
        }

    return alles_llm2_estimates

def LLM2_filter(alles_llm2_estimates: dict, test_cases: dict, llm1_code: str):
    '''
    Docstring for LLM2_filter
    
    :param alles_llm2_estimates: LLM2 estimated result
    :type alles_llm2_estimates: dict
    :param test_cases: test cases dictionary
    :type test_cases: dict

    Output in the following form:
        dpo_pairs = [
            {
                "chosen_llm2_estimate": {"error": "SyntaxError", "line": 5},
                "rejected_llm2_estimate": {"error": "NameError", "line": 8},
                "correct_rate": 0.85,
                "test_case": (1,)
            },
            and so on,
        ]
    '''
    correct_count = 0
    chosen_llm2_estimates = {}
    rejected_llm2_estimates = {}
    for key, alles_llm2_estimate in alles_llm2_estimates.items():
        true_output = test_cases.get(key, {})
        if not isinstance(true_output, dict):
            true_output = {"error": None, "line": -1}
        est1 = alles_llm2_estimate.get("llm2_estimates_1", {"error": None, "line": -1})
        est2 = alles_llm2_estimate.get("llm2_estimates_2", {"error": None, "line": -1})
        chosen = 1
        if est1 == true_output:
            chosen = 1
        elif est2 == true_output:
            chosen = 2
        else:
            true_error = true_output.get("error", None)
            est1_error = est1.get("error", None)
            est2_error = est2.get("error", None)
            if est1_error == true_error and est2_error != true_error:
                chosen = 1
            elif est2_error == true_error and est1_error != true_error:
                chosen = 2
            else:
                true_line = true_output.get("line", -1)
                est1_line = est1.get("line", -1)
                est2_line = est2.get("line", -1)
                if est1_line == true_line and est2_line != true_line:
                    chosen = 1
                elif est2_line == true_line and est1_line != true_line:
                    chosen = 2
                else:
                    count_est1, count_est2 = 0, 0
                    for error_test_case in test_cases.values():
                        error_test_case = error_test_case.get("error") if type(error_test_case) == dict else None
                        if est1_error == error_test_case:
                            count_est1 += 1
                        elif est2_error == error_test_case:
                            count_est2 += 1
                    if count_est1 > count_est2:
                        chosen = 1
                    elif count_est1 < count_est2:
                        chosen = 2
                    else:
                        chosen = 1
        if chosen == 1:
            chosen_llm2_estimates[key] = alles_llm2_estimate["llm2_estimates_1"]
            rejected_llm2_estimates[key] = alles_llm2_estimate["llm2_estimates_2"]
        else:
            chosen_llm2_estimates[key] = alles_llm2_estimate["llm2_estimates_2"]
            rejected_llm2_estimates[key] = alles_llm2_estimate["llm2_estimates_1"]

    for key in chosen_llm2_estimates:
        if type(test_cases[key]) == dict and "error" in test_cases[key]:
            if chosen_llm2_estimates[key] == test_cases[key]:
                correct_count += 1
        else:
            if chosen_llm2_estimates[key] == {"error": None, "line": -1}:
                correct_count += 1
    correct_rate = correct_count / len(chosen_llm2_estimates)

    dpo_pairs = []
    dpo_pairs.append(
        {
            "chosen_llm2_estimate": chosen_llm2_estimates,
            "rejected_llm2_estimate": rejected_llm2_estimates,
            "correct_rate": correct_rate,
            "llm1_code": llm1_code,
            "test_case": test_cases,
        }
    )

    return dpo_pairs

def LLM2_DPO(dpo_pairs: list, model_input: str, model_output: str):
    prompts = []
    chosens = []
    rejecteds = []

    for pair in dpo_pairs:
        llm1_code = pair["llm1_code"]
        chosen_llm2_estimate = pair["chosen_llm2_estimate"]
        rejected_llm2_estimate = pair["rejected_llm2_estimate"]
        test_cases = pair["test_cases"]

        # chosen_str = {str(k): v for k, v in chosen_llm2_estimates.items()}
        # rejected_str = {str(k): v for k, v in rejected_llm2_estimates.items()}
    
        prompt = f"""
    You are an expert Python programmer.
    Here is a Python function:

    {llm1_code}

    It will be tested on {test_cases}.
    """.strip()
        
        prompts.append(prompt)
        chosens.append(str(chosen_llm2_estimate))
        rejecteds.append(str(rejected_llm2_estimate))
    
    dpo_dataset = Dataset.from_dict(
        {
            "prompt": prompts,
            "chosen": chosens,
            "rejected": rejecteds,
        }
    )

    model_name = model_input

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

    training_output_dir = model_output + "/training_args"
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
    model_output_dir = model_output + "/model"
    trainer.save_model(model_output_dir)

    return model
