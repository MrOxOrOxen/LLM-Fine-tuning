import ast, traceback, sys
from qwen_run import qwen
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
from qwen_command import cmd_llm3_optimize

# MAX_ATTEMPT = 5
MAX_ATTEMPT = 25

def LLM3_optimize(chosen_llm2_estimates:dict, code:str, qwen_path: str):
    '''
    Docstring for LLM3_optimize
    
    :param chosen_llm2_estimates: the better LLM2 estimated result
    :type chosen_llm2_estimates: dict
    :param code: LLM1 code
    :type code: str

    Output in the following form:
        alles_llm3_codes = {
            "llm3_code_1": "def is_prime(n):...",
            "llm3_code_2": "def is_prime(n):...",
            "llm3_code_3": ...,
            "llm3_code_4": ...,
            "llm3_code_5": ...
        }
    '''
    messages = [
        {"role": "system", "content": cmd_llm3_optimize}, 
        {"role": "user", "content": f"LLM2 Estimate: {chosen_llm2_estimates}, code: {code}"}
    ]
    attempt = 0
    alles_llm3_codes = []
    code_cache = set()
    index = 1
    # llm3_code_cache = ""
    while attempt < MAX_ATTEMPT and index <= 5:
        llm3_code = qwen(messages, qwen_path).strip()
        if llm3_code.startswith(("'", '"')) and llm3_code.endswith(("'", '"')):
            llm3_code = llm3_code[1:-1]

        if llm3_code not in code_cache:
            alles_llm3_codes[f"llm3_code_{index}"] = llm3_code
            code_cache.add(llm3_code)
            index += 1
        
        attempt += 1

    while index <= 5:
        alles_llm3_codes[f"llm3_code_{index}"] = llm3_code
        index += 1

    return alles_llm3_codes

def LLM3_exec(code:str, test_cases:dict):
    '''
    Docstring for LLM3_exec
    
    :param code: LLM3 optimized code
    :type code: str
    :param test_cases: test cases dictionary
    :type test_cases: dict

    Output in the following form:
        llm3_exec_results = {
            (1,): {"error": "SyntaxError", "line": 5},
            (2,): {"error": None, "line": -1},
            and so on,
        }
    '''
    llm3_exec_results = {}
    for test_case in test_cases:
        namespace = {"__builtins__": __builtins__}
        try:
            tree = ast.parse(code)
            function_name = None
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    function_name = node.name
                    break

            if function_name is None:
                llm3_exec_results[test_case] = {"error": "NoFunctionError", "line": 0}
                continue
            
            exec(code, namespace)
            func = namespace.get(function_name)

            if func is None:
                llm3_exec_results[test_case] = {"error": "FunctionNotFoundError", "line": 0}
                continue
            
            func(*test_case)
            llm3_exec_results[test_case] = {"error": None, "line": -1}
        
        except SyntaxError as e:
            llm3_exec_results[test_case] = {"error": type(e).__name__, "line": e.lineno}
        
        except Exception as e:
            error_type = type(e).__name__
            tb_frames = traceback.extract_tb(sys.exc_info()[2])
            for frame in reversed(tb_frames):
                if frame.filename == "<string>":
                    llm3_exec_results[test_case] = {"error": error_type, "line": frame.lineno}
                    found = True
                    break

            if not found:
                llm3_exec_results[test_case] = {"error": error_type, "line": 0}

    return llm3_exec_results

def LLM3_filter(chosen_llm2_estimates:dict, alles_llm3_codes:dict, test_cases:dict, problem_description: str):
    '''
    Docstring for LLM3_filter
    
    :param chosen_llm2_estimates: the better LLM2 estimated result
    :type chosen_llm2_estimates: dict
    :param alles_llm3_codes: LLM3 optimized code group
    :type alles_llm3_codes: dict
    :param test_cases: test cases dictionary
    :type test_cases: dict

    Output in the following form:
        dpo_pairs = [
            {
                "chosen_llm3_code": "def is_prime():...",
                "rejected_llm3_code": "def is_prime():...",
                "correct_rate": 0.85,
                "problem description": ...
            },
            and so on,
        ]
    '''
    code_scores = {}
    
    # chosen_llm3_code = ""
    # rejected_llm3_code = ""
    # point = [0, 0]
    for key, llm3_code in alles_llm3_codes.items():
        pass_count = 0
        llm3_exec_results = LLM3_exec(llm3_code, test_cases)

        for test_case, llm3_exec_result in llm3_exec_results.items():
            if llm3_exec_result == {"error": None, "line": -1}:
                pass_count += 1
        pass_rate = pass_count / len(test_cases)

        code_scores[key] = {
            "code": llm3_code,
            "pass_rate": pass_rate
        }

    sorted_codes = sorted(
        code_scores.values(),
        key=lambda x: x["pass_rate"],
        reverse=True
    )

    dpo_pairs = []
    for i in range(len(sorted_codes)):
        for j in range(i+1, len(sorted_codes)):
            chosen_llm3_code = sorted_codes[i]["code"]
            rejected_llm3_code = sorted_codes[j]["code"]
            correct_rate = sorted_codes[i]["pass_rate"]

            dpo_pairs.append(
                {
                    "chosen_llm3_code": chosen_llm3_code,
                    "rejected_llm3_code": rejected_llm3_code,
                    "correct_rate": correct_rate,
                    "problem_description": problem_description
                }
            )

    return dpo_pairs

    for serial, llm3_code in enumerate(alles_llm3_codes.values()):
        llm3_exec_results = LLM3_exec(llm3_code, test_cases)
        for key, llm3_exec_result in llm3_exec_results.items():
            if llm3_exec_result == {"error": None, "line": -1}:
                point[serial] += 1
            elif llm3_exec_result == chosen_llm2_estimates[key]:
                point[serial] += 0
            else:
                point[serial] -= 1
        serial += 1
    if point[0] < point[1]:
        chosen_llm3_code = alles_llm3_codes["llm3_code_2"]
        rejected_llm3_code = alles_llm3_codes["llm3_code_1"]
        correct_rate = point[1] / len(test_cases)
    else:
        chosen_llm3_code = alles_llm3_codes["llm3_code_1"]
        rejected_llm3_code = alles_llm3_codes["llm3_code_2"]
        correct_rate = point[0] / len(test_cases)

    return chosen_llm3_code, rejected_llm3_code, correct_rate

def LLM3_DPO(dpo_pairs: list, model_input: str, model_output: str):
    prompts = []
    chosens = []
    rejecteds = []

    for pair in dpo_pairs:
        problem_desc = pair["problem_description"]
        prompt = f"""
    You are an expert Python programmer.
    Solve the following problem.
    ### Problem
    {problem_desc}

    ### Python solution
    """.strip()
        
        chosen_code = pair["chosen_llm3_code"]
        rejected_code = pair["rejected_llm3_code"]

        if chosen_code == rejected_code:
            continue

        prompts.append(prompt)
        chosens.append(chosen_code)
        rejecteds.append(rejected_code)

    dpo_dataset = Dataset.from_dict(
        {
            "prompt": prompts,
            "chosen": chosens,
            "rejected": rejecteds,
        }
    )

    model_name = model_input

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

    training_output_dir = model_output + "/training_args"
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
    model_output_dir = model_output + "/model"
    trainer.save_model(model_output_dir)

    return model