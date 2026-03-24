from human_eval.data import read_problems
from llm1_functions import LLM1
from llm2_functions_v2 import LLM2_estimate, LLM2_filter, LLM2_DPO
from llm3_functions_v2 import LLM3_optimize, LLM3_filter, LLM3_DPO
from llm_path import *
import re
import traceback
MAX_CIRC = 150
MAX_CYCLE = 50


circ = 0
cycle = 0
problems = read_problems()



for cycle in range(MAX_CYCLE):
    codes = {}
    test_cases = {}
    dpo_pairs = {}

    for circ in range(MAX_CIRC):
        id = f"HumanEval/{circ}"
        problem = problems[id]
        test = problem['test']
        pattern = r'candidate\((.*?)\)\s*=='
        matches = re.findall(pattern, test, re.DOTALL)

        for m in matches:
            test = m.strip()
            test = re.sub(r'\s+', '', test)

        test_cases[id] = test

    for circ in range(MAX_CIRC):
        HE_ready = True

        id = f"HumanEval/{circ}"
        problem = problems[id]
        prompt = problem['prompt']
        test = problem['test']
        test_case = test_cases[id]

        code = LLM1(prompt, llm1_qwen_path)
        codes[id] = code

        llm2_qwen_path = "/home/yjx/qwen/Qwen3-4B" + f"_llm2_v{circ}" if circ > 0 else "/home/yjx/qwen/Qwen3-4B"

        estimation_pair = LLM2_estimate(code, llm2_qwen_path)
        dpo_pair = LLM2_filter(code, estimation_pair, test_case)
        dpo_pairs[id] = dpo_pair

    for dpo_pair in dpo_pairs:
        if dpo_pair["chosen"] != {"error": None, "line": None}:
            HE_ready = False
        
    if HE_ready == False:
        llm2_model = LLM2_DPO(dpo_pairs, llm2_input_path, llm2_output_path)

        dpo_groups = {}
        for circ in range(MAX_CIRC):
            id = f"HumanEval/{circ}"
            chosen = dpo_pairs[id]["chosen"]
            code = codes[id]
            test_case = test_cases[id]
            prompt = problems[id]['prompt']

            llm3_qwen_path = "/home/yjx/qwen/Qwen3-4B" + f"_llm3_v{circ}" if circ > 0 else "/home/yjx/qwen/Qwen3-4B"

            optimized_code_group = LLM3_optimize(chosen, code, llm3_qwen_path)
            dpo_group, optimized_code = LLM3_filter(chosen, optimized_code_group, test_case, prompt)
            dpo_groups[id] = dpo_group

            codes[id] = optimized_code

        llm3_model = LLM3_DPO(dpo_groups, llm3_input_path, llm3_output_path)

    elif HE_ready:
        all_pass = True
        for circ in range(MAX_CIRC):
            id = f"HumanEval/{circ}"
            prompt = problems[id]['prompt']
            test = problems[id]['test']

            test_code = code[id]

            test_code = test.replace('    """', '    """'+test_code)

            total_count = len(re.findall(r'assert\s+candidate', test))
            pass_count = 0

            try:
                local_namespace = {}
                exec(test_code, local_namespace)
                intersperse_func = local_namespace["intersperse"]
                check_func = local_namespace["check"]

                check_func(intersperse_func)

                print(f"Pass {pass_count}/{total_count}")

            except AssertionError:
                all_pass = False
                print(f"AssertionError {pass_count}/{total_count}")

            except Exception as e:
                all_pass = False
                tb = traceback.extract_tb(e.__traceback__)
                line = tb[-1].lineno if tb else e.lineno if hasattr(e, 'lineno') else -1

                print(f"Error Type: {type(e).__name__}, line: {line}")

        if all_pass:
            print("All pass. Training done.")
            break
        else:
            continue