from human_eval.data import read_problems
from llm1_function import LLM1
from llm2_functions import LLM2_estimate, LLM2_filter, LLM2_DPO
from llm3_functions import LLM3_optimize, LLM3_filter, LLM3_DPO
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
    ids = []
    prompts = []

    for circ in range(MAX_CIRC):
        id = f"HumanEval/{circ}"
        problem = problems[id]
        test = problem['test']
        pattern = r'candidate\((.*?)\)\s*=='
        matches = re.findall(pattern, test, re.DOTALL)

        cases = []
        for m in matches:
            case = re.sub(r'\s+', '', m.strip())
            cases.append(case)

        test_cases[id] = cases

    for circ in range(MAX_CIRC):
        id = f"HumanEval/{circ}"
        ids.append(id)
        prompt = problems[id]["prompt"]
        prompts.append(prompt)

    codes_batch = LLM1(prompts, llm1_qwen_path)

    codes = {
        id: code for id, code in zip(ids, codes_batch)
    }

    codes_list = [codes[id] for id in ids]
    test_cases_list = [test_cases[id] for id in ids]

    llm2_qwen_path = "/home/yjx/qwen/Qwen3-4B"
    estimation_pairs = LLM2_estimate(codes_list, llm2_qwen_path)

    dpo_pairs = []

    for id, code, est_pair, test_case in zip(ids, codes_list, estimation_pairs, test_cases_list):
        dpo_pair = LLM2_filter(code, est_pair, test_case)
        dpo_pairs.append(dpo_pair)

    HE_ready = True

    for dpo_pair in dpo_pairs:
        if dpo_pair["chosen"] != {"error": None, "line": -1}:
            HE_ready = False
            break

    if not HE_ready:
        llm2_model = LLM2_DPO(dpo_pairs, llm2_input_path, llm2_output_path)

        dpo_groups = {}
        chosens_list = []
        codes_list = []
        ids = []
        test_cases_list = []
        prompts_list = []

        for circ in range(MAX_CIRC):
            id = f"HumanEval/{circ}"
            ids.append(id)
            chosens_list.append(dpo_pairs[id]["chosen"])
            codes_list.append(codes[id])
            test_cases_list.append(test_cases[id])
            prompts_list.append(problems[id]['prompt'])

        llm3_qwen_paths = [
            "/home/yjx/qwen/Qwen3-4B" + f"_llm3_v{circ}" if circ > 0 else "/home/yjx/qwen/Qwen3-4B"
            for circ in range(MAX_CIRC)
        ]

        optimized_groups = LLM3_optimize(chosens_list, codes_list, llm3_qwen_paths[0])

        for i, id in enumerate(ids):
            optimized_code_group = optimized_groups[i]
            dpo_group, optimized_code = LLM3_filter(
                chosens_list[i],
                optimized_code_group,
                test_cases_list[i],
                prompts_list[i]
            )
            dpo_groups[id] = dpo_group
            codes[id] = optimized_code

        llm3_model = LLM3_DPO(list(dpo_groups.values()), llm3_input_path, llm3_output_path)

