import numpy as np
from llm1_functions import LLM1
from llm2_functions import LLM2_estimate, LLM2_filter, LLM2_DPO
from llm3_functions import LLM3_optimize, LLM3_filter, LLM3_DPO, LLM3_exec
from concurrent.futures import ProcessPoolExecutor
import sys
# from corrections import defaultdict

############
# __main__ #
############

MAX_GEN = 50
MAX_CYCLE = 50

'''
# llm1_code, test_cases, problem_description are lists, containing 50 elements respectively.
def run_llm1_task(gen, user_input):
    print(f"### LLM1 generating No.{gen+1} ###")
    llm1_qwen_path = "/home/yjx/qwen/Qwen3-4B"
    llm1_code_single, test_cases_single, problem_description_single = LLM1(user_input, llm1_qwen_path)

    return llm1_code_single, test_cases_single, problem_description_single

def run_llm2_task(gen, llm1_code, test_cases, alles_circ):
    print(f"### LLM2 generating No.{gen+1} ###")
    code = llm1_code[gen]
    test = test_cases[gen]
    llm2_qwen_path = "/home/yjx/qwen/Qwen3-4B" + str(alles_circ) if alles_circ != 0 else "/qwen/Qwen3-4B"

    alles_llm2_estimates = LLM2_estimate(code, test, llm2_qwen_path)
    dpo_pair = LLM2_filter(alles_llm2_estimates, test, code)

    return dpo_pair

def run_llm3_task(gen, dpo_pairs_llm2, problem_desc):
    print(f"### LLM3 generating No.{gen+1} ###")
    code = dpo_pairs_llm2[0]["llm1_code"]
    estimate = {}
    for group in dpo_pairs_llm2:
        estimate[group["test_case"]] = group["chosen_llm2_estimate"]

    llm3_qwen_path = "/qwen/Qwen3-4B" + str(alles_circ) if alles_circ != 0 else "/qwen/Qwen3-4B"

    alles_llm3_codes = LLM3_optimize(estimate, code, llm3_qwen_path)

    dpo_pair = LLM3_filter(estimate, alles_llm3_codes, estimate, problem_desc)
    # dpo_pair的第三个元素本应是test_cases, 但是由于函数只使用了test_cases的key, 可以直接用estimate代替

    return dpo_pair
'''

def chunk_list(data, size):
    for i in range(0, len(data), size):
        yield data[i:i+size]

'''
This function is not used now.
def group_by_llm1_code(dpo_pairs_llm2):
    groups = defaultdict(list)

    for item in dpo_pairs_llm2:
        groups[item["llm1_code"]].append(item)

    return list(groups.values())
'''

alles_circ = 0
for alles_circ in range(MAX_CYCLE):
    HE_ready = 1

    # LLM1
    if alles_circ == 0:
        llm1_code = []
        test_cases = []
        problem_description = []

        user_input = input("User input: ")
        print("### LLM1 generating ###")
        # llm1_qwen_path = "/home/yjx/qwen/Qwen3-4B"
        # llm1_code, test_cases, problem_description = LLM1(user_input, llm1_qwen_path)

        qwen_path = "/home/yjx/qwen/Qwen3-4B"
        # for gen in range(MAX_GEN):
        print(f"### LLM1 generating No.1 ###")
        code_single, test_single, desc_single = LLM1(user_input, qwen_path)
        llm1_code.append(code_single)
        test_cases.append(test_single)
        problem_description.append(desc_single)

        def run_llm1_code(gen):
            print(f"### LLM1 generating No.{gen+1} ###")
            return LLM1(user_input, qwen_path)
        
        remaining_gen = list(range(1, MAX_GEN))
        with ProcessPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(run_llm1_code, remaining_gen))

        for res in results:
            code_single, test_single, desc_single = res
            llm1_code.append(code_single)
            test_cases.append(test_single)
            problem_description.append(desc_single)


        '''
        def run_llm1_task(gen):
            return LLM1(user_input, qwen_path)

        with ProcessPoolExecutor(max_workers=MAX_GEN) as executor:
            futures = [executor.submit(run_llm1_task, gen) for gen in range(MAX_GEN)]
            for future in futures:
                code_single, test_single, desc_single = future.result()
                llm1_code.extend(code_single)
                test_cases.extend(test_single)
                problem_desciption.extend(desc_single)
        '''
        
        with open("llm1_code.txt", "w", encoding="utf-8") as f:
            f.write(str(llm1_code))

        with open("test_cases.txt", "w", encoding="utf-8") as f:
            f.write(str(test_cases))

        with open("problem_description", "w", encoding="utf-8") as f:
            f.write(str(problem_description))

        sys.exit(0)
        

    '''
    for gen in range(MAX_GEN):
        print(f"### LLM2 generating No.{gen+1} ###")
        code = llm1_code[gen]
        test = test_cases[gen]
        llm2_qwen_path = "/qwen/Qwen3-4B" + str(alles_circ) if alles_circ != 0 else "/qwen/Qwen3-4B"

        alles_llm2_estimates = LLM2_estimate(code, test, llm2_qwen_path)
        dpo_pair = LLM2_filter(alles_llm2_estimates, test, code)
        dpo_pairs.extend(dpo_pair)
    '''
    # LLM2
    dpo_pairs_llm2 = []

    # This code block is not used now.
    '''
    with ProcessPoolExecutor(max_workers=MAX_GEN) as executor:
        futures = [
            executor.submit(run_llm2_task, gen, llm1_code, test_cases, alles_circ)
            for gen in range(MAX_GEN)
        ]

        for future in futures:
            dpo_pairs_llm2.extend(future.result())
            # dpo_pairs_llm2 is a dict with 50k elements.
    '''

    for gen in range(MAX_GEN):
        print(f"### LLM2 generating No.{gen+1} ###")
        code = llm1_code[gen]
        test = test_cases[gen]
        llm2_qwen_path = "/home/yjx/qwen/Qwen3-4B" + str(alles_circ) if alles_circ != 0 else "/qwen/Qwen3-4B"

        alles_llm2_estimates = LLM2_estimate(code, test, llm2_qwen_path)
        dpo_pair = LLM2_filter(alles_llm2_estimates, test, code)
        dpo_pairs_llm2.extend(dpo_pair)

    for dpo_pair_llm2 in dpo_pairs_llm2:
        if dpo_pair_llm2["chosen_llm2_estimate"] != {"error": None, "line": -1}:
            humaneval_ready = 0

    if humaneval_ready == 0:
        # LLM2 DPO
        print("### LLM2 DPO updating ###")
        llm2_model_input = "/home/yjx/qwen/Qwen3-4B" + str(alles_circ) if alles_circ != 0 else "/home/yjx/qwen/Qwen3-4B"
        llm2_model_output = "/home/yjx/qwen/Qwen3-4B" + str(alles_circ+1)
        llm2_dpo_model = LLM2_DPO(dpo_pairs_llm2, llm2_model_input, llm2_model_output)

        '''
        for gen in range(MAX_GEN):
            print(f"### LLM3 generating No.{gen+1} ###")
            estimate = dpo_pairs[gen]["chosen_llm2_estimate"]
            code = dpo_pairs[gen]["llm1_code"]
            test = dpo_pairs[gen]["test_cases"]
            problem_desc = problem_description[gen]
            llm3_qwen_path = "/qwen/Qwen3-4B" + str(alles_circ) if alles_circ != 0 else "/qwen/Qwen3-4B"

            alles_llm3_codes = LLM3_optimize(estimate, code, llm3_qwen_path)
            dpo_pair = LLM3_filter(estimate, alles_llm3_codes, test, problem_desc)
            dpo_pairs_llm3.extend(dpo_pair)
        '''

        # LLM3
        dpo_pairs_llm3 = []
        llm1_code = []
        chunks = list(chunk_list(dpo_pairs_llm2, 1000))

        '''
        with ProcessPoolExecutor(max_workers=MAX_GEN) as executor:
            results = executor.map(run_llm3_task, chunks)
        
        for result in results:
            dpo_pairs_llm3.extend(result)
            # dpo_pairs_llm3 is a list with 10*50 elements.
        '''

        for gen in range(MAX_GEN):
            print(f"### LLM3 generating No.{gen+1} ###")
            code = dpo_pairs_llm2[0]["llm1_code"]
            estimate = {}
            for group in dpo_pairs_llm2:
                estimate[group["test_case"]] = group["chosen_llm2_estimate"]

            llm3_qwen_path = "/qwen/Qwen3-4B" + str(alles_circ) if alles_circ != 0 else "/qwen/Qwen3-4B"

            alles_llm3_codes = LLM3_optimize(estimate, code, llm3_qwen_path)

            problem_desc = problem_description[gen]

            dpo_pair = LLM3_filter(estimate, alles_llm3_codes, estimate, problem_desc)
            # dpo_pair的第三个元素本应是test_cases, 但是由于函数只使用了test_cases的key, 可以直接用estimate代替

            dpo_pairs_llm3.extend(dpo_pair)

        # LLM3 final output (llm1_code = llm3_output_list)
        llm3_output_list = []
        llm3_output_score = {}
        llm3_output_chunk = list(chunk_list(dpo_pairs_llm3, 10))
        for llm3_outputs in llm3_output_chunk:
            for llm3_output in llm3_outputs:
                if llm3_output["chosen_llm3_code"] not in llm3_output_score:
                    # llm3_output_list.append(llm3_output["chosen_llm3_code"])
                    llm3_output_score[llm3_output["chosen_llm3_code"]] = llm3_output["correct_rate"]
                else:
                    llm3_output_score[llm3_output["chosen_llm3_code"]] += llm3_output["correct_rate"]

            best_key = max(llm3_output_score, key=llm3_output_score.get)
            llm3_output_list.append(best_key)
            llm1_code = llm3_output_list

        with open("updated_code.txt", "a", encoding="utf-8") as f:
            f.write(f"Circ {alles_circ+1}:\n"+str(llm1_code))

        print("### LLM3 DPO updating ###")
        llm3_model_input = "/home/yjx/qwen/Qwen3-4B" + str(alles_circ) if alles_circ != 0 else "/home/yjx/qwen/Qwen3-4B"
        llm3_model_output = "/home/yjx/qwen/Qwen3-4B" + str(alles_circ+1)
        llm3_dpo_model = LLM3_DPO(dpo_pairs_llm3, llm3_model_input, llm3_model_output)

    elif HE_ready == 1:
        print("### LLM2 found no error. Ready to HumanEval. ###")
        is_pass = 1
        for gen in range(MAX_GEN):
            HE_code = llm1_code[gen]
            HE_test = test_cases[gen]
            HE_exec = LLM3_exec(HE_code, HE_test)
            for k, v in HE_test.items():
                if v == HE_exec[k]:
                    is_pass = 0

    if is_pass == 1:
        print("Training ends.")
    else:
        continue
                
