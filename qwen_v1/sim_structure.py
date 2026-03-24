from pathlib import Path
from code_exec import get_answer
from llm_functions import *

llm1_output = None
llm2_guess = None
llm3_output = None
llm4_output = None

# task = input("Task Description: ")
# llm1_output = LLM1(task)
args_dict = {
        (1,): 3.14,
        (2,): 12.57,
        (1.5,): 7.07,
        ("2",): "TypeError",
        (0,): "TypeError",
        (3,): 28.27,
        (-1,): "TypeError",
        (10,): 314.16,
        (19,): 1134.15,
        (9,): 254.47,
    }

file_path = Path("/home/yjx/qwen/qwen_v1/sim_fault.txt")
llm1_output = file_path.read_text()

llm2_estimate_errors = LLM2(llm1_output)
llm2_actual_error_dict = {}
for args in args_dict.keys():
    llm2_actual_error = LLM2_actual_error(llm1_output, *args)
    llm2_actual_error_dict[args] = llm2_actual_error

llm3_output = LLM3(llm1_output, llm2_estimate_errors)
llm3_actual_error_dict = {}
for args in args_dict.keys():
    llm3_actual_error = LLM3_actual_error(llm3_output, *args)
    llm3_actual_error_dict[args] = llm3_actual_error

llm2_feedback = LLM2_feedback(llm2_estimate_errors, llm2_actual_error_dict)
llm3_feedback = LLM3_feedback(llm2_actual_error_dict, llm3_actual_error_dict, args_dict)

llm4_output = LLM4(llm3_output, llm3_feedback)
llm4_answer_dict = {}
for args in args_dict.keys():
    llm4_answer = get_answer(llm4_output, *args)
    llm4_answer_dict[args] = llm4_answer

llm4_feedback = LLM4_feedback(llm4_answer_dict, args_dict)


print(f"Initial try:")
print(f"LLM2 estimate errors: {llm2_estimate_errors}")
print(f"LLM2 actual error: {llm2_actual_error_dict}")
print(f"LLM2 feedback: {llm2_feedback}")
print(f"LLM3 output: {llm3_output}")
print(f"LLM3 actual error: {llm3_actual_error_dict}")
print(f"LLM3 feedback: {llm3_feedback}")
print(f"LLM4 output: {llm4_output}")
print(f"LLM4 answer dict: {llm4_answer_dict}")
print(f"LLM4 feedback: {llm4_feedback}")