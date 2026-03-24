from qwen_run import qwen
from system_command import *
import ast
from code_exec import run_code, get_answer

# LLM1
def LLM1(task):
    '''
    Docstring for LLM1
    
    :param task: task description
    '''
    print("===LLM1: Initial code generating===")
    system_command = system_command_llm1
    messages = [
        {"role": "system", "content": system_command}, 
        {"role": "user", "content": task}
    ]
    code = qwen(messages)

    return code

# LLM2
def LLM2_actual_error(code, *args):
    '''
    Docstring for LLM2_actual_error
    
    :param code: LLM1 output
    :param args: function parameters
    '''
    print("===Calculating LLM2 actual error===")
    return run_code(code, *args)

def LLM2(code):
    '''
    Docstring for LLM2
    
    :param code: LLM1 output
    '''
    print("===LLM2: Generating estimate error===")
    system_command = system_command_llm2
    messages = [
        {"role": "system", "content": system_command}, 
        {"role": "user", "content": code}
    ]
    estimate_errors = qwen(messages)
    estimate_errors = ast.literal_eval(estimate_errors)

    return estimate_errors

def LLM2_feedback(estimate_errors, actual_error_dict):
    print("===Calculating LLM2 feedback value===")
    llm2_feedback = 0
    for args in actual_error_dict:
        if actual_error_dict[args]["error_type"] == None:
            if estimate_errors == {}:
                llm2_feedback += 1
            else:
                llm2_feedback -= sum(estimate_errors.values())
        else:
            for estimate_error in estimate_errors:
                if estimate_error == actual_error_dict[args]["error_type"]:
                    llm2_feedback += estimate_errors[estimate_error]
    llm2_feedback /= len(actual_error_dict)
    return llm2_feedback

# LLM3
def LLM3(code, estimate_errors):
    '''
    Docstring for LLM3
    
    :param code: LLM1 output
    :param estimate_errors: LLM2 estimate errors
    '''
    print("===LLM3: Code optimizing===")
    system_command = system_command_llm3
    messages = [
        {"role": "system", "content": system_command}, 
        {"role": "user", "content": f"Code: {code}, estimate_errors: {estimate_errors}"}
    ]
    llm3_output = qwen(messages)

    return llm3_output

def LLM3_actual_error(fixed_code, *args):
    '''
    Docstring for LLM3_actual_error
    
    :param fixed_code: LLM3 fixed code
    :param args: function parameters
    '''
    print("===Calculating LLM3 actual error===")
    return run_code(fixed_code, *args)
    
def LLM3_feedback(llm2_actual_error_dict, llm3_actual_error_dict, args_dict):
    '''
    Docstring for LLM3_feedback
    
    :param llm2_actual_error_dict: dictionary of LLM2 actual error
    :param llm3_actual_error_dict: dictionary of LLM3 actual error
    :param args_dict: function parameters and correct outputs
    '''
    print("===Calculating LLM3 feedback value===")
    llm3_feedback = 0
    for args in llm2_actual_error_dict:
        llm2_err = llm2_actual_error_dict[args]
        llm3_err = llm3_actual_error_dict[args]
        if llm3_err["error_type"] == args_dict[args]:
            llm3_feedback += 1
        else:
            if not llm3_err["has_error"]:
                llm3_feedback += 1
            else:
                if llm3_err["error_type"] == llm2_err["error_type"]:
                    llm3_feedback += 0
                else:
                    llm3_feedback += 0.5
    llm3_feedback /= len(llm2_actual_error_dict)
    return llm3_feedback

# LLM4
def LLM4(fixed_code, llm3_feedback):
    '''
    Docstring for LLM4
    
    :param fixed_code: LLM3 fixed code
    :param llm3_feedback: the value of llm3_feedback
    '''
    print("===LLM4: Code Optimizing===")
    system_command = system_command_llm4
    messages = [
        {"role": "system", "content": system_command}, 
        {"role": "user", "content": f"Code: {fixed_code}, no_error_prob: {llm3_feedback}"}
    ]
    llm4_output = qwen(messages)

    return llm4_output

def LLM4_feedback(llm4_answer_dict, args_dict):
    '''
    Docstring for LLM4_feedback
    
    :param code: LLM4 fixed code
    :param args_dict: function parameters and correct outputs
    '''
    print("===Calculating LLM4 feedback value===")
    llm4_feedback = 0
    for args in llm4_answer_dict:
        if llm4_answer_dict[args] == args_dict[args]:
            llm4_feedback += 1
        else:
            llm4_feedback += 0
    llm4_feedback /= len(args_dict)

    return llm4_feedback