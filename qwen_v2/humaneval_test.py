from human_eval.data import read_problems

problems = read_problems()

task_id = "HumanEval/5"
problem = problems[task_id]

print(f"Task ID: {task_id}")
print(f"Prompt: {problem['prompt']}")
print(f"Test cases: {problem['test']}")