# Function LLM1()

## input

- prompt: a single str
- qwen_path: a single str

## output

- code: a single str

# Function LLM2_estimate()

## input

- code: a single str
- qwen_path: a single str

## output

- estimation_pair: a list
```python
estimation_error = [
    {"error": None, "line": None},
    {"error": SyntaxError, "line": 5}
]
```

# Function LLM2_filter()

## input

- code: a single str
- estimation_pair: as above
- test_case: a list
```python
test_case = [5, 6, 7]
```

## output

- dpo_pair: a dict
```python
dpo_pair = {
    "chosen": {"error": None, "line": None},
    "rejected": {"error": SyntaxError, "line": 5},
    "code": "def is_prime(n):",
    "test_case": [5, 6, 7]
}
```

# Function LLM2_DPO()

## input

- dpo_pairs: a dict
```python
dpo_pairs = {
    "HumanEval/0": dpo_pair as above,
    "HumanEval/1": dpo_pair as above,
    ...
}
```
- input_path: a single str
- output_path: a single str

## output

- model: trained model

# Function LLM3_optimize()

## input

- chosen: a dict
```python
chosen = {"error": None, "line": None}
```
- code: a single str
- qwen_path: a single str

## output

- optimized_code_group: a list
```python
optimized_code_group = [
    "def is_prime(n):",
    "def is_prime(n):",
    "def is_prime(n):"
]
```

# Function LLM3_filter()

## input

- chosen: as above
- optimized_code_group: as above
- test_case: as above
- problem_str: a single str

## output

- dpo_group: a list
```python
dpo_group = [
    dpo_pair as above,
    dpo_pair as above,
    dpo_pair as above
]
```

- optimized_code: a single str

# Function LLM3_DPO()

## input

- dpo_groups: a dict
```python
dpo_groups = {
    "HumanEval/0": dpo_group as above,
    "HumanEval/1": dpo_group as above,
    ...
}
```
- input_path: a single str
- output_path: a single str

## output

- model: trained model