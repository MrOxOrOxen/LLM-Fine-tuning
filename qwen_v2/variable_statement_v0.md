# Function LLM1()

## input

- user_input: str
- qwen_path: str

## output

- llm1_code: 50 elements list, like:

```python
llm1_code = [
    "def is_prime():",
    "def is_prime():",
    ...
    "def is_prime():"
]
```

- test_cases: 50 elements list, with a 1000-element-dictionary in each element of the list, like:

```python
test_cases = [
    {(1,): {"error": "SyntaxError", "line": 5},
     (2,): {"error": "SyntaxError", "line": 5},
     ...
     (1000,): {"error": "SyntaxError", "line": 5}},
    {(1,): {"error": "SyntaxError", "line": 5},
     (2,): {"error": "SyntaxError", "line": 5},
     ...
     (1000,): {"error": "SyntaxError", "line": 5}},
    ...
    {(1,): {"error": "SyntaxError", "line": 5},
     (2,): {"error": "SyntaxError", "line": 5},
     ...
     (1000,): {"error": "SyntaxError", "line": 5}}
]
```

- problem_description: 50 elements list, like:

```python
problem_description = [
    "This is a desc.",
    "This is another desc.",
    ...
    "This is the final desc."
]
```

# Function LLM2_estimate()

## input

- code: a **single** str
- test_cases: a dict with 1k elements in it, like:

```python
test_cases = {
    (1,): {"error": "SyntaxError", "line": 5},
    (2,): {"error": "SyntaxError", "line": 5},
    ...
    (1000,): {"error": "SyntaxError", "line": 5}
}
```

- qwen_path: a single str

## output

- alles_llm2_estimates: a dict with 1k elements in it, each element has two estimations, like:

```python
alles_llm2_estimates = {
            (1,): {
                "llm2_estimates_1": {"error": "SyntaxError", "line": 5},
                "llm2_estimates_2": {"error": "NameError", "line": 8}
            },
            (2,): {
                "llm2_estimates_1": {"error": "SyntaxError", "line": 5},
                "llm2_estimates_2": {"error": None, "line": -1}
            },
            ...,
            (1000,): {
                "llm2_estimates_1": {"error": "SyntaxError", "line": 5},
                "llm2_estimates_2": {"error": None, "line": -1}
            },
        }
```

# Function LLM2_filter()

## input

- alles_llm2_estimates: a dict with 1k elements in it, each element has two estimations, like:

```python
alles_llm2_estimates = {
            (1,): {
                "llm2_estimates_1": {"error": "SyntaxError", "line": 5},
                "llm2_estimates_2": {"error": "NameError", "line": 8}
            },
            (2,): {
                "llm2_estimates_1": {"error": "SyntaxError", "line": 5},
                "llm2_estimates_2": {"error": None, "line": -1}
            },
            ...,
            (1000,): {
                "llm2_estimates_1": {"error": "SyntaxError", "line": 5},
                "llm2_estimates_2": {"error": None, "line": -1}
            },
        }
```

- test_cases: a dict with 1k elements in it, like:

```python
test_cases = {
    (1,): {"error": "SyntaxError", "line": 5},
    (2,): {"error": "SyntaxError", "line": 5},
    ...
    (1000,): {"error": "SyntaxError", "line": 5}
}
```

- llm1_code: a single str

## output

- dpo_pairs: a list with 1k dictionaries, like:

```python
dpo_pairs = [
    {
        "chosen_llm2_estimate": {"error": "SyntaxError", "line": 5},
        "rejected_llm2_estimate": {"error": "NameError", "line": 8},
        "correct_rate": 0.85,
        "llm1_code": "def is_prime():",
        "test_case": (1,)
    },
    {
        "chosen_llm2_estimate": {"error": "SyntaxError", "line": 5},
        "rejected_llm2_estimate": {"error": "NameError", "line": 8},
        "correct_rate": 0.85,
        "llm1_code": "def is_prime():",
        "test_case": (2,)
    },
    ...
    {
        "chosen_llm2_estimate": {"error": "SyntaxError", "line": 5},
        "rejected_llm2_estimate": {"error": "NameError", "line": 8},
        "correct_rate": 0.85,
        "llm1_code": "def is_prime():",
        "test_case": (1000,)
    }
]
```

# Function LLM2_DPO()

## input

- dpo_pairs: a list with 1k dictionaries, like:

```python
dpo_pairs = [
    {
        "chosen_llm2_estimate": {"error": "SyntaxError", "line": 5},
        "rejected_llm2_estimate": {"error": "NameError", "line": 8},
        "correct_rate": 0.85,
        "llm1_code": "def is_prime():",
        "test_case": (1,)
    },
    {
        "chosen_llm2_estimate": {"error": "SyntaxError", "line": 5},
        "rejected_llm2_estimate": {"error": "NameError", "line": 8},
        "correct_rate": 0.85,
        "llm1_code": "def is_prime():",
        "test_case": (2,)
    },
    ...
    {
        "chosen_llm2_estimate": {"error": "SyntaxError", "line": 5},
        "rejected_llm2_estimate": {"error": "NameError", "line": 8},
        "correct_rate": 0.85,
        "llm1_code": "def is_prime():",
        "test_case": (1000,)
    }
]
```

- model_input: a single str

- model_output: a single str

## output

- model: updated model

# Function LLM3_optimize()

## input

- chosen_llm2_estimate: a dict with 1k elements, each has a test_case and a code, like:

```python
chosen_llm2_estimate = {
    (1,): "def is_prime():",
    (2,): "def is_prime():",
    ...
    (1000,): "def is_prime():"
}
```

- code: a single str

- qwen_path: str

## output:

- alles_llm3_codes: a dict with 5 elements, like:

```python
alles_llm3_codes = {
            "llm3_code_1": "def is_prime(n):...",
            "llm3_code_2": "def is_prime(n):...",
            "llm3_code_3": ...,
            "llm3_code_4": ...,
            "llm3_code_5": ...
        }
```

# Function LLM3_exec()

## input

- code: a single str

- test_cases: a dict with 1k elements, like:

```python
test_cases = {
    (1,): {"error": "SyntaxError", "line": 5},
    (2,): {"error": "SyntaxError", "line": 5},
    ...
    (1000,): {"error": "SyntaxError", "line": 5}
}
```

## output

- llm3_exec_results: a dict with 1k elements, like:

```python
llm3_exec_results = {
    (1,): {"error": "SyntaxError", "line": 5},
    (2,): {"error": None, "line": -1},
    ...
    (1000,): {"error": None, "line": -1}
}
```

# Function LLM3_filter()

## input

- chosen_llm2_estimates: a wasted input

- alles_llm3_codes: a dict with 5 elements, like:

```python
alles_llm3_codes = {
            "llm3_code_1": "def is_prime(n):...",
            "llm3_code_2": "def is_prime(n):...",
            "llm3_code_3": ...,
            "llm3_code_4": ...,
            "llm3_code_5": ...
        }
```

- test_cases: a dict with 1k elements, like:

```python
test_cases = {
    (1,): {"error": "SyntaxError", "line": 5},
    (2,): {"error": "SyntaxError", "line": 5},
    ...
    (1000,): {"error": "SyntaxError", "line": 5}
}
```

- problem_description: a single str

## output

- dpo_pairs: a list with 10 elements, comparing every two codes of the input, like:

```python
dpo_pairs = [
    {
        "chosen_llm3_code": "def is_prime():...",
        "rejected_llm3_code": "def is_prime():...",
        "correct_rate": 0.85,
        "problem description": ...
    },
    {
        "chosen_llm3_code": "def is_prime():...",
        "rejected_llm3_code": "def is_prime():...",
        "correct_rate": 0.6,
        "problem description": ...
    },
    ...
    {
        "chosen_llm3_code": "def is_prime():...",
        "rejected_llm3_code": "def is_prime():...",
        "correct_rate": 0.45,
        "problem description": ...
    }
]
```

# Function LLM3_DPO()

## input

- dpo_pairs: a list with 50k elements, where every 10 elements belong to a same code, like:

```python
dpo_pairs = [
    {
        "chosen_llm3_code": "def is_prime():...",
        "rejected_llm3_code": "def is_prime():...",
        "correct_rate": 0.85,
        "problem description": ...
    },
    {
        "chosen_llm3_code": "def is_prime():...",
        "rejected_llm3_code": "def is_prime():...",
        "correct_rate": 0.6,
        "problem description": ...
    },
    ...
    {
        "chosen_llm3_code": "def is_prime():...",
        "rejected_llm3_code": "def is_prime():...",
        "correct_rate": 0.45,
        "problem description": ...
    }
]
```

- model_input: a single str

- model_output: a single str

## output

- model: updated model