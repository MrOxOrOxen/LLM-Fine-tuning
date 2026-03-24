import ast
import traceback

def run_code(code, *args):
    '''
    Docstring for run_code
    
    :param code: code to run
    :param args: function parameters
    '''
    namespace = {"__builtins__": __builtins__}
    try:
        tree = ast.parse(code)
        function_name = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                break

        if function_name is None:
            return {
                "has_error": True,
                "error_type": "NameError",
                "error_msg": "No function definition found in code",
                "traceback": None,
            }
        
        exec(code, namespace)
        func = namespace.get(function_name)
        if func is None:
            return {
                "has_error": True,
                "error_type": "NameError",
                "error_msg": f"Function '{function_name}' not found after exec",
                "traceback": None,
            }
        
        func(*args)
        return {
            "has_error": False,
            "error_type": None,
            "error_msg": None,
            "traceback": None,
        }
    
    except Exception as e:
        return {
            "has_error": True,
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "traceback": traceback.format_exc(),
        }
    
def get_answer(code, *args):
    namespace = {"__builtins__": __builtins__}
    try:
        tree = ast.parse(code)
        function_name = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                break

        if function_name is None:
            return "NoFunctionError"
        
        exec(code, namespace)
        func = namespace.get(function_name)
        if func is None:
            return "FunctionNotFoundError"
        
        answer = func(*args)
        return answer
    
    except Exception as e:
        return type(e).__name__