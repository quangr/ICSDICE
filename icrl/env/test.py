import os
import importlib.util

def load_cost_function(envs_id):
    # Get the directory of the current file
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    # Set the path to the envs directory relative to the current file's location
    envs_dir = os.path.join(current_file_dir, "envs")

    # Find all files in the envs directory
    files = os.listdir(envs_dir)

    # Look for the Python module that matches the envs_id
    for file in files:
        if file.endswith(".py") and file[:-3] == envs_id:
            module_name = f"envs.{file[:-3]}"
            module_path = os.path.join(envs_dir, file)

            # Dynamically load the module
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check if the cost_function function exists in the module
            if hasattr(module, "cost_function"):
                return module
            else:
                raise AttributeError(f"Module '{module_name}' does not have 'cost_function' function.")
    
    # If the envs_id does not match any module, raise an error
    raise ValueError(f"No module found for envs_id '{envs_id}' in the 'envs' directory.")

# Example usage
envs_id = "BiasedReacher"
try:
    cost_function = load_cost_function(envs_id)
    print(cost_function)
    # Now you can use the cost_function
    # For example: result = cost_function(some_arguments)
except ValueError as e:
    print(e)
except AttributeError as e:
    print(e)
