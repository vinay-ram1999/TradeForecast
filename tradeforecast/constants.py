import sys
import os

# Make sure to declare the PYTHONPATH variable in-order for this to point the correct directory 
python_path = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
assert python_path == sys.path[1] # Check if python_path matches PYTHONPATH and you have activated the correct .venv

data_dir = os.path.abspath(os.path.join(python_path, 'data'))
models_dir = os.path.abspath(os.path.join(python_path, 'models'))

