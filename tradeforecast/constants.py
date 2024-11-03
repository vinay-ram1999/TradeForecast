from pathlib import Path
import sys
import os

# Make sure to declare the PYTHONPATH variable in-order for this to point the correct directory
"""
Warning: python_path considers the path from which the file is executed, which may vary.
Should find out an alternative, untill then we will declare $PYTHONPATH variable for project working dir in venv and use it.
"""
#python_path = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
#assert python_path == sys.path[1] # Check if python_path matches PYTHONPATH and you have activated the correct .venv

#data_dir = os.path.abspath(os.path.join(sys.path[1], 'data'))
#models_dir = os.path.abspath(os.path.join(sys.path[1], 'models'))

data_dir = Path(f'data').resolve()
models_dir = Path(f'models').resolve()

os.makedirs(data_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

#WARNING use ..model_dir and check it as a dir and then load it into model_dir var