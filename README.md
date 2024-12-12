# TradeForecast



## Setting up virtual environment and adding PYTHONPATH

```
# Create virtual env using virtualenv
virtualenv .venv

# Edit .venv/bin/activate
nano .venv/bin/activate

# Add this line at the end of .venv/bin/activate
export PYTHONPATH=$PWD

# Add this line in the deactivate function in .venv/bin/activate
unset PYTHONPATH # This will reset this variable once you deactivate your venv

# save the changes and close the venv/bin/activate file

# Make sure to activate your virtual env from your project root
source venv/bin/activate
echo $PYTHONPATH # Check if the variable is created
```
