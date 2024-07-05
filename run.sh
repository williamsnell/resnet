# Install poetry for python dependencies
curl -sSL https://install.python-poetry.org | python3 -

# Make sure the poetry command is recognised
export PATH="$HOME/.poetry/bin:$PATH"

# Set up the environment
poetry install

poetry shell

# Run the actual scripts
python train_res.py 
