# Install poetry for python dependencies
curl -sSL https://install.python-poetry.org | python3 -

# Set up the environment
poetry install

poetry shell

# Run the actual scripts
python train_res.py 
