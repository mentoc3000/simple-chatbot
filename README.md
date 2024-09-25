# A simple chatbot

## Setup

1. Install [pyenv](https://github.com/pyenv/pyenv)
2. `pyenv install 3.11`
3. Install [Poetry](https://python-poetry.org/)
4. `poetry self add poetry-dotenv-plugin`
5. `poetry config virtualenvs.in-project true`
6. `poetry config virtualenvs.prefer-active-python true`
7. `poetry install`

## Run

`poetry run chainlit run app.py`

## Debug

The following command is available to attach VSCode's debugging tools. Run it via the VSCode debugger.

`python debug.py`
