from chainlit.cli import run_chainlit
from chainlit.config import config

if __name__ == "__main__":
    config.run.watch = True
    run_chainlit("app.py")
