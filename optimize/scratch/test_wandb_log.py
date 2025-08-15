import wandb
import os

print(os.path.realpath(__file__))

with wandb.init() as run:
    run.log_code(os.path.realpath(__file__))
    run.log({"test": 1})
    run.log({"test": 2})
