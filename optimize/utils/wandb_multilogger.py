"""
Class to spawn separate wandb processes so that multiple runs
started from one python process can be logged separately.

Wandb typically only allows one process to spawn for each python
process, so this allows us to spawn multiple processes for each
wandb run, place them in a queue awaiting inputs, and log to them.
"""

import wandb
import multiprocessing as mp


def worker(project, group, job_type, name, config, mode, queue, save_code_path=None):
    runner = wandb.init(
        project=project,
        group=group,
        job_type=job_type,
        name=name,
        config=config,
        mode=mode,
    )
    # if save_code_path is not None:
    #     runner.log_code(save_code_path)
    try:
        while True:
            data = queue.get()
            if data is None:
                # Sentinel to end logging
                break
            # Log the received data to W&B
            wandb.log(data)
    finally:
        # Ensure W&B run is properly closed
        wandb.finish()


class WandbMultiLogger:
    """
    Keeps a pair of dictionaries indexed by seed indices (0,1,etc.)
    self.processes contains references for each wandb process and
    self.queues keeps a queue for each process indexed by the same key (seed no.)
    """

    def __init__(
        self,
        project,
        group,
        job_type,
        config,
        mode,
        seed,
        num_seeds,
        save_code_path=None,
    ):
        wandb_settings = {
            "project": project,
            "group": group,
            "job_type": job_type,
            "config": config,
            "mode": mode,
        }
        self.processes = {}
        self.queues = {}
        for i in range(num_seeds):
            q = mp.Queue()
            self.queues[i] = q
            wandb_settings.update({"name": f"{seed}_{i}", "queue": q})
            # if i == 0 and save_code_path is not None:
            #     wandb_settings.update({"save_code_path": save_code_path})
            p = mp.Process(target=worker, kwargs=wandb_settings)
            p.start()
            self.processes[i] = p

    def log(self, seed, data_dict):
        self.queues[seed].put(data_dict)

    def finish(self):
        for seed in self.processes.keys():
            self.queues[seed].put(None)
        for seed in self.processes.keys():
            self.processes[seed].join()
