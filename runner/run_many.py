"""Run many experiments, hyperparameter sweeps, etc."""
import subprocess
import sys
import time
from os.path import join

from algorithms.utils import experiment_dir
from utils.utils import log


def run_many(run_description):
    experiments = run_description.experiments
    max_parallel = run_description.max_parallel

    log.info('Starting processes with base cmds: %r', [e.cmd for e in experiments])
    log.info('Max parallel processes is %d', max_parallel)
    log.info('Monitor log files using tail -f train_dir/%s/**/**/log.txt', run_description.run_name)

    processes = []

    experiments = run_description.generate_experiments()
    next_experiment = next(experiments, None)

    while len(processes) > 0 or next_experiment is not None:
        while len(processes) < max_parallel and next_experiment is not None:
            cmd, name, root_dir = next_experiment
            log.info('Starting experiment "%s"', cmd)
            cmd_tokens = cmd.split(' ')

            logfile = open(join(experiment_dir(name, root_dir), 'log.txt'), 'wb')
            process = subprocess.Popen(cmd_tokens, stdout=logfile, stderr=logfile)
            process.process_logfile = logfile

            processes.append(process)

            next_experiment = next(experiments, None)

        remaining_processes = []
        for process in processes:
            if process.poll() is None:
                remaining_processes.append(process)
                continue
            else:
                process.process_logfile.close()

            log.info('Process %r finished with code %r', process, process.returncode)

        processes = remaining_processes
        time.sleep(0.1)

    log.info('Done!')

    return 0


def main():
    """Script entry point."""
    from runner.run_descriptions.runs.timer import DOOM_TIMER
    return run_many(DOOM_TIMER)


if __name__ == '__main__':
    sys.exit(main())
