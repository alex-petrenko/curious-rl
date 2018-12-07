import os
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import join

from algorithms.utils import summaries_dir, experiment_dir, experiments_dir, ensure_dir_exists
from utils.utils import log


sns.set()


COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'purple', 'pink',
          'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
          'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue', 'yellow']


class Experiment:
    def __init__(self, name, descr):
        self.name = name
        self.descr = descr


class Plot:
    def __init__(self, name, axis, descr):
        self.name = name
        self.axis = axis
        self.descr = descr


def running_mean(x, n):
    """Courtesy of https://stackoverflow.com/a/27681394/1645784"""
    cumsum = np.cumsum(np.insert(x, 0, 0))
    n += 1
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def main():
    """Script entry point."""
    stop_at = 80 * 1000 * 1000
    prefix = 'simple'

    # noinspection PyUnusedLocal
    experiments_very_sparse = [
        Experiment('doom_curious_vs_vanilla/doom_maze_very_sparse/doom_maze_very_sparse_pre_0.0', 'A2C (no curiosity)'),
        Experiment('doom_sweep_very_sparse/doom_sweep_i_0.5_p_0.05', 'A2C+ICM (curious)'),
    ]

    # noinspection PyUnusedLocal
    experiments_sparse = [
        Experiment('doom_curious_vs_vanilla/doom_maze_sparse/doom_maze_sparse_pre_0.0', 'A2C (no curiosity)'),
        Experiment('doom_curious_vs_vanilla/doom_maze_sparse/doom_maze_sparse_pre_0.05', 'A2C+ICM (curious)'),
    ]

    # noinspection PyUnusedLocal
    experiments_basic = [
        Experiment('doom_curious_vs_vanilla/doom_maze/doom_maze_pre_0.0', 'A2C (no curiosity)'),
        Experiment('doom_curious_vs_vanilla/doom_maze/doom_maze_pre_0.05', 'A2C+ICM (curious)'),
    ]

    experiments = [
        Experiment('doom_curious_vs_vanilla/doom_basic/doom_basic_pre_0.0', 'A2C (no curiosity)'),
        Experiment('doom_curious_vs_vanilla/doom_basic/doom_basic_pre_0.05', 'A2C+ICM (curious)'),
    ]

    plots = [
        Plot('a2c_aux_summary/avg_reward', 'average reward', 'Avg. reward for the last 1000 episodes'),
        Plot(
            'a2c_agent_summary/policy_entropy',
            'policy entropy, nats',
            'Stochastic policy entropy',
        ),
    ]

    for plot in plots:
        fig = plt.figure(figsize=(5, 4))
        fig.add_subplot()

        for ex_i, experiment in enumerate(experiments):
            experiment_name = experiment.name.split(os.sep)[-1]
            experiments_root = join(*(experiment.name.split(os.sep)[:-1]))
            exp_dir = experiment_dir(experiment_name, experiments_root)

            path_to_events_dir = summaries_dir(exp_dir)
            events_files = []
            for f in os.listdir(path_to_events_dir):
                if f.startswith('events'):
                    events_files.append(join(path_to_events_dir, f))

            if len(events_files) == 0:
                log.error('No events file for %s', experiment)
                continue

            events_files = sorted(events_files)
            steps, values = [], []

            for events_file in events_files:
                iterator = tf.train.summary_iterator(events_file)
                while True:
                    try:
                        e = next(iterator, None)
                    except Exception as exc:
                        log.warning(str(exc))
                        break

                    if e is None:
                        break

                    for v in e.summary.value:
                        if e.step >= stop_at:
                            break

                        if v.tag == plot.name:
                            steps.append(e.step)
                            values.append(v.simple_value)

            # just in case
            values = np.nan_to_num(values)

            smooth = 10
            values_smooth = running_mean(values, smooth)
            steps = steps[smooth:]
            values = values[smooth:]

            plt.plot(steps, values, color=COLORS[ex_i], alpha=0.2, label='__nolegend__')
            plt.plot(steps, values_smooth, color=COLORS[ex_i], label=experiment.descr, linewidth=2)

        plt.xlabel('environment steps')
        plt.ylabel(plot.axis)
        plt.title(plot.descr)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plots_dir = ensure_dir_exists(join(experiments_dir(), 'plots'))
        plot_name = plot.name.replace('/', '_')
        plt.savefig(join(plots_dir, f'{prefix}_{plot_name}.png'))
        plt.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
