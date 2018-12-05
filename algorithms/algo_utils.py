import numpy as np


EPS = 1e-8


class RunningMeanStd(object):
    """
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    Courtesy of OpenAI Baselines.

    """
    def __init__(self, max_past_samples=None, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.max_past_samples = max_past_samples

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count, self.max_past_samples,
        )


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count, max_past_samples):
    """Courtesy of OpenAI Baselines."""
    if max_past_samples is not None:
        # pretend we never have more than n past samples, this will guarantee a constant convergence rate
        count = min(count, max_past_samples)

    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    m_2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = m_2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def extract_keys(list_of_dicts, *keys):
    """Turn a lists of dicts into a tuple of lists, with one entry for every given key."""
    res = []
    for k in keys:
        res.append([d[k] for d in list_of_dicts])
    return tuple(res)


def extract_key(list_of_dicts, key):
    return extract_keys(list_of_dicts, key)[0]
