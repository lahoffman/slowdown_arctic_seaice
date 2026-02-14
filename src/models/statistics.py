"""
Statistical utilities for ML analysis (bootstrap, permutation tests, etc.).
"""

import numpy as np


def bootstrap_mean_ci(vals, B=1000, alpha=0.05, random_state=None):
    """
    Bootstrap confidence interval for the mean.

    Parameters
    ----------
    vals : array-like
        Values to bootstrap
    B : int, optional
        Number of bootstrap samples (default: 1000)
    alpha : float, optional
        Significance level (default: 0.05 for 95% CI)
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple
        (mean, lower_ci, upper_ci)
    """
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    n = vals.size

    if n == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(random_state)
    means = np.empty(B)

    for b in range(B):
        sample = rng.choice(vals, size=n, replace=True)
        means[b] = sample.mean()

    lower, upper = np.percentile(means, [100*alpha/2, 100*(1 - alpha/2)])
    return vals.mean(), lower, upper


def bootstrap_p_value_diff_means(vals1, vals2, B=5000, random_state=None):
    """
    Two-sided p-value for difference in means using bootstrap/permutation.

    Parameters
    ----------
    vals1, vals2 : array-like
        Two samples to compare
    B : int, optional
        Number of bootstrap samples (default: 5000)
    random_state : int, optional
        Random seed

    Returns
    -------
    float
        Two-sided p-value
    """
    vals1 = np.asarray(vals1)
    vals2 = np.asarray(vals2)

    vals1 = vals1[np.isfinite(vals1)]
    vals2 = vals2[np.isfinite(vals2)]

    n1, n2 = vals1.size, vals2.size
    if n1 == 0 or n2 == 0:
        return np.nan

    # Observed difference
    diff_obs = vals1.mean() - vals2.mean()

    # Null: both samples from same parent distribution
    pooled = np.concatenate([vals1, vals2])
    rng = np.random.default_rng(random_state)

    diffs_null = np.empty(B)
    for b in range(B):
        sample = rng.choice(pooled, size=pooled.size, replace=True)
        s1 = sample[:n1]
        s2 = sample[n1:]
        diffs_null[b] = s1.mean() - s2.mean()

    # Two-sided p-value
    p = np.mean(np.abs(diffs_null) >= np.abs(diff_obs))
    return p


def bootstrap_std(vals, B=1000, random_state=None):
    """
    Bootstrap estimate of standard deviation.

    Parameters
    ----------
    vals : array-like
        Values
    B : int, optional
        Number of bootstrap samples
    random_state : int, optional
        Random seed

    Returns
    -------
    float
        Bootstrap standard deviation estimate
    """
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    n = vals.size

    if n == 0:
        return np.nan

    rng = np.random.default_rng(random_state)
    stds = np.empty(B)

    for b in range(B):
        sample = rng.choice(vals, size=n, replace=True)
        stds[b] = sample.std()

    return stds.mean()


def permutation_test(vals1, vals2, n_permutations=10000, random_state=None):
    """
    Permutation test for difference in distributions.

    Parameters
    ----------
    vals1, vals2 : array-like
        Two samples to compare
    n_permutations : int, optional
        Number of permutations
    random_state : int, optional
        Random seed

    Returns
    -------
    tuple
        (statistic, p_value)
    """
    vals1 = np.asarray(vals1)
    vals2 = np.asarray(vals2)

    # Observed statistic (difference in means)
    stat_obs = np.mean(vals1) - np.mean(vals2)

    # Combined data
    combined = np.concatenate([vals1, vals2])
    n1 = len(vals1)

    # Permutation distribution
    rng = np.random.default_rng(random_state)
    perm_stats = np.empty(n_permutations)

    for i in range(n_permutations):
        perm = rng.permutation(combined)
        perm_stats[i] = np.mean(perm[:n1]) - np.mean(perm[n1:])

    # Two-sided p-value
    p_value = np.mean(np.abs(perm_stats) >= np.abs(stat_obs))

    return stat_obs, p_value
