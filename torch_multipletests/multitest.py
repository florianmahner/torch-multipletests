#!/usr/bin/env python3

""" Pytorch p-value correction for multiple hypothesis testing """

import torch

__all__ = ["multipletests"]

methods = ["bonferroni", "fdr_bh", "fdr_by"]  # only works for these right now!


def _ecdf_torch(x):
    """no frills empirical cdf used in fdrcorrection (torch version)"""
    nobs = len(x)
    return torch.arange(1, nobs + 1, dtype=torch.float64) / float(nobs)


def _check_method(method):
    if method not in methods:
        raise ValueError("method not recognized, must be one of %s" % method)


def multipletests(pvals, alpha=0.05, method="bonferroni", is_sorted=False):
    """
    Test results and p-value correction for multiple tests

    Ported from https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
    to Pytorch

    Parameters
    ----------
    pvals : array_like, 1-d
        uncorrected p-values.   Must be 1-dimensional.
    alpha : float
        FWER, family-wise error rate, e.g. 0.1
    method : str
        Method used for testing and adjustment of pvalues. Can be either the
        full name or initial letters. Available methods are:

        - `bonferroni` : one-step correction
        - `sidak` : one-step correction
        - `holm-sidak` : step down method using Sidak adjustments
        - `holm` : step-down method using Bonferroni adjustments
        - `simes-hochberg` : step-up method  (independent)
        - `hommel` : closed method based on Simes tests (non-negative)
        - `fdr_bh` : Benjamini/Hochberg  (non-negative)
        - `fdr_by` : Benjamini/Yekutieli (negative)
        - `fdr_tsbh` : two stage fdr correction (non-negative)
        - `fdr_tsbky` : two stage fdr correction (non-negative)

    is_sorted : bool
        If False (default), the p_values will be sorted, but the corrected
        pvalues are in the original order. If True, then it assumed that the
        pvalues are already sorted in ascending order.

    Returns
    -------
    reject : ndarray, boolean
        true for hypothesis that can be rejected for given alpha
    pvals_corrected : ndarray
        p-values corrected for multiple tests

    alphacBonf : float
        corrected alpha for Bonferroni method
    """

    _check_method(method)

    assert pvals.ndim == 1, "pvals must be 1-dimensional, that is of shape (n,)"

    pvals = torch.as_tensor(pvals)

    if not is_sorted:
        pvals_sortind = torch.argsort(pvals)
        pvals = torch.take(pvals, pvals_sortind)

    n_permutations = len(pvals)
    alpha_corrected_bonferroni = alpha / float(n_permutations)

    if method in ["bonferroni"]:
        reject = pvals <= alpha_corrected_bonferroni
        pvals_corrected = pvals * float(n_permutations)

    elif method in ["fdr_bh"]:
        reject, pvals_corrected = fdrcorrection_torch(
            pvals, alpha=alpha, method="indep", is_sorted=is_sorted
        )

    elif method in ["fdr_by"]:
        reject, pvals_corrected = fdrcorrection_torch(
            pvals, alpha=alpha, method="n", is_sorted=is_sorted
        )

    return reject, pvals_corrected, alpha_corrected_bonferroni


def fdrcorrection_torch(pvals, alpha=0.05, method="indep", is_sorted=False):
    """
    Benjamini/Hochberg (1995) False Discovery Rate (FDR) Correction procedure for multiple tests.
    pvals is a vector of p-values.
    alpha is the desired family-wise alpha level.
    method : {'i', 'indep', 'p', 'poscorr', 'n', 'negcorr'}, optional
        Which method to use for FDR correction.
        ``{'i', 'indep', 'p', 'poscorr'}`` all refer to ``fdr_bh``
        (Benjamini/Hochberg for independent or positively
        correlated tests). ``{'n', 'negcorr'}`` both refer to ``fdr_by``
        (Benjamini/Yekutieli for general or negatively correlated tests).
        Defaults to ``'indep'``.
    is_sorted : bool, optional
        If False (default), the p_values will be sorted, but the corrected
        pvalues are in the original order. If True, then it assumed that the
        pvalues are already sorted in ascending order.

    Ported from https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.fdrcorrection.html to Pytorch
    See there for full docs and explanations!
    """

    assert method in [
        "i",
        "indep",
        "n",
        "negcorr",
    ], "method must be one of 'i', 'indep', 'n', 'negcorr'"

    if not is_sorted:
        pvals_sortind = torch.argsort(pvals)
        pvals_sorted = torch.take(pvals, pvals_sortind)

    else:
        pvals_sorted = pvals

    if method in ["i", "indep"]:
        ecdf_factor = _ecdf_torch(pvals_sorted)

    elif method in ["n", "negcorr"]:
        cm = torch.sum(
            1.0 / torch.arange(1, len(pvals_sorted) + 1, dtype=torch.float64)
        )
        ecdf_factor = _ecdf_torch(pvals_sorted) / cm

    reject = pvals_sorted <= ecdf_factor * alpha

    if reject.any():
        rejectmax = max(torch.nonzero(reject)[0])
        reject[:rejectmax] = True

    pvals_corrected_raw = pvals_sorted / ecdf_factor

    # same as np.minimum.accumulate
    # note torch.flip(a) same as a[::-1]
    pvals_cummin, _ = torch.cummin(torch.flip(pvals_corrected_raw, dims=(0,)), dim=0)
    pvals_corrected = torch.flip(pvals_cummin, dims=(0,))
    del pvals_corrected_raw
    pvals_corrected[pvals_corrected > 1] = 1

    if not is_sorted:
        pvals_corrected_ = torch.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        del pvals_corrected
        reject_ = torch.empty_like(reject)
        reject_[pvals_sortind] = reject
        return reject_, pvals_corrected_

    else:
        return reject, pvals_corrected
