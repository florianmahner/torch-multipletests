#!/usr/bin/env python3

""" Pytorch p-value correction for multiple hypothesis testing """

import torch

# TODO Maybe put this in __init__.py
__all__ = ['multipletests']

methods = ['bonferroni', 'fdr_bh', 'fdr_by'] # only works for these right now!


def _ecdf_torch(x):
    '''no frills empirical cdf used in fdrcorrection (torch version)
    '''
    nobs = len(x)
    return torch.arange(1,nobs+1, dtype=torch.float64) /float(nobs)


def _check_method(method):
    if method not in methods:
        raise ValueError('method must be one of %s' % method)


def multipletests(pvals, alpha=0.05, method='hs', is_sorted=False,
                  returnsorted=False):
    # TODO return_sorted not implemented yet!

    _check_method(method)

    reject, pvals_corrected = fdrcorrection_torch(pvals, alpha=alpha, is_sorted=is_sorted)
    
    return reject, pvals_corrected


def fdrcorrection_torch(pvals, alpha=0.05, is_sorted=False):
    assert pvals.ndim == 1, "pvals must be 1-dimensional, that is of shape (n,)"

    if not is_sorted:
        pvals_sortind = torch.argsort(pvals)
        pvals_sorted = torch.take(pvals, pvals_sortind)

    else:
        pvals_sorted = pvals

    ecdf_factor = _ecdf_torch(pvals_sorted)
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
    pvals_corrected[pvals_corrected>1] = 1

    if not is_sorted:
        pvals_corrected_ = torch.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        del pvals_corrected
        reject_ = torch.empty_like(reject)
        reject_[pvals_sortind] = reject
        return reject_, pvals_corrected_
    else:
        return reject, pvals_corrected

