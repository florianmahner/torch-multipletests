import numpy as np
import torch
from scipy.stats import norm
from functools import partial
import torch.nn as nn

def _ecdf(x):
    '''no frills empirical cdf used in fdrcorrection
    '''
    nobs = len(x)
    return np.arange(1,nobs+1)/float(nobs)

def _ecdf_torch(x):
    '''no frills empirical cdf used in fdrcorrection (torch version)
    '''
    nobs = len(x)
    return torch.arange(1,nobs+1, dtype=torch.float64) /float(nobs)


def fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False):
    pvals = np.asarray(pvals)
    assert pvals.ndim == 1, "pvals must be 1-dimensional, that is of shape (n,)"

    if not is_sorted:
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = np.take(pvals, pvals_sortind)
    else:
        pvals_sorted = pvals  # alias


    if method in ['i', 'indep', 'p', 'poscorr']:
        ecdffactor = _ecdf(pvals_sorted)
    elif method in ['n', 'negcorr']:
        cm = np.sum(1./np.arange(1, len(pvals_sorted)+1))   #corrected this
        ecdffactor = _ecdf(pvals_sorted) / cm
##    elif method in ['n', 'negcorr']:
##        cm = np.sum(np.arange(len(pvals)))
##        ecdffactor = ecdf(pvals_sorted)/cm
    else:
        raise ValueError('only indep and negcorr implemented')


    print('np ecdf', ecdffactor[0:3])

    reject = pvals_sorted <= ecdffactor*alpha

    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
        reject[:rejectmax] = True

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    del pvals_corrected_raw
    pvals_corrected[pvals_corrected>1] = 1

    print('np pvals', pvals_corrected[0:3])

    if not is_sorted:
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        del pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[pvals_sortind] = reject
        return reject_, pvals_corrected_
    else:
        return reject, pvals_corrected



def fdr_correction_torch(pvals, alpha=0.05, is_sorted=False):
    
    assert pvals.ndim == 1, "pvals must be 1-dimensional, that is of shape (n,)"

    if not is_sorted:
        pvals_sortind = torch.argsort(pvals)
        pvals_sorted = torch.take(pvals, pvals_sortind)

    else:
        pvals_sorted = pvals

    ecdf_factor = _ecdf_torch(pvals_sorted)
    reject = pvals_sorted <= ecdf_factor * alpha

    print('torch ecdf', ecdf_factor[0:3].numpy())


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

    print('torch pvals', pvals_corrected[0:3].numpy())

    if not is_sorted:
        pvals_corrected_ = torch.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        del pvals_corrected
        reject_ = torch.empty_like(reject)
        reject_[pvals_sortind] = reject
        return reject_, pvals_corrected_
    else:
        return reject, pvals_corrected

def pval(W_loc, W_scale, j):
    # the cdf describes the probability that a random sample X of n objects at dimension j 
    # will be less than or equal to 0 in our case) for a given mean (mu) and standard deviation (sigma):
    return norm.cdf(0.0, W_loc[:, j], W_scale[:, j])

def compute_pvals(W_loc, W_scale):
    # Adapted from https://github.com/LukasMut/VICE/utils.py
    # Compute the probability for an embedding value x_{ij} <= 0,
    # given mu and sigma of the variational posterior q_{\theta}
        
    # we compute the cdf probabilities >0 for all dimensions
    fn = partial(pval, W_loc, W_scale)
    n_dim = W_loc.shape[1]
    range_dim = np.arange(n_dim)
    pvals = fn(range_dim)

    return pvals.T

class Pruning(nn.Module):
    def __init__(self, n_objects):
        super().__init__()
        self.register_buffer("cdf_loc", torch.Tensor([0]))
        self.register_buffer("ecdf_factor", self._ecdf_torch(n_objects))


    def pval_torch(self, q_mu, q_var, j):
        # the cdf describes the probability that a random sample X of n objects at dimension j 
        # will be less than or equal to 0 in our case) for a given mean (mu) and standard deviation (sigma):
        return torch.distributions.Normal(q_mu[:, j], q_var[:, j]).cdf(self.cdf_loc)

    def compute_pvals_torch(self, q_mu, q_var):
        # we compute the cdf probabilities >0 for all dimensions
        fn = partial(self.pval_torch, q_mu, q_var)
        n_dim = q_mu.shape[1]
        range_dim = torch.arange(n_dim)
        pvals = fn(range_dim)

        return pvals.T

        

    def adjust_pvals_mutliple_comparisons_torch(self, p_vals, alpha=0.05):
        def pval_rejection(p):
            return self.fdr_correction_torch(p, alpha=alpha)[0]

        fdr = torch.empty_like(p_vals)
        n_pvals = p_vals.shape[0]
        for i in range(n_pvals):
            fdr[i] = pval_rejection(p_vals[i])

        return fdr

    def get_importance_torch(self, rejections):
        importance = rejections.sum(dim=1)
        
        return importance

    def _ecdf_torch(self, nobs):
        '''no frills empirical cdf used in fdrcorrection (torch version)
        '''
        return torch.arange(1,nobs+1, dtype=torch.float64) /float(nobs)

    def fdr_correction_torch(self, pvals, alpha=0.05, is_sorted=False):
        """ pytorch implementation of fdr correction, adapted from scipy.stats.multipletests """

        import time
        start_time = time.time()
        
        assert pvals.ndim == 1, "pvals must be 1-dimensional, that is of shape (n,)"
        if not is_sorted:
            pvals_sortind = torch.argsort(pvals)
            pvals_sorted = torch.take(pvals, pvals_sortind)
        else:
            pvals_sorted = pvals

        
        reject = pvals_sorted <= self.ecdf_factor * alpha

        if reject.any():
            rejectmax = max(torch.nonzero(reject)[0])
            reject[:rejectmax] = True

        pvals_corrected_raw = pvals_sorted / self.ecdf_factor

        # same as np.minimum.accumulate
        # note torch.flip(a, dims=(0,)) is the same as a[::-1]
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
            print('torch fdr time', time.time() - start_time)
            return reject_, pvals_corrected_
        else:
            return reject, pvals_corrected

        
    def __call__(self, q_mu, q_var, alpha=0.05):
        import time
        start_time = time.time()
        pvals = self.compute_pvals_torch(q_mu, q_var)
        print('torch pvals time', time.time() - start_time)
        start_time = time.time()
        rejections = self.adjust_pvals_mutliple_comparisons_torch(pvals, alpha)
        print('torch rejections time', time.time() - start_time)
    
        start_time = time.time()
        importance = self.get_importance_torch(rejections)
        print('torch importance time', time.time() - start_time)

        return importance


if __name__ == '__main__':
    
    q_mu = np.loadtxt('/LOCAL/fmahner/DeepEmbeddings/weights_sslab12_40mio_gamma_099/params/pruned_q_mu_epoch_250.txt')
    q_var = np.loadtxt('/LOCAL/fmahner/DeepEmbeddings/weights_sslab12_40mio_gamma_099/params/pruned_q_var_epoch_250.txt')

    '''
    pvals = compute_pvals(q_mu, q_var)


    reject_np, pvals_corrected_np = fdrcorrection(pvals[0,:], alpha=0.05, is_sorted=True) 


    pvals = torch.from_numpy(pvals)
    reject, pvals_corrected = fdr_correction_torch(pvals[0,:], alpha=0.05, is_sorted=True)
    
    torch_corrected = pvals_corrected.numpy()

    pnp = pvals_corrected_np

    print((pvals_corrected_np == torch_corrected).sum())
    '''

    pruner = Pruning(q_mu.shape[0]).to(torch.device('cuda'))
    q_mu_torch = torch.from_numpy(q_mu).to(torch.device('cuda'))
    q_var_torch = torch.from_numpy(q_var).to(torch.device('cuda'))

    pruner(q_mu_torch, q_var_torch)





    

    






