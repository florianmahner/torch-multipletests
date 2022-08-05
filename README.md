# `torch_multipletests`

Pytroch implementation of `statsmodels.stats.multitests.multipletests` to control for False-Discovery-Rates and correct P-Values on GPU for accelerated training and evaluation.

The current implementation supports correction using Bonferroni and Benjamin Hochberg.

## Example Usage:

```python
import torch
from torch_multipletests.multitest import multipletest

alpha = 0.05
method = 'bonferroni' # bonferroni correction 

# Create synthetic P-Values following the CDF of a Gaussian. Replace these with your own
loc, scale  = torch.randn(100), torch.randn(100).exp()
cut_off = torch.as_tensor(0)
pvals = torch.distributions.Normal(loc, scale).cdf(cut_off)

fdr_reject, pvals_corrected, alpha_bonferroni_correction = multipletest(pvals, alpha=alpha)
```