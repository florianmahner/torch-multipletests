![Python version](https://img.shields.io/badge/python%20-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)
![example workflow](https://github.com/florianmahner/torch_multipletests/actions/workflows/tests.yml/badge.svg)
![code style](https://img.shields.io/badge/code%20style-black-black)
<!-- [![codecov](https://codecov.io/gh/LukasMut/VICE/branch/main/graph/badge.svg?token=gntaL1yrXI)](https://codecov.io/gh/LukasMut/VICE) -->

# `torch_multipletests`

Pytroch implementation of `statsmodels.stats.multitests.multipletests` to control for False-Discovery-Rates and correct P-Values on GPU for accelerated training and evaluation.

The current implementation supports correction using Bonferroni and Benjamin Hochberg.

## Example Usage:

```python
import torch
from torch_multipletests.multitest import multipletest

alpha = 0.05
method = 'bonferroni' # bonferroni correction 

# create synthetic p-values following the cdf of a gaussian. replace these with your own
loc, scale  = torch.randn(100), torch.randn(100).exp()
cut_off = torch.as_tensor(0)
pvals = torch.distributions.Normal(loc, scale).cdf(cut_off)

fdr_reject, pvals_corrected, alpha_bonferroni_correction = multipletest(pvals, alpha)
```
