![Python version](https://img.shields.io/badge/python%20-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)
![example workflow](https://github.com/florianmahner/torch_multipletests/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/florianmahner/torch_multipletests/branch/main/graph/badge.svg?token=75FIYZG8BD)](https://codecov.io/gh/florianmahner/torch_multipletests)
![code style](https://img.shields.io/badge/code%20style-black-black)


# `torch_multipletests`

Simple Pytorch implementation of `statsmodels.stats.multitests.multipletests` to control for False-Discovery-Rates and correct p values on GPU for accelerated training and evaluation.

The functionality is currently limited compared to the original implementation. Right now the implementation only supports correcting for multiple comparions using Bonferroni (one-step), Benjamini/Hochberg (non-negative) and Benjamini/Yekutieli (negative) methods. Feel free to contribute.

## Installation

Execute the following lines to clone the repository and install the package using pip

```bash
git clone https://github.com/florianmahner/torch_multipletests.git
cd torch_multipletests
pip install -e .
```


## Example Usage:

```python
import torch
from torch_multipletests.multitest import multipletests

alpha = 0.05
method = 'bonferroni' # bonferroni correction 

# create synthetic p-values following the cdf of a gaussian. replace these with your own
loc, scale  = torch.randn(100), torch.randn(100).exp()
cut_off = torch.as_tensor(0)
pvals = torch.distributions.Normal(loc, scale).cdf(cut_off)

fdr_reject, pvals_corrected, alpha_bonferroni_correction = multipletest(pvals, alpha)
```
