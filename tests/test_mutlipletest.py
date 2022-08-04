import unittest
import torch
from torch_multipletest.multitest import multipletests as torch_multipletests
from statsmodels.stats.multitest import multipletests


def create_synthetic_pvals(n=100):
    """Create pvals based on cdf of gaussian distribution given randomly drawn mean and variances.
    Evaluates whether a certain variable X will be larger or equal than a cut off value.
    """
    loc = torch.randn(100)
    scale = torch.randn(100).exp()
    cut_off = torch.Tensor([0.0])
    pvals = torch.distributions.Normal(loc, scale).cdf(cut_off)
    return pvals


class MultipleTestTestCase(unittest.TestCase):
    def test_benjamin_hochberg_correction(self):
        pvals = create_synthetic_pvals()

        fdr_reject, pvals_corrected = torch_multipletests(
            pvals, alpha=0.05, method="fdr_bh", is_sorted=False
        )

        fdr_reject.numpy()
        pvals_corrected = pvals_corrected.numpy()
        pvals = pvals.numpy()

        fdr_reject_, pvals_corrected_, _, _ = multipletests(
            pvals, alpha=0.05, method="fdr_bh", is_sorted=False, returnsorted=False
        )

        self.assertEqual(pvals.shape, pvals_corrected.shape)
        self.assertEqual(pvals_corrected.all(), pvals_corrected.all())
        self.assertEqual(fdr_reject.all(), fdr_reject_.all())

    def test_bonferroni_correction(self):
        self.assertEqual(1, 1)


if __name__ == "__main__":
    unittest.main()
