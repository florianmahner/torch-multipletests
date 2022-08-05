import unittest
import torch
from torch_multipletests.multitest import multipletests as torch_multipletests
from statsmodels.stats.multitest import multipletests


def create_synthetic_pvals(n=100):
    """Create pvals based on cdf of gaussian distribution given randomly drawn mean and variances.
    Evaluates whether a certain variable X will be larger or equal than a cut off value.
    """
    loc, scale = torch.randn(100), torch.randn(100).exp()
    cut_off = torch.as_tensor(0)
    pvals = torch.distributions.Normal(loc, scale).cdf(cut_off)

    return pvals


class MultipleTestTestCase(unittest.TestCase):
    def test_corrections(self):

        for method in ["bonferroni", "fdr_bh", "fdr_by"]:
            pvals = create_synthetic_pvals()

            fdr_reject, pvals_corrected, alphacBonferroni = torch_multipletests(
                pvals, alpha=0.05, method=method, is_sorted=False
            )

            fdr_reject = fdr_reject.numpy()
            pvals_corrected = pvals_corrected.numpy()
            pvals = pvals.numpy()
            fdr_reject_, pvals_corrected_, alphacBonferroni_, _ = multipletests(
                pvals, alpha=0.05, method=method, is_sorted=False
            )

            self.assertTrue(alphacBonferroni, float(alphacBonferroni_))
            self.assertEqual(pvals.shape, pvals_corrected.shape)
            self.assertEqual(pvals_corrected.all(), pvals_corrected_.all())
            self.assertEqual(fdr_reject.all(), fdr_reject_.all())


if __name__ == "__main__":
    unittest.main()
