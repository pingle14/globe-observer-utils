import matplotlib.pyplot as plt
import pandas as pd
import pytest

from go_utils import mhm, lc


@pytest.mark.plotting
@pytest.mark.parametrize("filename, module", [("mhm.csv", mhm), ("lc.csv", lc)])
def test_diagnostic_plots(filename, module):
    df = pd.read_csv(f"go_utils/tests/sample_data/{filename}")
    df = module.apply_cleanup(df)
    module.add_flags(df)
    module.diagnostic_plots(df)
    for num in plt.get_fignums():
        fig = plt.figure(num)
        assert fig.get_axes()
