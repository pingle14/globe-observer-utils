import numpy as np
import pandas as pd
import pytest  # noqa: F401

from go_utils.mhm import larvae_to_num


@pytest.mark.mosquito
@pytest.mark.cleanup
def test_larvae_to_num():
    df = pd.DataFrame.from_dict(
        {
            "mhm_LarvaeCount": [
                np.nan,
                "more than 100",
                "25-50",
                "10",
                "200",
                "1000",
                "10000",
                "1e+27",
            ]
        }
    )
    desired = {
        "mhm_LarvaeCountMagnitude": [0, 1, 0, 0, 1, 2, 3, 4],
        "mhm_LarvaeCount": [-9999, 101, 25, 10, 101, 101, 101, 101],
        "mhm_LarvaeCountIsRangeFlag": [0, 1, 1, 0, 0, 0, 0, 0],
    }
    larvae_to_num(df)
    for key, desired_values in desired.items():
        for i in range(len(desired_values)):
            assert df.loc[i, key] == desired_values[i]
