import numpy as np
import pandas as pd
import pytest

from go_utils.cleanup import (
    replace_column_prefix,
    remove_homogenous_cols,
    rename_latlon_cols,
    round_cols,
    standardize_null_vals,
)


def test_homogenous_cols():
    df = pd.DataFrame.from_dict({"col_1": [3, 2, 1], "col_2": [0, 0, 0]})
    remove_homogenous_cols(df)
    assert "col_2" not in df.columns
    assert "col_1" in df.columns


# Make sure the method supports both the official naming and naming without underscores
@pytest.mark.parametrize("protocol", ["land_covers", "landcovers"])
def test_col_replace(protocol):
    df = pd.DataFrame.from_dict({"landcoversTest1": [1], "landcoversTest2": [1]})
    replace_column_prefix(df, protocol, "lc")
    assert "landcoversTest1" not in df.columns
    assert "landcoversTest2" not in df.columns
    assert "lc_Test1" in df.columns
    assert "lc_Test2" in df.columns


def test_lat_lon_replace():
    df = pd.DataFrame.from_dict(
        {
            "latitude": [1],
            "longitude": [2],
            "testMeasurementLatitude": [3],
            "testMeasurementLongitude": [4],
        }
    )
    rename_latlon_cols(df)
    assert df.loc[0, "Latitude"] == 3
    assert df.loc[0, "Longitude"] == 4
    assert df.loc[0, "MGRSLatitude"] == 1
    assert df.loc[0, "MGRSLongitude"] == 2


def test_column_round():
    df = pd.DataFrame.from_dict(
        {
            "Latitude": [1.123456],
            "longitude": [2.123],
            "number": [3.212],
            "text": ["text"],
        }
    )
    round_cols(df)
    assert df.loc[0, "Latitude"] == 1.12346
    assert df.loc[0, "longitude"] == 2.123
    assert df.loc[0, "number"] == 3
    assert df.loc[0, "text"] == "text"


def test_null_standardize():
    df = pd.DataFrame.from_dict(
        {"data": ["", "nan", "null", "NaN", None, "test", 5, np.nan]}
    )

    # Using "." to not overlap with null values (not recommended for practical use)
    standardize_null_vals(df, ".")
    desired = [".", ".", ".", ".", ".", "test", 5, "."]
    for i in range(len(desired)):
        assert df.loc[i, "data"] == desired[i]
