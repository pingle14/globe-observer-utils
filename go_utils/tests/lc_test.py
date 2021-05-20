import numpy as np
import pandas as pd
import pytest

from go_utils.lc import (
    camel_case,
    extract_classification_name,
    extract_classification_percentage,
    unpack_classifications,
    photo_bit_flags,
    classification_bit_flags,
    completion_scores,
)


camel_case_data = [
    ("abcd efg", [" "], "AbcdEfg"),
    ("abcd", [" "], "Abcd"),
    ("one two-three,four.five", [" ", ",", "-", "."], "OneTwoThreeFourFive"),
]


@pytest.mark.landcover
@pytest.mark.util
@pytest.mark.parametrize("input, delims, expected", camel_case_data)
def test_camel_case(input, delims, expected):
    assert camel_case(input, delims) == expected


test_classification = "60% MUC 02 (b) [Trees, Closely Spaced, Deciduous - Broad Leaved]"


@pytest.mark.landcover
@pytest.mark.util
@pytest.mark.parametrize(
    "func, expected",
    [
        (
            extract_classification_name,
            "Trees, Closely Spaced, Deciduous - Broad Leaved",
        ),
        (extract_classification_percentage, 60.0),
    ],
)
def test_classification_extraction(func, expected):
    assert func(test_classification) == expected


sample_data = "60% MUC 02 (b) [Category one]; 50% MUC 05 (b) [Category two]"


@pytest.mark.landcover
@pytest.mark.cleanup
def test_landcover_unpack():
    classifications = [
        "lc_WestClassifications",
        "lc_EastClassifications",
        "lc_NorthClassifications",
        "lc_SouthClassifications",
    ]
    df = pd.DataFrame.from_dict(
        {classification: [sample_data] for classification in classifications}
    )
    df["lc_pid"] = 0
    df = unpack_classifications(df)
    for classification in classifications:
        column_name = classification.replace("Classifications", "_")
        assert df.loc[0, f"{column_name}CategoryOne"] == 60.0
        assert df.loc[0, f"{column_name}CategoryTwo"] == 50.0


@pytest.mark.landcover
@pytest.mark.flagging
def test_photo_bit_flags():
    df = pd.DataFrame.from_dict(
        {
            "up": ["https://test", "pending", np.nan, "rejected", "pending"],
            "down": ["rejected", "https://test", "rejected", "https://test", "pending"],
            "north": [np.nan, "https://test", "pending", "rejected", np.nan],
            "east": [
                "https://test",
                np.nan,
                "pending",
                "rejected",
                "pending",
            ],
            "south": [
                np.nan,
                "https://test",
                "rejected",
                "pending",
                "https://test",
            ],
            "west": ["https://test", "https://test", "pending", "rejected", np.nan],
        }
    )

    photo_bit_flags(df, "up", "down", "north", "south", "east", "west")

    assert np.all(df["lc_PhotoCount"] == [3, 4, 0, 1, 1])
    assert np.all(df["lc_RejectedCount"] == [1, 0, 2, 4, 0])
    assert np.all(df["lc_PendingCount"] == [0, 1, 3, 1, 3])
    assert np.all(df["lc_EmptyCount"] == [2, 1, 1, 0, 2])
    assert np.all(
        df["lc_PhotoBitBinary"] == ["100011", "011101", "000000", "010000", "000100"]
    )
    assert np.all(df["lc_PhotoBitDecimal"] == [35, 29, 0, 16, 4])


@pytest.mark.landcover
@pytest.mark.flagging
def test_classification_flags():
    df = pd.DataFrame.from_dict(
        {
            "north": ["test", np.nan, "test", np.nan],
            "east": [np.nan, np.nan, "test", "test"],
            "south": [np.nan, "test", "test", np.nan],
            "west": ["test", np.nan, "test", np.nan],
        }
    )

    classification_bit_flags(df, "north", "south", "east", "west")

    assert np.all(df["lc_ClassificationCount"] == [2, 1, 4, 1])
    assert np.all(df["lc_ClassificationBitBinary"] == ["1001", "0100", "1111", "0010"])
    assert np.all(df["lc_ClassificationBitDecimal"] == [9, 4, 15, 2])


@pytest.mark.landcover
@pytest.mark.flagging
def test_completion_scores():
    df = pd.DataFrame.from_dict(
        {
            "up": ["https://test", "pending", np.nan, "rejected"],
            "down": ["rejected", "https://test", "rejected", "https://test"],
            "north": [np.nan, "https://test", "pending", "rejected"],
            "east": [
                "https://test",
                np.nan,
                "pending",
                "rejected",
            ],
            "south": [
                np.nan,
                "https://test",
                "rejected",
                "pending",
            ],
            "west": ["https://test", "https://test", "pending", "rejected"],
            "north_classification": ["test", np.nan, "test", np.nan],
            "east_classification": [np.nan, np.nan, "test", "test"],
            "south_classification": [np.nan, "test", "test", np.nan],
            "west_classification": ["test", np.nan, "test", np.nan],
            "extra": ["a", np.nan, "b", np.nan],
        }
    )

    photo_bit_flags(df, "up", "down", "north", "south", "east", "west")
    classification_bit_flags(
        df,
        "north_classification",
        "south_classification",
        "east_classification",
        "west_classification",
    )
    completion_scores(df)

    assert np.all(df["lc_SubCompletenessScore"] == [0.5, 0.5, 0.4, 0.2])
    assert np.all(df["lc_CumulativeCompletenessScore"] == [0.80, 0.75, 0.95, 0.80])
