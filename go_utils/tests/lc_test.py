import pandas as pd
import pytest

from go_utils.lc import (
    camel_case,
    extract_classification_name,
    extract_classification_percentage,
    unpack_classifications,
)


camel_case_data = [
    ("abcd efg", [" "], "AbcdEfg"),
    ("abcd", [" "], "Abcd"),
    ("one two-three,four.five", [" ", ",", "-", "."], "OneTwoThreeFourFive"),
]


@pytest.mark.parametrize("input, delims, expected", camel_case_data)
def test_camel_case(input, delims, expected):
    assert camel_case(input, delims) == expected


test_classification = "60% MUC 02 (b) [Trees, Closely Spaced, Deciduous - Broad Leaved]"


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
