import numpy as np
import pandas as pd
import pytest

from go_utils.lc import (  # isort: skip
    add_flags,
    apply_cleanup,
    classification_bit_flags,
    completion_scores,
    extract_classification_name,
    extract_classification_percentage,
    get_main_classifications,
    photo_bit_flags,
    qa_filter,
    unpack_classifications,
)

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


sample_data_1 = "60% MUC 02 (b) [Category one]; 50% MUC 05 (b) [Category two]"
sample_data_2 = "33% MUC 02 (b) [Category one]; 25% MUC 05 (b) [Category two]"
sample_data = [sample_data_1, sample_data_2]


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
        {classifications[i]: [sample_data[i % 2]] for i in range(len(classifications))}
    )
    df["lc_pid"] = 0
    df = unpack_classifications(df)
    for i in range(len(classifications)):
        column_name = classifications[i].replace("Classifications", "_")
        if i % 2 == 0:
            one, two = 60, 50
        else:
            one, two = 33, 25
        assert df.loc[0, f"{column_name}CategoryOne"] == one
        assert df.loc[0, f"{column_name}CategoryTwo"] == two

    assert df.loc[0, "lc_Overall_CategoryOne"] == 46.5
    assert df.loc[0, "lc_Overall_CategoryTwo"] == 37.5


@pytest.mark.landcover
@pytest.mark.flagging
def test_photo_bit_flags():
    df = pd.DataFrame.from_dict(
        {
            "lc_UpwardPhotoUrl": [
                "https://test",
                "pending",
                np.nan,
                "rejected",
                "pending",
            ],
            "lc_DownwardPhotoUrl": [
                "rejected",
                "https://test",
                "rejected",
                "https://test",
                "pending",
            ],
            "lc_NorthPhotoUrl": [np.nan, "https://test", "pending", "rejected", np.nan],
            "lc_EastPhotoUrl": [
                "https://test",
                np.nan,
                "pending",
                "rejected",
                "pending",
            ],
            "lc_SouthPhotoUrl": [
                np.nan,
                "https://test",
                "rejected",
                "pending",
                "https://test",
            ],
            "lc_WestPhotoUrl": [
                "https://test",
                "https://test",
                "pending",
                "rejected",
                np.nan,
            ],
        }
    )

    output_df = photo_bit_flags(df)

    assert np.all(output_df["lc_PhotoCount"] == [3, 4, 0, 1, 1])
    assert np.all(output_df["lc_RejectedCount"] == [1, 0, 2, 4, 0])
    assert np.all(output_df["lc_PendingCount"] == [0, 1, 3, 1, 3])
    assert np.all(output_df["lc_EmptyCount"] == [2, 1, 1, 0, 2])
    assert np.all(
        output_df["lc_PhotoBitBinary"]
        == ["100011", "011101", "000000", "010000", "000100"]
    )
    assert np.all(output_df["lc_PhotoBitDecimal"] == [35, 29, 0, 16, 4])

    assert not output_df.equals(df)
    photo_bit_flags(df, inplace=True)
    assert output_df.equals(df)


@pytest.mark.landcover
@pytest.mark.flagging
def test_classification_flags():
    df = pd.DataFrame.from_dict(
        {
            "lc_NorthClassifications": ["test", np.nan, "test", np.nan],
            "lc_EastClassifications": [np.nan, np.nan, "test", "test"],
            "lc_SouthClassifications": [np.nan, "test", "test", np.nan],
            "lc_WestClassifications": ["test", np.nan, "test", np.nan],
        }
    )

    output_df = classification_bit_flags(df)

    assert np.all(output_df["lc_ClassificationCount"] == [2, 1, 4, 1])
    assert np.all(
        output_df["lc_ClassificationBitBinary"] == ["1001", "0100", "1111", "0010"]
    )
    assert np.all(output_df["lc_ClassificationBitDecimal"] == [9, 4, 15, 2])

    assert not output_df.equals(df)
    classification_bit_flags(df, inplace=True)
    assert output_df.equals(df)


@pytest.mark.landcover
@pytest.mark.flagging
def test_main_classification():
    df = pd.DataFrame.from_dict(
        {
            "lc_NorthClassifications": [
                "25% [1]; 75% [2]",
                np.nan,
                "25% [1]; 25% [2]; 80% [3]",
                np.nan,
            ],
            "lc_EastClassifications": [
                "25% [1]; 35% [4]",
                "100% [2]",
                "10% [1]; 25% [3]; 60% [4]",
                "10% [2]; 25% [3]; 60% [4]",
            ],
            "lc_SouthClassifications": [
                "75% [2]",
                "100% [1]",
                "25% [1]; 60% [4]",
                "40% [1]; 60% [2]",
            ],
            "lc_WestClassifications": [
                "10% [1]; 25% [3]; 60% [4]",
                np.nan,
                "25% [1]; 75% [2]",
                "40% [2]; 40% [1]; 10% [3]; 10% [4]",
            ],
        }
    )

    desired_dict = {
        "lc_NorthPrimary": ["2", "NA", "3", "NA"],
        "lc_NorthSecondary": ["1", "NA", "1, 2", "NA"],
        "lc_EastPrimary": ["4", "2", "4", "4"],
        "lc_EastSecondary": ["1", "NA", "3", "3"],
        "lc_SouthPrimary": ["2", "1", "4", "2"],
        "lc_SouthSecondary": ["NA", "NA", "1", "1"],
        "lc_WestPrimary": ["4", "NA", "2", "2, 1"],
        "lc_WestSecondary": ["3", "NA", "1", "3, 4"],
        "lc_PrimaryClassification": ["2", "2, 1", "4", "2"],
        "lc_SecondaryClassification": ["4", "NA", "3", "1"],
        "lc_PrimaryPercentage": [37.5, 25, 30, 27.5],
        "lc_SecondaryPercentage": [23.75, 0, 26.25, 20],
    }

    output_df = get_main_classifications(df)

    for column, desired in desired_dict.items():
        assert np.all(output_df[column] == desired)

    assert not output_df.equals(df)
    get_main_classifications(df, inplace=True)
    assert output_df.equals(df)


@pytest.mark.landcover
@pytest.mark.flagging
def test_completion_scores():
    df = pd.DataFrame.from_dict(
        {
            "lc_UpwardPhotoUrl": ["https://test", "pending", np.nan, "rejected"],
            "lc_DownwardPhotoUrl": [
                "rejected",
                "https://test",
                "rejected",
                "https://test",
            ],
            "lc_NorthPhotoUrl": [np.nan, "https://test", "pending", "rejected"],
            "lc_EastPhotoUrl": [
                "https://test",
                np.nan,
                "pending",
                "rejected",
            ],
            "lc_SouthPhotoUrl": [
                np.nan,
                "https://test",
                "rejected",
                "pending",
            ],
            "lc_WestPhotoUrl": ["https://test", "https://test", "pending", "rejected"],
            "lc_NorthClassifications": ["test", np.nan, "test", np.nan],
            "lc_EastClassifications": [np.nan, np.nan, "test", "test"],
            "lc_SouthClassifications": [np.nan, "test", "test", np.nan],
            "lc_WestClassifications": ["test", np.nan, "test", np.nan],
            "extra": ["a", np.nan, "b", np.nan],
        }
    )

    photo_bit_flags(df, inplace=True)
    classification_bit_flags(df, inplace=True)
    output_df = completion_scores(df)

    assert np.all(output_df["lc_SubCompletenessScore"] == [0.5, 0.5, 0.4, 0.2])
    assert np.all(
        output_df["lc_CumulativeCompletenessScore"] == [0.80, 0.75, 0.95, 0.80]
    )

    assert not output_df.equals(df)
    completion_scores(df, inplace=True)
    assert output_df.equals(df)


@pytest.mark.landcover
@pytest.mark.util
def test_qa_filter():
    lc_df = pd.read_csv("go_utils/tests/sample_data/lc.csv")
    lc_df = apply_cleanup(lc_df)
    lc_df = add_flags(lc_df)

    # Make sure default changes nothing
    assert len(lc_df) == len(qa_filter(lc_df))

    single_classification_filtered = qa_filter(lc_df, has_classification=True)
    assert np.all(single_classification_filtered["lc_ClassificationBitDecimal"] > 0)
    single_photo_filtered = qa_filter(lc_df, has_photo=True)
    assert np.all(single_photo_filtered["lc_PhotoBitDecimal"] > 0)
    all_classifications_filtered = qa_filter(lc_df, has_all_classifications=True)
    assert np.all(all_classifications_filtered["lc_ClassificationBitDecimal"] == 15)
    all_photos_filtered = qa_filter(lc_df, has_all_photos=True)
    assert np.all(all_photos_filtered["lc_PhotoBitDecimal"] == 63)
