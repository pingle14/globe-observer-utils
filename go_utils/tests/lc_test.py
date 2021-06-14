import numpy as np
import pandas as pd
import pytest

from go_utils.lc import (
    extract_classification_name,
    extract_classification_percentage,
    unpack_classifications,
    photo_bit_flags,
    classification_bit_flags,
    completion_scores,
    qa_filter,
    apply_cleanup,
    add_flags,
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
