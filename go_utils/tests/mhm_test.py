import numpy as np
import pandas as pd
import pytest
from go_utils.mhm import (
    larvae_to_num,
    has_watersource_flag,
    has_genus_flag,
    is_container_flag,
    infectious_genus_flag,
    photo_bit_flags,
    completion_score_flag,
    qa_filter,
    apply_cleanup,
    add_flags,
)


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


@pytest.mark.mosquito
@pytest.mark.flagging
@pytest.mark.parametrize(
    "output_col, func",
    [("mhm_HasGenus", has_genus_flag), ("mhm_HasWaterSource", has_watersource_flag)],
)
def test_has_flags(output_col, func):
    df = pd.DataFrame.from_dict(
        {
            "col_of_interest": [
                np.nan,
                "pot",
                "container",
                "lake",
                None,
            ]
        }
    )
    func(df, "col_of_interest")
    desired = [0, 1, 1, 1, 0]

    assert np.all(desired == df[output_col])


@pytest.mark.mosquito
@pytest.mark.flagging
def test_infectious_genus():
    df = pd.DataFrame.from_dict(
        {
            "genus": [
                "Aedes",
                "Anopheles",
                "test",
                "Culex",
                "test",
            ]
        }
    )

    infectious_genus_flag(df, "genus")
    assert np.all(df["mhm_IsGenusOfInterest"] == [1, 1, 0, 1, 0])


@pytest.mark.mosquito
@pytest.mark.flagging
def test_is_container():
    df = pd.DataFrame.from_dict(
        {
            "watersource": [
                "container",
                "pot",
                "lake",
                "swamp",
                "tire",
                "ovitrap",
                "pond or estuary",
                "test or ocean",
            ]
        }
    )

    is_container_flag(df, "watersource")
    assert np.all(df["mhm_IsWaterSourceContainer"] == [1, 1, 0, 0, 1, 1, 0, 0])


@pytest.mark.mosquito
@pytest.mark.flagging
def test_photo_bit():
    df = pd.DataFrame.from_dict(
        {
            "abdomen": [
                "https://test;rejected;https://test",
                "pending;rejected",
                np.nan,
                "rejected",
                "pending",
            ],
            "larvae": [
                "rejected",
                "https://test",
                "rejected;https://test",
                "https://test",
                "pending",
            ],
            "watersource": [
                np.nan,
                "https://test;https://test;https://test",
                "pending;rejected;pending",
                "rejected;pending;rejected",
                np.nan,
            ],
        }
    )
    photo_bit_flags(df, "watersource", "larvae", "abdomen")
    assert np.all(df["mhm_PhotoCount"] == [2, 4, 1, 1, 0])
    assert np.all(df["mhm_RejectedCount"] == [2, 1, 2, 3, 0])
    assert np.all(df["mhm_PendingCount"] == [0, 1, 2, 1, 2])
    assert np.all(df["mhm_PhotoBitBinary"] == ["001", "110", "010", "010", "000"])
    assert np.all(df["mhm_PhotoBitDecimal"] == [1, 6, 2, 2, 0])


@pytest.mark.mosquito
@pytest.mark.flagging
def test_completeness():
    df = pd.DataFrame.from_dict(
        {
            "abdomen": ["https://test", "pending", np.nan, "rejected", "pending"],
            "larvae": [
                "rejected",
                "https://test",
                "rejected",
                "https://test",
                "pending",
            ],
            "watersource": [
                np.nan,
                "https://test",
                "pending;rejected;pending",
                "rejected;pending;rejected",
                np.nan,
            ],
            "genus": [np.nan, "test", np.nan, "test", "test"],
            "filler": ["test", np.nan, "test", "test", np.nan],
        }
    )

    has_genus_flag(df, "genus")
    photo_bit_flags(df, "watersource", "larvae", "abdomen")
    completion_score_flag(df)
    assert np.all(df["mhm_SubCompletenessScore"] == [0.25, 0.75, 0.0, 0.5, 0.25])
    assert np.all(
        df["mhm_CumulativeCompletenessScore"] == [0.82, 0.91, 0.82, 1.00, 0.82]
    )


@pytest.mark.mosquito
@pytest.mark.util
def test_qa_filter():
    mhm_df = pd.read_csv("go_utils/tests/sample_data/mhm.csv")
    mhm_df = apply_cleanup(mhm_df)
    add_flags(mhm_df)

    # Make sure default changes nothing
    assert len(mhm_df) == len(qa_filter(mhm_df))

    genus_filtered = qa_filter(mhm_df, has_genus=True)
    assert np.all(genus_filtered["mhm_HasGenus"] == 1)

    larvae_filtered = qa_filter(mhm_df, min_larvae_count=1)
    assert np.all(larvae_filtered["mhm_LarvaeCount"] >= 1)

    photo_filtered = qa_filter(mhm_df, has_photos=True)
    assert np.all(photo_filtered["mhm_PhotoBitDecimal"] > 0)

    container_filtered = qa_filter(mhm_df, is_container=True)
    assert np.all(container_filtered["mhm_IsWaterSourceContainer"] == 1)
