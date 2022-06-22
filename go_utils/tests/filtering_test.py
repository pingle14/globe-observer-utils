import numpy as np
import pandas as pd
import pytest

from go_utils.filtering import (
    filter_duplicates,
    filter_invalid_coords,
    filter_out_entries,
    filter_poor_geolocational_data,
)

filtering_df = pd.DataFrame.from_dict(
    {
        "index": [0, 1, 2, 3, 4, 5],
    }
)
filtering_test_data = [
    ([True, True, False, False, True, False], True, [0, 1, 4]),
    ([False, False, False, False, True, True], True, [4, 5]),
    ([True, True, False, False, True, False], False, [2, 3, 5]),
    ([False, False, False, False, False, False], True, []),
    ([True, True, True, True, True, True], False, []),
]


@pytest.mark.util
@pytest.mark.filtering
@pytest.mark.parametrize("mask, include, desired_indexes", filtering_test_data)
def test_filtering_util(mask, include, desired_indexes):
    np_mask = np.array(mask)

    df = filtering_df.copy()
    filtered_df = filter_out_entries(df, np_mask, include, False)
    assert len(filtered_df) == len(desired_indexes)
    for index in filtered_df.index.values.tolist():
        assert index in desired_indexes

    assert not filtered_df.equals(df)
    filter_out_entries(df, np_mask, include, True)
    assert filtered_df.equals(df)


@pytest.mark.util
@pytest.mark.filtering
def test_duplicate_filter():
    df = pd.DataFrame.from_dict(
        {
            "Latitude": [5, 5, 7, 8],
            "Longitude": [6, 6, 10, 2],
            "attribute1": ["foo", "foo", "foo", "bar"],
            "attribute2": ["baz", "baz", "baz", "baz"],
        }
    )

    filtered_df = filter_duplicates(df, ["Latitude", "Longitude", "attribute1"], 2)

    assert not np.any(
        (filtered_df["Latitude"] == 5)
        & (filtered_df["Longitude"] == 6)
        & (filtered_df["attribute1"] == "foo")
    )

    filtered_df = filter_duplicates(df, ["attribute1", "attribute2"], 3)
    assert not np.any(
        (filtered_df["attribute1"] == "foo") & (filtered_df["attribute2"] == "baz")
    )

    assert not filtered_df.equals(df)
    filter_duplicates(df, ["attribute1", "attribute2"], 3, True)
    print(filtered_df)
    print(df)
    assert filtered_df.equals(df)


@pytest.mark.util
@pytest.mark.filtering
def test_poor_geolocational_data_filter():
    df = pd.DataFrame.from_dict(
        {
            "Latitude": [36.5, 37.8, 39.2, 30, 19.2],
            "Longitude": [95.2, 28.6, 15, 13.5, 30.8],
            "MGRSLatitude": [36.5, 37.9, 39.3, 30.2, 19.3],
            "MGRSLongitude": [95.2, 28.6, 15.5, 14, 30.2],
        }
    )
    filtered_df = filter_poor_geolocational_data(
        df, "Latitude", "Longitude", "MGRSLatitude", "MGRSLongitude"
    )

    assert not np.any(
        (filtered_df["Latitude"] == filtered_df["MGRSLatitude"])
        & (filtered_df["Longitude"] == filtered_df["MGRSLongitude"])
    )
    assert not np.any(filtered_df["Latitude"] == filtered_df["Latitude"].astype(int))
    assert not np.any(filtered_df["Longitude"] == filtered_df["Longitude"].astype(int))

    assert not filtered_df.equals(df)
    filter_poor_geolocational_data(
        df, "Latitude", "Longitude", "MGRSLatitude", "MGRSLongitude", True
    )
    assert filtered_df.equals(df)


latlon_data = [
    (
        {
            "lat": [-90, 90, 50, -9999, 0, 2, -10, 36.5, 89.999],
            "lon": [-180, 180, 179.99, -179.99, -9999, 90, -90, 35.6, -17.8],
        }
    ),
    (
        {
            "latitude": [-90, 90, -89.999, -23.26, -9999, 12.75, -10, 36.5, 89.999],
            "longitude": [-180, 22.2, -37.85, -179.99, 180, 90, -90, -9999, 179.99],
        }
    ),
]


@pytest.mark.util
@pytest.mark.filtering
@pytest.mark.parametrize("df_dict", latlon_data)
def test_latlon_filter(df_dict):
    df = pd.DataFrame.from_dict(df_dict)
    latitude, longitude = df.columns

    # Test exclusive filtering
    filtered_df = filter_invalid_coords(df, latitude, longitude)
    assert np.all(filtered_df[latitude] > -90)
    assert np.all(filtered_df[latitude] < 90)
    assert np.all(filtered_df[longitude] > -180)
    assert np.all(filtered_df[longitude] < 180)

    # Test inclusive filtering
    filtered_df = filter_invalid_coords(df, latitude, longitude, inclusive=True)
    assert np.all(filtered_df[latitude] >= -90)
    assert np.all(filtered_df[latitude] <= 90)
    assert np.all(filtered_df[longitude] >= -180)
    assert np.all(filtered_df[longitude] <= 180)

    # Test inplace
    assert not filtered_df.equals(df)
    filter_invalid_coords(df, latitude, longitude, inclusive=True, inplace=True)
    assert filtered_df.equals(df)
