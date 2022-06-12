from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from go_utils.cleanup import (
    adjust_timezones,
    camel_case,
    filter_duplicates,
    filter_invalid_coords,
    filter_poor_geolocational_data,
    remove_homogenous_cols,
    rename_latlon_cols,
    replace_column_prefix,
    round_cols,
    standardize_null_vals,
)

camel_case_data = [
    ("abcd efg", [" "], "AbcdEfg"),
    ("abcd", [" "], "Abcd"),
    ("one two-three,four.five", [" ", ",", "-", "."], "OneTwoThreeFourFive"),
]


@pytest.mark.util
@pytest.mark.parametrize("input_text, delims, expected", camel_case_data)
def test_camel_case(input_text, delims, expected):
    assert camel_case(input_text, delims) == expected


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

time_zone_data = [
    (48.21, 16.36, 1),  # Vienna GMT+1
    (41.8781, -87.6298, -6),  # Chicago GMT-6
    (39.7392, -104.9903, -7),  # Denver GMT-7
    (40.7128, -74.0060, -5),  # NYC GMT-5
    (34.0522, -118.2437, -8),  # LA GMT-8
    (33.9249, 18.4241, 2),  # Cape Town GMT+2
    (3.1390, 101.6869, 7),  # Kuala Lumpur GMT=7
]


@pytest.mark.util
@pytest.mark.cleanup
@pytest.mark.parametrize("lat, lon, offset", time_zone_data)
def test_datetime_convert(lat, lon, offset):
    test_times = pd.to_datetime(
        np.array(
            [
                "2021-01-05T17:42:00",
                "2021-12-17T03:10:00",
                "2018-07-01T00:00:00",
                "2020-08-10T23:59:00",
                "2020-02-29T3:00:00",
                "2020-02-29T23:00:00",
            ]
        )
    )
    lat_col = np.full((len(test_times)), lat)
    lon_col = np.full((len(test_times)), lon)

    df = pd.DataFrame.from_dict(
        {"lat": lat_col, "lon": lon_col, "measuredAt": test_times}
    )

    print(type(df["measuredAt"].to_numpy()[0]))

    output_df = adjust_timezones(df, "measuredAt", "lat", "lon")

    for i, date in enumerate(test_times):
        output_df.loc[i, "measuredAt"] == date + timedelta(hours=offset)

    assert not output_df.equals(df)
    adjust_timezones(df, "measuredAt", "lat", "lon", inplace=True)
    assert output_df.equals(df)


@pytest.mark.util
@pytest.mark.cleanup
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


@pytest.mark.util
@pytest.mark.cleanup
def test_homogenous_cols():
    df = pd.DataFrame.from_dict({"col_1": [3, 2, 1], "col_2": [0, 0, 0]})
    output_df = remove_homogenous_cols(df)
    assert "col_2" not in output_df.columns
    assert "col_1" in output_df.columns

    assert not output_df.equals(df)
    remove_homogenous_cols(df, inplace=True)
    assert output_df.equals(df)


@pytest.mark.util
@pytest.mark.cleanup
def test_col_replace():
    df = pd.DataFrame.from_dict({"landcoversTest1": [1], "landcoversTest2": [1]})
    output_df = replace_column_prefix(df, "landcovers", "lc")
    assert "landcoversTest1" not in output_df.columns
    assert "landcoversTest2" not in output_df.columns
    assert "lc_Test1" in output_df.columns
    assert "lc_Test2" in output_df.columns

    assert not output_df.equals(df)
    replace_column_prefix(df, "landcovers", "lc", inplace=True)
    assert output_df.equals(df)


@pytest.mark.util
@pytest.mark.cleanup
def test_lat_lon_replace():
    df = pd.DataFrame.from_dict(
        {
            "latitude": [1],
            "longitude": [2],
            "testMeasurementLatitude": [3],
            "testMeasurementLongitude": [4],
        }
    )
    output_df = rename_latlon_cols(df)
    assert output_df.loc[0, "Latitude"] == 3
    assert output_df.loc[0, "Longitude"] == 4
    assert output_df.loc[0, "MGRSLatitude"] == 1
    assert output_df.loc[0, "MGRSLongitude"] == 2

    assert not output_df.equals(df)
    rename_latlon_cols(df, inplace=True)
    assert output_df.equals(df)


@pytest.mark.util
@pytest.mark.cleanup
def test_column_round():
    df = pd.DataFrame.from_dict(
        {
            "Latitude": [1.123456],
            "longitude": [2.123],
            "number": [3.212],
            "text": ["text"],
        }
    )
    output_df = round_cols(df)
    assert output_df.loc[0, "Latitude"] == 1.12346
    assert output_df.loc[0, "longitude"] == 2.123
    assert output_df.loc[0, "number"] == 3
    assert output_df.loc[0, "text"] == "text"

    assert not output_df.equals(df)
    round_cols(df, inplace=True)
    assert output_df.equals(df)


@pytest.mark.util
@pytest.mark.cleanup
def test_null_standardize():
    df = pd.DataFrame.from_dict(
        {"data": ["", "nan", "null", "NaN", None, "test", 5, np.nan]}
    )

    # Using "." to not overlap with null values (not recommended for practical use)
    output_df = standardize_null_vals(df, ".")
    desired = [".", ".", ".", ".", ".", "test", 5, "."]
    for i in range(len(desired)):
        assert output_df.loc[i, "data"] == desired[i]

    assert not output_df.equals(df)
    standardize_null_vals(df, ".", inplace=True)
    assert output_df.equals(df)


@pytest.mark.util
@pytest.mark.cleanup
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
    assert filtered_df.equals(df)


@pytest.mark.util
@pytest.mark.cleanup
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
