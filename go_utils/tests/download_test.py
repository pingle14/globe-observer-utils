import json

import numpy as np
import pytest
from test_data import globe_down_json, sample_lc_json, sample_mhm_json

from go_utils.download import get_api_data, parse_api_data
from go_utils.geoenrich import get_country_api_data

globe_test_data = [sample_lc_json, sample_mhm_json]
protocol_prefixes = [("land_covers", "lc"), ("mosquito_habitat_mapper", "mhm")]


def assert_dates(df, prefix):
    first_index = df.index[0]
    date_cols = [col for col in df.columns if "MeasuredAt" in col or "Date" in col]
    for col in date_cols:
        assert type(df.loc[first_index, col]) is not str


@pytest.mark.util
def test_globe_down():
    with pytest.raises(RuntimeError, match="down"):
        parse_api_data(json.loads(globe_down_json))


@pytest.mark.util
@pytest.mark.parametrize("test_data", globe_test_data)
def test_globe_normal(test_data):
    df = parse_api_data(json.loads(sample_lc_json))
    assert not df.empty


@pytest.mark.downloadtest
def test_bad_api_call():
    with pytest.raises(RuntimeError, match="settings"):
        get_api_data("asdf")


@pytest.mark.downloadtest
@pytest.mark.geoenrich
@pytest.mark.parametrize("protocol, prefix", protocol_prefixes)
def test_country_download(protocol, prefix):
    # Test regular method
    df = get_country_api_data(protocol)
    assert not df.empty
    assert_dates(df, prefix)

    # Test country based filtering
    df = get_country_api_data(protocol, countries=["United States"])
    assert not df.empty
    assert np.all(df[f"{prefix}_COUNTRY"] == "United States")
    assert_dates(df, prefix)

    # Test region based filtering
    df = get_country_api_data(protocol, regions=["North America"])
    assert not df.empty
    assert np.all(
        (df[f"{prefix}_COUNTRY"] == "United States")
        | (df[f"{prefix}_COUNTRY"] == "Canada")
    )
    assert_dates(df, prefix)


@pytest.mark.downloadtest
@pytest.mark.parametrize("protocol, prefix", protocol_prefixes)
def test_api_download(protocol, prefix):
    df = get_api_data(protocol)
    assert not df.empty
    assert_dates(df, prefix)
