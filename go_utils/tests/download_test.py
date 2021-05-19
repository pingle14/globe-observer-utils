import pytest
import json
import numpy as np
from go_utils.download import get_api_data, parse_api_data, get_country_api_data
from test_data import sample_lc_json, sample_mhm_json, globe_down_json

globe_test_data = [sample_lc_json, sample_mhm_json]
supported_protocols = ["land_covers", "mosquito_habitat_mapper"]


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
@pytest.mark.parametrize("protocol", supported_protocols)
def test_country_download(protocol):
    # Test regular method
    df = get_country_api_data(protocol)
    assert not df.empty

    # Test country based filtering
    df = get_country_api_data(protocol, countries=["United States"])
    assert not df.empty
    assert np.all(df["COUNTRY"] == "United States")

    # Test region based filtering
    df = get_country_api_data(protocol, regions=["North America"])
    assert not df.empty
    assert np.all((df["COUNTRY"] == "United States") | (df["COUNTRY"] == "Canada"))


@pytest.mark.downloadtest
@pytest.mark.parametrize("protocol", supported_protocols)
def test_api_download(protocol):
    df = get_api_data(protocol)
    assert not df.empty
