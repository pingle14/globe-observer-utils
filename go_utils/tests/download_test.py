import pytest
import json
import logging
from go_utils import *

logging.basicConfig(level=logging.WARNING)

globe_protocols = ["mosquito_habitat_mapper", "land_covers"]


@pytest.mark.parametrize("protocol", globe_protocols)
def test_default_download(protocol):
    try:
        df = get_api_data(protocol)
        assert not df.empty
    except RuntimeError:
        logging.warning("Data Request failed, API may be down.")


def test_bad_api_call():
    with pytest.raises(RuntimeError, match="settings"):
        get_api_data("asdf")


def test_globe_down():
    globe_down_json = """{
    "type": "FeatureCollection",
    "name": "GLOBE-data",
    "features": [],
    "message": "failure: Connection refused"
    }"""

    with pytest.raises(RuntimeError, match="down"):
        parse_api_data(json.loads(globe_down_json), [])
