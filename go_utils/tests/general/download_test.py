import pytest
from go_utils import get_api_data


def test_mhm_lc_download():
    mhm_df = get_api_data("mosquito_habitat_mapper")
    assert not mhm_df.empty

    lc_df = get_api_data("land_covers")
    assert not lc_df.empty
