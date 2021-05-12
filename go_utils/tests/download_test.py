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
        parse_api_data(json.loads(globe_down_json))


def test_globe_normal():

    sample_lc_json = """
    {
    "count": 1710,
    "message": "success",
    "results": [
        {
        "protocol": "land_covers",
        "measuredDate": "2010-10-05",
        "createDate": "2021-03-29T10:43:01",
        "updateDate": "2021-03-29T10:43:01",
        "publishDate": "2021-03-29T10:49:19",
        "organizationId": 326812,
        "organizationName": "Escuela de Educacion  Media N 8",
        "siteId": 132,
        "siteName": "VICENTE LOPEZ:LCS-01",
        "countryName": "Argentina",
        "countryCode": "ARG",
        "latitude": -34.552,
        "longitude": -58.483,
        "elevation": 42,
        "pid": 187086150,
        "data": {
            "landcoversDownwardPhotoUrl": null,
            "landcoversEastExtraData": null,
            "landcoversEastPhotoUrl": null,
            "landcoversMucCode": "M91",
            "landcoversUpwardPhotoUrl": null,
            "landcoversEastCaption": null,
            "landcoversWestClassifications": null,
            "landcoversNorthCaption": null,
            "landcoversNorthExtraData": null,
            "landcoversDataSource": "GLOBE Data Entry Site Definition",
            "landcoversDryGround": null,
            "landcoversSouthClassifications": null,
            "landcoversWestCaption": null,
            "landcoversNorthPhotoUrl": null,
            "landcoversUpwardCaption": null,
            "landcoversDownwardExtraData": null,
            "landcoversEastClassifications": null,
            "landcoversMucDetails": null,
            "landcoversMeasuredAt": "2010-10-05T00:00:00",
            "landcoversDownwardCaption": null,
            "landcoversSouthPhotoUrl": null,
            "landcoversMuddy": null,
            "landcoversWestPhotoUrl": null,
            "landcoversStandingWater": null,
            "landcoversLeavesOnTrees": null,
            "landcoversUserid": null,
            "landcoversSouthExtraData": null,
            "landcoversSouthCaption": null,
            "landcoversRainingSnowing": null,
            "landcoversUpwardExtraData": null,
            "landcoversWestExtraData": null,
            "landcoversLandCoverId": 12736,
            "landcoversMucDescription": "Urban, Residential",
            "landcoversSnowIce": null,
            "landcoversNorthClassifications": null,
            "landcoversFieldNotes": "Urban, Residential :: please replace with Land Cover Site Comments"
        }
        },
        {
        "protocol": "land_covers",
        "measuredDate": "2012-05-11",
        "createDate": "2021-03-29T10:43:12",
        "updateDate": "2021-03-29T10:43:12",
        "publishDate": "2021-03-29T10:49:19",
        "organizationId": 326053,
        "organizationName": "Domingo Faustino Sarmiento",
        "siteId": 133,
        "siteName": "Mision Baja:LCS-01",
        "countryName": "Argentina",
        "countryCode": "ARG",
        "latitude": -54.4892,
        "longitude": -68.18545,
        "elevation": -12.2,
        "pid": 187086632,
        "data": {
            "landcoversDownwardPhotoUrl": null,
            "landcoversEastExtraData": "null",
            "landcoversEastPhotoUrl": "https://data.globe.gov/system/photos/2012/11/03/4070/original.JPG",
            "landcoversMucCode": "M94",
            "landcoversUpwardPhotoUrl": null,
            "landcoversEastCaption": "",
            "landcoversWestClassifications": null,
            "landcoversNorthCaption": "",
            "landcoversNorthExtraData": "null",
            "landcoversDataSource": "GLOBE Data Entry Site Definition",
            "landcoversDryGround": null,
            "landcoversSouthClassifications": null,
            "landcoversWestCaption": "",
            "landcoversNorthPhotoUrl": "https://data.globe.gov/system/photos/2012/11/03/4073/original.JPG",
            "landcoversUpwardCaption": null,
            "landcoversDownwardExtraData": null,
            "landcoversEastClassifications": null,
            "landcoversMucDetails": null,
            "landcoversMeasuredAt": "2012-05-11T00:00:00",
            "landcoversDownwardCaption": null,
            "landcoversSouthPhotoUrl": "https://data.globe.gov/system/photos/2012/11/03/4075/original.JPG",
            "landcoversMuddy": null,
            "landcoversWestPhotoUrl": "https://data.globe.gov/system/photos/2012/11/03/4076/original.JPG",
            "landcoversStandingWater": null,
            "landcoversLeavesOnTrees": null,
            "landcoversUserid": null,
            "landcoversSouthExtraData": "null",
            "landcoversSouthCaption": "",
            "landcoversRainingSnowing": null,
            "landcoversUpwardExtraData": null,
            "landcoversWestExtraData": "null",
            "landcoversLandCoverId": 12737,
            "landcoversMucDescription": "Urban, Other",
            "landcoversSnowIce": null,
            "landcoversNorthClassifications": null,
            "landcoversFieldNotes": "Urban, Other :: SCRC data (please do not delete me!)   SCRC data (please do not delete me!)   SCRC data (please do not delete me!) el situo de estudio esta medianamente poblada, la superficie es de pasto muy cerca de la Bahia y con calles de tierra."
        }
        }
    ]
    }
    """
    df = parse_api_data(json.loads(sample_lc_json))
    assert not df.empty
