from datetime import datetime
import numpy as np
import pandas as pd
import requests
import logging

from arcgis.features import GeoAccessor
from arcgis.gis import GIS
from go_utils.info import start_date, end_date, region_dict


def parse_api_data(response_json):
    try:
        results = response_json["results"]
        df = pd.DataFrame(results)
    except KeyError:
        raise RuntimeError("Data Download Failed. The GLOBE API is most likely down.")

    # Expand the 'data' column by listing the contents and passing as a new dataframe
    df = pd.concat([df, pd.DataFrame(list(df["data"]))], axis=1)
    # Drop the previously nested data column
    df = df.drop("data", 1)

    # Display the dataframe
    return df


def is_valid_latlon_box(latlon_box):

    valid_lat_checks = (
        latlon_box["min_lat"] < latlon_box["max_lat"]
        and latlon_box["max_lat"] <= 90
        and latlon_box["min_lat"] >= -90
    )
    valid_lon_checks = (
        latlon_box["min_lon"] < latlon_box["max_lon"]
        and latlon_box["max_lon"] <= 180
        and latlon_box["min_lon"] >= -180
    )

    return valid_lon_checks and valid_lat_checks


def get_api_data(
    protocol,
    start_date=start_date,
    end_date=end_date,
    latlon_box={"min_lat": -90, "max_lat": 90, "min_lon": -180, "max_lon": 180},
):
    """Utility function for interfacing with the GLOBE API.
    More information about the API can be viewed [here](https://www.globe.gov/es/globe-data/globe-api).

    Parameters
    ----------
    protocol : str
               The desired GLOBE Observer Protocol. Protocols for the App protocols include: `land_covers` (Landcover), `mosquito_habitat_mapper` (Mosquito Habitat Mapper), `sky_conditions` (Clouds), `tree_heights` (Trees).
    start_date : str, default= 2017-05-31
                 The desired start date of the dataset in the format of (YYYY-MM-DD).
    end_date : str, default= today's date in YYYY-MM-DD form.
               The desired end date of the dataset in the format of (YYYY-MM-DD).
    latlon_box : dict of {str, double}, optional
                 The longitudes and latitudes of a bounding box for the dataset. The minimum/maximum latitudes and longitudes must be specified with the following keys: "min_lat", "min_lon", "max_lat", "max_lon". The default value specifies all latitude and longitude coordinates.

    Returns
    -------
    pd.DataFrame
      A DataFrame containing Raw GLOBE Observer Data of the specified parameters
    """

    if is_valid_latlon_box(latlon_box):
        url = f"https://api.globe.gov/search/v1/measurement/protocol/measureddate/lat/lon/?protocols={protocol}&startdate={start_date}&enddate={end_date}&minlat={str(latlon_box['min_lat'])}&maxlat={str(latlon_box['max_lat'])}&minlon={str(latlon_box['min_lon'])}&maxlon={str(latlon_box['max_lon'])}&geojson=FALSE&sample=FALSE"
    else:
        logging.warning(
            "You did not enter any valid/specific coordinates, so we gave you all the observations for your protocol, date_range, and any countryNames you may have specified.\n"
        )
        url = f"https://api.globe.gov/search/v1/measurement/protocol/measureddate/?protocols={protocol}&startdate={start_date}&enddate={end_date}&geojson=FALSE&sample=FALSE"

    # Downloads data from the GLOBE API
    response = requests.get(url)

    if not response:
        raise RuntimeError(
            "Failed to get data from the API. Double check your specified settings to make sure they are valid."
        )

    # Convert measured date data into datetime
    df = parse_api_data(response.json())
    measured_at = protocol.replace("_", "") + "MeasuredAt"
    vectorized_convert_to_datetime = np.vectorize(_convert_to_datetime)
    if type(df.loc[0, "measuredDate"]) is str:
        df["measuredDate"] = vectorized_convert_to_datetime(
            df["measuredDate"].to_numpy()
        )
    if type(df.loc[0, measured_at]) is str:
        df[measured_at] = vectorized_convert_to_datetime(df[measured_at].to_numpy())
    return df


def _convert_to_datetime(date):
    try:
        return datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        try:
            return datetime.strptime(date, "%Y-%m-%d")
        except (ValueError, TypeError):
            pass
    return np.nan


def _get_valid_countries(df, country_list):
    country_filter = np.vectorize(lambda country_col: country_col in country_list)
    mask = country_filter(df["COUNTRY"].to_numpy())
    return df[mask]


def get_country_api_data(
    protocol,
    start_date=start_date,
    end_date=end_date,
    countries=[],
    regions=[],
    latlon_box={"min_lat": -90, "max_lat": 90, "min_lon": -180, "max_lon": 180},
):
    """
    Gets country enriched API Data. Due note that this data comes from layers in ArcGIS that are updated daily. Therefore, there will be some delay between when an entry is uploaded onto the GLOBE data base and being on the ArcGIS dataset.

    Parameters
    ----------
    protocol : str, {"mosquito_habitat_mapper", "land_covers"}
        The desired GLOBE Observer Protocol. Currently only mosquito habitat mapper and land cover is supported.
    start_date : str, default= 2017-05-31
        The desired start date of the dataset in the format of (YYYY-MM-DD).
    end_date : str, default= today's date in YYYY-MM-DD form.
        The desired end date of the dataset in the format of (YYYY-MM-DD).
    countries : list of str, default=[]
        The list of desired countries. Look at go_utils.info.region_dict to see supported country names. If the list is empty, all data will be included.
    regions : list of str, default=[]
        The list of desired regions. Look at go_utils.info.region_dict to see supported region names and the countries they enclose. If the list is empty, all data will be included.
    latlon_box : dict of {str, double}, optional
        The longitudes and latitudes of a bounding box for the dataset. The minimum/maximum latitudes and longitudes must be specified with the following keys: "min_lat", "min_lon", "max_lat", "max_lon". The default value specifies all latitude and longitude coordinates.
    """

    item_id_dict = {
        "mosquito_habitat_mapper": "02e3c448f42e4c35a2dd0c6cbbf42d85",
        "land_covers": "c68acbfc68db4409b495fd4636646aa6",
    }

    if protocol not in item_id_dict:
        raise ValueError(
            "Invalid protocol, currently only 'mosquito_habitat_mapper' and 'land_covers' are supported."
        )

    gis = GIS()
    item = gis.content.get(itemid=item_id_dict[protocol])
    df = GeoAccessor.from_layer(item.layers[0])
    df.rename({"latitude": "Latitude", "longitude": "Longitude"}, axis=1, inplace=True)

    # Filter the dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    measured_at = protocol.replace("_", "") + "MeasuredAt"

    vectorized_convert_to_datetime = np.vectorize(_convert_to_datetime)
    if type(df.loc[0, "measuredDate"]) is str:
        df["measuredDate"] = vectorized_convert_to_datetime(
            df["measuredDate"].to_numpy()
        )
    if type(df.loc[0, measured_at]) is str:
        df[measured_at] = vectorized_convert_to_datetime(df[measured_at].to_numpy())

    df = df[(df[measured_at] >= start) & (df[measured_at] <= end)]
    # Filter Latitude and longitudes
    if is_valid_latlon_box(latlon_box):
        df = df[
            (df["Latitude"] >= latlon_box["min_lat"])
            & (df["Longitude"] >= latlon_box["min_lon"])
            & (df["Longitude"] <= latlon_box["max_lon"])
            & (df["Latitude"] <= latlon_box["max_lat"])
        ]

    if countries:
        df = _get_valid_countries(df, countries)

    if regions:
        for region in regions:
            df = _get_valid_countries(df, region_dict[region])

    return df
