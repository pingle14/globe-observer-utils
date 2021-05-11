import pandas as pd
import numpy as np
import requests
import logging

from go_utils.info import *


def parse_api_data(response_json, country_names):
    try:
        results = response_json["results"]
        df = pd.DataFrame(results)

        # Expand the 'data' column by listing the contents and passing as a new dataframe
        df = pd.concat([df, pd.DataFrame(list(df["data"]))], axis=1)
    except KeyError:
        raise RuntimeError("Data Download Failed. The GLOBE API is most likely down.")

    # Drop the previously nested data column
    df = df.drop("data", 1)

    # Country Filters
    if country_names and len(country_names) > 0:
        country_filt = np.vectorize(lambda x: x in country_names)
        df = df[country_filt(df["countryName"].to_numpy())]
        df.reset_index(drop=True, inplace=True)
        if len(df) == 0:
            print(
                "WARNING: There are no observations from your country(s) with your given parameters."
            )

    # Display the dataframe
    return df


def get_api_data(
    protocol,
    start_date=start_date,
    end_date=end_date,
    country_names=[],
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
    country_names : list of str, default= []
                    The desired countries of the dataset. If left empty, the dataset is not filtered by countries. Note that this may not be accurate due to entries in the GLOBE Observer Dataset that lack country names.
    latlon_box : dict of {str, double}, optional
                 The longitudes and latitudes of a bounding box for the dataset. The minimum/maximum latitudes and longitudes must be specified with the following keys: "min_lat", "min_lon", "max_lat", "max_lon". The default value specifies all latitude and longitude coordinates.

    Returns
    -------
    pd.DataFrame
      A DataFrame containing Raw GLOBE Observer Data of the specified parameters
    """

    valid_lat_checks = (
        abs(latlon_box["min_lat"]) < abs(latlon_box["max_lat"])
        and abs(latlon_box["max_lat"]) <= 90
    )
    valid_lon_checks = (
        abs(latlon_box["min_lon"]) < abs(latlon_box["max_lon"])
        and abs(latlon_box["max_lon"]) <= 180
    )

    if valid_lat_checks and valid_lon_checks:
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

    return parse_api_data(response.json(), country_names)
