import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime

region_dict = {
    "Africa": [
        "Benin",
        "Botswana",
        "Burkina Faso",
        "Cameroon",
        "Cape Verde",
        "Chad",
        "Congo, the Republic of",
        "Ethiopia",
        "Gabon",
        "Gambia",
        "Ghana",
        "Guinea",
        "Kenya",
        "Liberia",
        "Madagascar",
        "Mali",
        "Mauritius",
        "Namibia",
        "Niger",
        "Nigeria",
        "Rwada",
        "Senegal",
        "Seychelles",
        "South Africa",
        "Tanzania",
        "Togo",
        "Uganda",
    ],
    "Asia and the Pacific": [
        "Australia",
        "Bangladesh",
        "Fiji",
        "India",
        "Japan",
        "Maldives",
        "Marshall Islands",
        "Micronesia",
        "Mongolia",
        "Nepal",
        "New Zealand",
        "Palau",
        "Philippines",
        "Republic of Korea",
        "Sri Lanka",
        "Taiwan Partnership",
        "Thailand",
        "Vietnam",
    ],
    "Latin America and Caribbean": [
        "Argentina",
        "Bahamas",
        "Burmuda",
        "Bolivia",
        "Brazil",
        "Chile",
        "Colombia",
        "Costa Rica",
        "Dominican Republic",
        "ECUADOR",
        "El Salvador",
        "Guatemala",
        "Honduras",
        "Mexico",
        "Panama",
        "Paraguay",
        "Peru",
        "Suriname",
        "Trinidad and Tobago",
        "Uruguay",
    ],
    "Europe and Eurasia": [
        "Austria",
        "Belgium",
        "Bulgaria",
        "Croatia",
        "Cyprus",
        "Czech Republic",
        "Denmark",
        "Estonia",
        "Finland",
        "France",
        "Georgia",
        "Germany",
        "Greece",
        "Hungary",
        "Iceland",
        "Ireland",
        "Israel",
        "Italy",
        "Kazakhstan",
        "Kyrgyz Republic",
        "Latvia",
        "Liechtenstein",
        "Lithuania",
        "Luxembourg",
        "Malta",
        "Moldovia",
        "Montenegro",
        "Netherlands",
        "North Macedonia",
        "Norway",
        "Poland",
        "Portugal",
        "Romania",
        "Russia",
        "Serbia, The Republic Of",
        "Slovak Republic",
        "Spain",
        "Sweden",
        "Switzerland",
        "Turkey",
        "Ukraine",
        "United Kingdom",
    ],
    "Near East and North Africa": [
        "Bahrain",
        "Egypt",
        "Jordan",
        "Kuwait",
        "Lebanon",
        "Mauritania",
        "Morocco",
        "Oman",
        "Pakistan",
        "Qatar",
        "Saudi Arabia",
        "Tunisia",
        "United Arab Emirates",
    ],
    "North America": ["Canada", "United States"],
}

start_date = "2017-05-31"
end_date = datetime.now().strftime("%Y-%m-%d")


def get_api_data(
    protocol,
    date_range=(start_date, end_date),
    country_names=[],
    latlon_box={"min_lat": -90, "max_lat": 90, "min_lon": -180, "max_lon": 180},
):
    """Utility function for interfacing with the GLOBE Observer API.

    Parameters
    ----------
    protocol : str
               The desired GLOBE Observer Protocol.
    date_range : tuple of str
                 The desired date range of the dataset with a start and end date in the format of (YYYY-MM-DD).
    country_names : list of str
                    The desired countries of the dataset. Note that this may not be accurate due to entries in the GLOBE Observer Dataset that lack country names.
    latlon_box : dict of {str: double}
                 The longitudes and latitudes of a bounding box for the dataset. The minimum/maximum latitudes and longitudes must be specified with the following keys: "min_lat", "min_lon", "max_lat", "max_lon"

    Returns
    -------
    pd.DataFrame
      A DataFrame containing Raw GLOBE Observer Data of the specified parameters
    """

    start_date = date_range[0]
    end_date = date_range[1]

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
    results = response.json()["results"]

    df = pd.DataFrame(results)

    # Expand the 'data' column by listing the contents and passing as a new dataframe
    df = pd.concat([df, pd.DataFrame(list(df["data"]))], axis=1)

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
