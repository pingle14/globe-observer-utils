import pandas as pd
import numpy as np
import logging

__doc__ = """

# Overview
This submodule contains several methods to assist with data cleanup.
The following sections discuss some of the decisions behind these methods and their part of a larger data cleanup pipeline.

# Methods

## Remove Redundant/Homogenous Column:
[This method](#remove_homogenous_cols) indentifies and removes columns where all values were the same. If the logging level is set to INFO, the method will also print the names of the dropped columns and their respective singular value as a Python dictionary. 
For the raw mosquito habitat mapper data, the following were dropped: 
- {'protocol': 'mosquito_habitat_mapper'} 
- {'ExtraData': None} 
- {'MosquitoEggCount': None} 
- {'DataSource': 'GLOBE Observer App'}

For raw landcover data, the following were dropped:
- {'protocol': 'land_covers'}

## Rename columns:

### Differentiating between MGRS and GPS Columns: 
The GLOBE API data for `MosquitoHabitatMapper` and `LandCovers` report each observation’s Military Grid Reference System (MGRS) Coordinates in the `latitude` and `longitude` fields. The GPS Coordinates are stored in the `MeasurementLatitude` and `MeasurementLongitude` fields. 
To avoid confusion between these measuring systems, [this method](#rename_latlon_cols) renames `latitude` and `longitude` to `MGRSLatitude` and `MGRSLongitude`, respectively, and `MeasurementLatitude` and `MeasurementLongitude` to `Latitude` and `Longitude`, respectively. Now, the official `Latitude` and `Longitude` columns are more intuitively named.

### Protocol Abbreviation:
To better support future cross-protocol analysis and data enrichment, [this method](#replace_column_prefix) following naming scheme for all column names: `protocolAbbreviation_columnName`, where `protocolAbbreviation` was the abbreviation for the protocol (`mhm` for mosquito habitat mapper and `lc` for land cover) and `columnName` was the original name of the column. For example, the mosquito habitat mapper “MGRSLongitude” column was renamed “mhm_MGRSLongitude” and the corresponding land cover column was renamed “lc_MGRSLongitude”.

Do note that if you would like to use the previously mentioned `mhm` and `lc` naming scheme for you data, the `go_utils.mhm` and `go_utils.lc` submodules each have a method called `cleanup_column_prefix` which uses the previously mentioned naming scheme as opposed to the `replace_column_prefix` method which requires that you specify the current prefix and desired prefix.

## Standardize no-data values:
The GLOBE API CSV’s lacked standardization in indicating No Data. Indicators ranged from Python's `None`, to `“null”`, to an empty cell, to `NaN` (`np.nan`). To improve the computational efficiency in future mathematical algorithms on the GLOBE datasets, [this method](#standardize_null_values) converts all No Data Indicators to np.nan (Python NumPy’s version of No-Data as a float). Do note that later in Round Appropriate Columns, all numerical extraneous values are converted from np.nan to -9999.  Thus, Users will receive the pre-processed GLOBE API Mosquito Habitat Mapper and Land Cover Data in accordance with the standards described by Cook et al (2018). 

## Round Appropriate Columns
[This method](#round_cols) does the following:
1. Identifies all numerical columns (e.g. `float64`, `float`, `int`, `int64`).
2. Rounds Latitudes and Longitude Columns to 5 places. To reduce data density, all latitude and longitude values were rounded to 5 decimal places. This corresponds to about a meter of accuracy. Furthermore, any larger number of decimal places consume unnecessary amounts of storage as the GLOBE Observer app cannot attain such precision.
3. Converts other Numerical Data to Integers. To improve the datasets’ memory and performance, non latitude and longitude numerical values were converted to integers for the remaining columns, including `Id`, `MeasurementElevation`, and `elevation` columns.  This is appropriate since ids are always discrete values. `MeasurementElevation` and `elevation` are imprecise estimates from 3rd party sources, rendering additional precision an unnecessary waste of memory. However, by converting these values to integers, we could no longer use np.nan, a float, to denote extraneous/empty values. Thus, for integer columns, we used -9999 to denote extraneous/empty values.

**Note**: Larvae Counts were also converted to integers and Land Classification Column percentages were also converted to integers, reducing our data density. This logic is further discussed in go_utils.mhm.larvae_to_num for mosquito habitat mapper and go_utils.lc.unpack_classifications

"""


def remove_homogenous_cols(df):
    """
    Removes columns froma DataFrame if they contain only 1 unique value. This method will change the DataFrame that is passed.

    For example, if this code is run:
    ```python
    from go_utils.cleanup import remove_homogenous_cols
    remove_homogenous_cols(df)
    ```

    Then the original `df` variable that was passed is now updated with these dropped columns.

    If you would like to see the columns that are dropped, setting the logging level to info will allow for that to happen.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame that will be modified
    """

    for column in df.columns:
        if len(pd.unique(df[column])) == 1:
            logging.info(f"Dropped: {df[column][0]}")
            df.drop(column, axis=1, inplace=True)


def replace_column_prefix(df, protocol, replacement_text):
    """
    Replaces the protocol prefix (e.g. mosquito_habitat_mapper/mosquitohabitatmapper) for the column names with another prefix in the format of `newPrefix_columnName`.

    If you are interested in replacing the prefixes for the raw mosquito habitat mapper and landcover datasets, use the go_utils.lc.cleanup_column_prefix and go_utils.mhm.cleanup_column_prefix methods.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame you would like updated
    protocol : str
        A string representing the protocol prefix.
    replacement_text : str
        A string representing the desired prefix for the column name.
    """

    protocol = protocol.replace("_", "")
    df.columns = [
        f"{replacement_text}_{column.replace(protocol,'')}" for column in df.columns
    ]


def find_column(df, keyword):
    """Finds the first column that contains a certain keyword. Mainly intended to be a utility function for some of the other methods.


    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the columns that need to be searched.
    keyword: str
        The keyword that needs to be present in the desired column.
    """

    return [column for column in df.columns if keyword in column][0]


def rename_latlon_cols(df):
    """Renames the latitude and longitude columns of **raw** GLOBE Observer Data to make the naming intuitive.

    [This](#differentiating-between-mgrs-and-gps-columns) explains the motivation behind the method.

    Example usage:
    ```python
    from go_utils.cleanup import rename_latlon_cols
    rename_latlon_cols(df)
    ```

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose columns require renaming.
    """
    latitude_col = find_column(df, "MeasurementLatitude")
    longitude_col = find_column(df, "MeasurementLongitude")
    df.rename(
        {
            latitude_col: "Latitude",
            longitude_col: "Longitude",
            "latitude": "MGRSLatitude",
            "longitude": "MGRSLongitude",
        },
        axis=1,
        inplace=True,
    )


def round_cols(df):
    """This rounds columns in the DataFrame. More specifically, latitude and longitude data is rounded to 5 decimal places, other fields are rounded to integers, and null values (for the integer columns) are set to -9999.

    See [here](#round-appropriate-columns) for more information.

    Example usage:
    ```python
    from go_utils.cleanup import round_cols
    round_cols(df)
    ```

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame that requires rounding.
    """
    # Identifies all the numerical cols
    number_cols = [
        df.columns[i]
        for i in range(len(df.dtypes))
        if (df.dtypes[i] == "float64")
        or (df.dtypes[i] == "float")
        or (df.dtypes[i] == "int")
        or (df.dtypes[i] == "int64")
    ]

    # Rounds cols appropriately
    column_round = np.vectorize(lambda x, digits: round(x, digits))
    for name in number_cols:
        df[name] = df[name].fillna(-9999)
        if ("latitude" in name.lower()) or ("longitude" in name.lower()):
            logging.info(f"Rounded to 5 decimals: {name}")
            df[name] = column_round(df[name].to_numpy(), 5)
        else:
            logging.info(f"Converted to integer: {name}")
            df[name] = df[name].to_numpy().astype(int)


def standardize_null_vals(df, null_val=np.nan):
    """
    This method standardizes the null values of **raw** GLOBE Observer Data.

    ```python
    from go_utils.cleanup import standardize_null_vals
    standardize_null_vals(df)
    ```

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame that needs null value standardization
    null_val : obj, default=np.nan
        The value that all null values should be set to
    """

    # Replace Null Values with null_val
    df.fillna(null_val, inplace=True)

    # Replace any text null values
    df.replace(
        {
            "null": null_val,
            "": null_val,
            "NaN": null_val,
            "nan": null_val,
            None: null_val,
        },
        inplace=True,
    )
