import numpy as np
import pandas as pd
import re

from go_utils.cleanup import replace_column_prefix


__doc__ = """

## Unpacking the Landcover Classification Data
The classification data for each entry is condensed into several entries separated by a semicolon. [This method](#unpack_classifications) identifies and parses Land Cover Classifications and percentages to create new columns. The columns are also reordered to better group directional information together.

The end result is a DataFrame that contains columns for every Unique Landcover Classification (per direction) and its respective percentages for each entry.

There are four main steps to this procedure:
1.Identifying Land Cover Classifications for each Cardinal Direction: An internal method returns the unique description (e.g. HerbaceousGrasslandTallGrass) listed in a column. This method is run for all 4 cardinal directions to obtain the all unique classifications per direction.
2. Creating empty columns for each Classification from each Cardinal Direction: Using the newly identified classifications new columns are made for each unique classification. These columns initially contained the default float64 value of 0.0. By initializing all the classification column values to 0.0, we ensure no empty values are set to -9999 in the round_cols(df) method (discussed in General Cleanup Procedures - Round Appropriate Columns). This step eases future numerical analysis.
3. Grouping and Alphabetically Sorting Directional Column Information: To better organize the DataFrame, columns containing any of the following directional substrings: "downward", "upward", "west", "east", "north", "south" (case insensitive) are identified and alphabetically sorted. Then an internal method called move_cols, specified column headers to move (direction_data_cols), and the location before the desired point of insertion, the program returns a reordered DataFrame, where all directional columns are grouped together. This greatly improves the Land Covers dataset’s organization and accessibility.
4. Adding Classification Percentages to their respective Land Cover Classification Columns - To fill in each classification column with their respective percentages, an internal method is applied to each row in the dataframe. This method iterates through each classification direction (ie “lc_EastClassifications”) and sets each identified Classification column with its respective percentage.

NOTE: After these procedures, the original directional classification columns (e.g. “lc_EastClassifications”) are not dropped.
"""

classifications = [
    "lc_WestClassifications",
    "lc_EastClassifications",
    "lc_NorthClassifications",
    "lc_SouthClassifications",
]


def cleanup_column_prefix(df):
    """Method for shortening raw landcover column names.

    Example usage:
    ```python
    from go_utils.lc import cleanup_column_prefix

    cleanup_column_prefix(df)
    ```

    The df object will now replace the verbose `landcovers` prefix in some of the columns with `lc_`

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing raw landcover data. The DataFrame object itself will be modified.
    """

    replace_column_prefix(df, "land_covers", "lc")


def camel_case(string, delimiters=[" "]):
    """Converts a string into camel case

    Parameters
    ----------
    string: str, the string to convert
    delimiter: str, the character that denotes separate words
    """
    for delimiter in delimiters:
        str_list = [s[0].upper() + s[1:] for s in string.split(delimiter)]
        string = "".join([s for s in str_list])
    return string


def extract_classification_name(entry):
    """
    Extracts the name (landcover description) of a singular landcover classification. For example in the classification of `"60% MUC 02 (b) [Trees, Closely Spaced, Deciduous - Broad Leaved]"`, the `"Trees, Closely Spaced, Deciduous - Broad Leaved"` is extracted.

    Parameters
    ----------
    entry : str
        A single landcover classification.

    Returns
    -------
    str
        The Landcover description of a classification
    """

    return re.search(r"(?<=\[).*(?=\])", entry).group()


def extract_classification_percentage(entry):
    """
    Extracts the percentage of a singular landcover classification. For example in the classification of `"60% MUC 02 (b) [Trees, Closely Spaced, Deciduous - Broad Leaved]"`, the `60` is extracted.

    Parameters
    ----------
    entry : str
        A single landcover classification.

    Returns
    -------
    float
        The percentage of a landcover classification
    """

    return float(re.search(".*(?=%)", entry).group())


def _extract_landcover_items(func, info):
    entries = info.split(";")
    return [func(entry) for entry in entries]


def extract_classifications(info):
    """Extracts the name/landcover description (see [here](#extract_classification_name) for a clearer definition) of a landcover classification entry in the GLOBE Observer Data.

    Parameters
    ----------
    info : str
        A string representing a landcover classification entry in the GLOBE Observer Datset.

    Returns
    -------
    list of str
        The different landcover classifications stored within the landcover entry.
    """
    return _extract_landcover_items(extract_classification_name, info)


def extract_percentages(info):
    """Extracts the percentages (see [here](#extract_classification_percentage) for a clearer definition) of a landcover classification in the GLOBE Observer Datset.

    Parameters
    ----------
    info : str
        A string representing a landcover classification entry in the GLOBE Observer Datset.

    Returns
    -------
    list of float
        The different landcover percentages stored within the landcover entry.
    """

    return _extract_landcover_items(extract_classification_percentage, info)


def extract_classification_dict(info):
    """Extracts the landcover descriptions and percentages of a landcover classification entry as a dictionary.

    Parameters
    ----------
    info : str
        A string representing a landcover classification entry in the GLOBE Observer Datset.

    Returns
    -------
    dict of str, float
        The landcover descriptions and percentages stored as a dict in the form: `{"description" : percentage}`.
    """

    entries = info.split(";")
    return {
        extract_classification_name(entry): extract_classification_percentage(entry)
        for entry in entries
    }


def _get_classifications_for_direction(df, direction_col_name):
    list_of_land_types = []
    for info in df[direction_col_name]:
        # Note: Sometimes info = np.nan, a float -- In that case we do NOT parse/split
        if type(info) == str:
            [
                list_of_land_types.append(camel_case(entry, [" ", ",", "-", "/"]))
                for entry in extract_classifications(info)
            ]
    return np.unique(list_of_land_types).tolist()


def _move_cols(df, cols_to_move=[], ref_col=""):
    col_names = df.columns.tolist()
    index_before_desired_loc = col_names.index(ref_col)

    cols_before_index = col_names[: index_before_desired_loc + 1]
    cols_at_index = cols_to_move

    cols_before_index = [i for i in cols_before_index if i not in cols_at_index]
    cols_after_index = [
        i for i in col_names if i not in cols_before_index + cols_at_index
    ]

    return df[cols_before_index + cols_at_index + cols_after_index]


def _set_directions(row):
    for classification in classifications:
        if not pd.isnull(row[classification]):
            entries = row[classification].split(";")
            for entry in entries:
                percent, name = (
                    extract_classification_percentage(entry),
                    extract_classification_name(entry),
                )
                name = camel_case(name, [" ", ",", "-", "/"])
                classification = classification.replace("Classifications", "_")
                row[f"{classification}{name.strip()}"] = percent
    return row


def unpack_classifications(lc_df):
    """
    Unpacks the classification data in the *raw* GLOBE Observer Landcover data. This method assumes that the columns have been renamed with accordance to the [column cleanup](#cleanup_column_prefix) method.

    See [here](#unpacking-the-landcover-classification-data) for more information.

    *Note:* The returned DataFrame will have around 250 columns.

    Parameters
    ----------
    lc_df : pd.DataFrame
        A DataFrame containing Raw GLOBE Observer Landcover data that has had the column names simplified.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the unpacked classification columns.
    """

    land_type_columns_to_add = {
        classification: _get_classifications_for_direction(lc_df, classification)
        for classification in classifications
    }
    for key, values in land_type_columns_to_add.items():
        key = key.replace("Classifications", "_")
        for value in values:
            lc_df[key + value] = 0.0
    direction_data_cols = sorted(
        [
            name
            for name in lc_df.columns
            if any(
                direction in name.lower()
                for direction in (
                    "downward",
                    "upward",
                    "west",
                    "east",
                    "north",
                    "south",
                )
            )
        ]
    )
    lc_df = _move_cols(lc_df, cols_to_move=direction_data_cols, ref_col="lc_pid")
    return lc_df.apply(_set_directions, axis=1)
