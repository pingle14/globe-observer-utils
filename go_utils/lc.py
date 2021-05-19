import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns

from go_utils.cleanup import (
    replace_column_prefix,
    remove_homogenous_cols,
    rename_latlon_cols,
    standardize_null_vals,
    round_cols,
)
from go_utils.plot import completeness_histogram, plot_freq_bar, multiple_bar_graph


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


def photo_bit_flags(df, up, down, north, south, east, west):
    """
    Creates the following flags:
    - `lc_PhotoCount`: The number of valid photos per record.
    - `lc_RejectedCount`: The number of photos that were rejected per record.
    - `lc_PendingCount`: The number of photos that are pending approval per record.
    - `lc_PhotoBitBinary`: A string that represents the presence of a photo in the Up, Down, North, South, East, and West directions. For example, if the entry is `110100`, that indicates that there is a valid photo for the Up, Down, and South Directions but no valid photos for the North, East, and West Directions.
    - `lc_PhotoBitDecimal`: The numerical representation of the lc_PhotoBitBinary string.

    This modifies the DataFrame itself.

    Parameters
    ----------
    df : pd.DataFrame
        A mosquito habitat mapper DataFrame
    up : str
        The column name in the land cover DataFrame that contains the url for the upwards photo.
    down : str
        The column name in the land cover DataFrame that contains the url for the downwards photo.
    north : str
        The column name in the land cover DataFrame that contains the url for the north photo.
    south : str
        The column name in the land cover DataFrame that contains the url for the south photo.
    east : str
        The column name in the land cover DataFrame that contains the url for the east photo.
    west : str
        The column name in the land cover DataFrame that contains the url for the west photo.
    """

    def pic_data(*args):
        pic_count = 0
        rejected_count = 0
        pending_count = 0
        empty_count = 0
        valid_photo_bit_mask = ""

        for entry in args:
            if not pd.isna(entry) and "http" in entry:
                valid_photo_bit_mask += "1"
                pic_count += entry.count("http")
            else:
                valid_photo_bit_mask += "0"
            if pd.isna(entry):
                empty_count += 1
            else:
                pending_count += entry.count("pending")
                rejected_count += entry.count("rejected")
        return (
            pic_count,
            rejected_count,
            pending_count,
            empty_count,
            valid_photo_bit_mask,
            int(valid_photo_bit_mask, 2),
        )

    get_photo_data = np.vectorize(pic_data)
    (
        df["lc_PhotoCount"],
        df["lc_RejectedCount"],
        df["lc_PendingCount"],
        df["lc_EmptyCount"],
        df["lc_PhotoBitBinary"],
        df["lc_PhotoBitDecimal"],
    ) = get_photo_data(
        df[up].to_numpy(),
        df[down].to_numpy(),
        df[north].to_numpy(),
        df[south].to_numpy(),
        df[east].to_numpy(),
        df[west].to_numpy(),
    )


def classification_bit_flags(df, north, south, east, west):
    """
    Creates the following flags:
    - `lc_ClassificationCount`: The number of classifications per record.
    - `lc_BitBinary`: A string that represents the presence of a classification in the North, South, East, and West directions. For example, if the entry is `1101`, that indicates that there is a valid classification for the North, South, and West Directions but no valid classifications for the East Direction.
    - `lc_BitDecimal`: The number of photos that are pending approval per record.

    This modifies the DataFrame itself.

    Parameters
    ----------
    df : pd.DataFrame
        A mosquito habitat mapper DataFrame
    north : str
        The column name in the land cover DataFrame that contains the north classification.
    south : str
        The column name in the land cover DataFrame that contains the south classification.
    east : str
        The column name in the land cover DataFrame that contains the east classification.
    west : str
        The column name in the land cover DataFrame that contains the west classification.
    """

    def classification_data(*args):
        classification_count = 0
        classification_bit_mask = ""
        for entry in args:
            if pd.isna(entry) or entry is np.nan:
                classification_bit_mask += "0"
            else:
                classification_count += 1
                classification_bit_mask += "1"
        return (
            classification_count,
            classification_bit_mask,
            int(classification_bit_mask, 2),
        )

    get_classification_data = np.vectorize(classification_data)

    (
        df["lc_ClassificationCount"],
        df["lc_ClassificationBitBinary"],
        df["lc_ClassificationBitDecimal"],
    ) = get_classification_data(
        df[north],
        df[south],
        df[east],
        df[west],
    )


def completion_scores(df):
    """
    Adds the following completness score flags:
    - `lc_SubCompletenessScore`: The percentage of valid landcover classifications and photos that are filled out.
    - `lc_CumulativeCompletenessScore`: The percentage of non null values out of all the columns.

    This modifies the DataFrame itself.

    Parameters
    ----------
    df : pd.DataFrame
        A landcover DataFrame with the [`lc_PhotoBitBinary`](#photo_bit_flags) and [`lc_ClassificationBitBinary`](#classification_bit_flags) flags.
    """

    def sum_bit_mask(bit_mask="0"):
        sum = 0.0
        for char in bit_mask:
            sum += int(char)
        return sum

    scores = {}
    scores["sub_score"] = []
    # Cummulative Completion Score
    scores["cumulative_score"] = round(df.count(1) / len(df.columns), 2)
    # Sub-Score
    for index in range(len(df)):
        bit_mask = (
            df["lc_PhotoBitBinary"][index] + df["lc_ClassificationBitBinary"][index]
        )
        sub_score = round(sum_bit_mask(bit_mask=bit_mask), 2)
        sub_score /= len(bit_mask)
        scores["sub_score"].append(sub_score)

    df["lc_SubCompletenessScore"], df["lc_CumulativeCompletenessScore"] = (
        scores["sub_score"],
        scores["cumulative_score"],
    )


def apply_cleanup(lc_df):
    """Applies a full cleanup procedure to the landcover data.
    It follows the following steps:
    - Removes Homogenous Columns
    - Renames Latitude and Longitudes
    - Cleans the Column Naming
    - Unpacks landcover classifications
    - Rounds Columns
    - Standardizes Null Values

    Parameters
    ----------
    lc_df : pd.DataFrame
        A DataFrame containing **raw** Landcover Data from the API.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the cleaned Landcover Data
    """
    lc_df = lc_df.copy()
    remove_homogenous_cols(lc_df)
    rename_latlon_cols(lc_df)
    cleanup_column_prefix(lc_df)
    lc_df = unpack_classifications(lc_df)
    round_cols(lc_df)
    standardize_null_vals(lc_df)
    return lc_df


def add_flags(lc_df):
    """Adds the following flags to the landcover data:
    - Photo Bit Flags
    - Classification Bit Flags
    - Completeness Score Flags

    This modifies the DataFrame itself.

    Parameters
    ----------
    lc_df : pd.DataFrame
        A DataFrame containing cleaned up Landcover Data ideally from the [apply_cleanup](#apply_cleanup) method.
    """

    photo_bit_flags(
        lc_df,
        "lc_UpwardPhotoUrl",
        "lc_DownwardPhotoUrl",
        "lc_NorthPhotoUrl",
        "lc_SouthPhotoUrl",
        "lc_EastPhotoUrl",
        "lc_WestPhotoUrl",
    )
    classification_bit_flags(
        lc_df,
        "lc_NorthPhotoUrl",
        "lc_SouthPhotoUrl",
        "lc_EastPhotoUrl",
        "lc_WestPhotoUrl",
    )
    completion_scores(lc_df)


def direction_frequency(lc_df, direction_list, bit_binary, entry_type):
    """
    Plots the amount of a variable of interest for each direction.

    Parameters
    ----------
    lc_df : pd.DataFrame
        The DataFrame containing Land Cover Data.
    direction_list : list of str
        The column names of the different variables of interest for each direction.
    bit_binary: str
        The Bit Binary Flag associated with the variable of interest.
    entry_type: str
        The variable of interest (e.g. Photos or Classifications)
    """
    direction_photos = pd.DataFrame()
    direction_photos["category"] = direction_list
    direction_counts = [0 for i in range(len(direction_photos))]
    for mask in lc_df[bit_binary]:
        for i in range(len(mask) - 1, -1, -1):
            direction_counts[i] += int(mask[i])
    direction_counts
    direction_photos["count"] = [math.log10(value) for value in direction_counts]
    direction_photos

    plt.figure(figsize=(15, 6))
    title = f"Land Cover -- {entry_type} Direction Frequency (Log Scale)"
    plt.title(title)
    plt.ylabel("Count (Log Scale)")
    sns.barplot(data=direction_photos, x="category", y="count", color="lightblue")


def diagnostic_plots(lc_df):
    """
    Generates (but doesn't display) diagnostic plots to gain insight into the current data.

    Plots:
    - Valid Photo Count Distribution
    - Photo Distribution by direction
    - Classification Distribution by direction
    - Photo Status Distribution
    - Completeness Score Distribution
    - Subcompleteness Score Distribution

    Parameters
    ----------
    lc_df : pd.DataFrame
        The DataFrame containing Flagged and Cleaned Land Cover Data.
    """
    plot_freq_bar(
        lc_df, "Land Cover", "lc_PhotoCount", "Valid Photo Count", log_scale=True
    )
    direction_frequency(
        lc_df,
        [
            "lc_UpwardPhotoUrl",
            "lc_DownwardPhotoUrl",
            "lc_NorthPhotoUrl",
            "lc_SouthPhotoUrl",
            "lc_EastPhotoUrl",
            "lc_WestPhotoUrl",
        ],
        "lc_PhotoBitBinary",
        "Photo",
    )
    direction_frequency(
        lc_df,
        [
            "lc_NorthClassifications",
            "lc_SouthClassifications",
            "lc_EastClassifications",
            "lc_WestClassifications",
        ],
        "lc_ClassificationBitBinary",
        "Classification",
    )
    multiple_bar_graph(
        lc_df,
        "Land Cover",
        ["lc_PhotoCount", "lc_RejectedCount", "lc_EmptyCount"],
        "Photo Summary",
        log_scale=True,
    )

    completeness_histogram(
        lc_df, "Land Cover", "lc_CumulativeCompletenessScore", "Cumulative Completeness"
    )
    completeness_histogram(
        lc_df, "Land Cover", "lc_SubCompletenessScore", "Sub Completeness"
    )
