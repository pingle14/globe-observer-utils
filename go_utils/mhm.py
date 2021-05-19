import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

from go_utils.cleanup import (
    replace_column_prefix,
    remove_homogenous_cols,
    rename_latlon_cols,
    round_cols,
    standardize_null_vals,
)

from go_utils.plot import (
    plot_int_distribution,
    completeness_histogram,
)


__doc__ = r"""

## Mosquito Specific Cleanup Procedures

### Converting Larvae Data to Integers
Larvae Data is stored as a string in the raw GLOBE Observer dataset. To facillitate analysis, [this method](#larvae_to_num) converts this data to numerical data.

It needs to account for 4 types of data:
1. Regular Data: Converts it to a number
2. Extraneously large data ($\geq 100$ as its hard to count more than that amount accurately): To maintain the information from that entry, the `LarvaeCountMagnitude` flag is used to indicate the real value
3. Ranges (e.g. "25-50"): Chooses the lower bound and set the `LarvaeCountIsRangeFlag` to true.
4. Null Values: Sets null values to $-9999$


It generates the following flags:
- `LarvaeCountMagnitude`: The integer flag contains the order of magnitude (0-4) by which the larvae count exceeds the maximum Larvae Count of 100. This is calculated by $1 + \lfloor \log{\frac{num}{100}} \rfloor$. As a result:
    - `0`: Corresponds to a Larvae Count $\leq 100$
    - `1`: Corresponds to a Larvae Count between $100$ and $999$
    - `2`: Corresponds to a Larvae Count between $1000$ and $9999$
    - `3`: Corresponds to a Larvae Count between $10,000$ and $99,999$
    - `4`: Corresponds to a Larvae Count $\geq 100,000$
- `LarvaeCountIsRange`: Either a $1$ which indicates the entry was a range (e.g. 25-50) or $0$ which indicates the entry wasn't a range.

Additionally, there were extremely large values that Python was unable to process (`1e+27`) and so there was an initial preprocessing step to set those numbers to 100000 (which corresponds to the maximum magnitude flag).
"""


def cleanup_column_prefix(df):
    """Method for shortening raw mosquito habitat mapper column names.

    Example usage:
    ```python
    from go_utils.mhm import cleanup_column_prefix

    cleanup_column_prefix(df)
    ```

    The df object will now replace the verbose `mosquitohabitatmapper` prefix in some of the columns with `mhm_`

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing raw mosquito habitat mapper data. The DataFrame object itself will be modified
    """
    replace_column_prefix(df, "mosquito_habitat_mapper", "mhm")


def larvae_to_num(mhm_df, larvae_count_col="mhm_LarvaeCount"):
    """Converts the Larvae Count of the Mosquito Habitat Mapper Dataset from being stored as a string to integers.

    See [here](#converting-larvae-data-to-integers) for more information.

    Parameters
    ----------
    mhm_df : pd.DataFrame
        A DataFrame of Mosquito Habitat Mapper data that needs the larvae counts to be set to numbers
    larvae_count_col : str, default="mhm_LarvaeCount" (this is intended for DataFrames who have already cleaned their column names)
        The name of the column storing the larvae count. **Note**: The columns will be output in the format: `prefix_ColumnName` where `prefix` is all the characters that preceed the words `LarvaeCount` in the specified name.
    """

    def entry_to_num(entry):
        try:
            if entry == "more than 100":
                return 101, 1, 1
            if pd.isna(entry):
                return -9999, 0, 0
            elif float(entry) > 100:
                return 101, min(math.floor(math.log10(float(entry) / 100)) + 1, 4), 0
            return float(entry), 0, 0
        except ValueError:
            return float(re.sub(r"-.*", "", entry)), 0, 1

    prefix = larvae_count_col.lower().replace("larvaecount", "")
    # Preprocessing step to remove extremely erroneous values
    for i in range(len(mhm_df[larvae_count_col])):
        if (
            not pd.isna(mhm_df[larvae_count_col][i])
            and type(mhm_df[larvae_count_col][i]) is str
            and "e+" in mhm_df[larvae_count_col][i]
        ):
            mhm_df.at[i, larvae_count_col] = "100000"

    larvae_conversion = np.vectorize(entry_to_num)
    (
        mhm_df[larvae_count_col],
        mhm_df[f"{prefix}LarvaeCountMagnitude"],
        mhm_df[f"{prefix}LarvaeCountIsRangeFlag"],
    ) = larvae_conversion(mhm_df[larvae_count_col].to_numpy())


def has_genus_flag(df, genus_col):
    """
    Creates a bit flag: `mhm_HasGenus` where 1 denotes a recorded Genus and 0 denotes the contrary.
    This modifies the DataFrame itself.

    Parameters
    ----------
    df : pd.DataFrame
        A mosquito habitat mapper DataFrame
    genus_col : str
        The column name in the mosquito habitat mapper DataFrame that contains the genus records.
    """
    df["mhm_HasGenus"] = (~pd.isna(df[genus_col].to_numpy())).astype(int)


def infectious_genus_flag(df, genus_col):
    """
    Creates a bit flag: `mhm_IsGenusOfInterest` where 1 denotes a Genus of a infectious mosquito and 0 denotes the contrary.
    This modifies the DataFrame itself.

    Parameters
    ----------
    df : pd.DataFrame
        A mosquito habitat mapper DataFrame
    genus_col : str
        The column name in the mosquito habitat mapper DataFrame that contains the genus records.
    """
    infectious_genus_flag = np.vectorize(
        lambda genus: genus in ["Aedes", "Anopheles", "Culex"]
    )
    df["mhm_IsGenusOfInterest"] = infectious_genus_flag(
        df[genus_col].to_numpy()
    ).astype(int)


def is_container_flag(df, watersource_col):
    """
    Creates a bit flag: `mhm_IsWaterSourceContainer` where 1 denotes if a watersource is a container (e.g. ovitrap, pots, tires, etc.) and 0 denotes the contrary.
    This modifies the DataFrame itself.

    Parameters
    ----------
    df : pd.DataFrame
        A mosquito habitat mapper DataFrame
    watersource_col : str
        The column name in the mosquito habitat mapper DataFrame that contains the watersource records.
    """

    def is_container(entry):
        non_container_keywords = [
            "puddle",
            "still water",
            "stream",
            "estuary",
            "lake",
            "pond",
            "ditch",
            "bay",
            "ocean",
            "swamp",
            "wetland",
        ]
        lowercase = entry.lower()
        for item in non_container_keywords:
            if item in lowercase:
                return False
        return True

    mark_containers = np.vectorize(is_container)
    df["mhm_IsWaterSourceContainer"] = mark_containers(
        df[watersource_col].to_numpy()
    ).astype(int)


def has_watersource_flag(df, watersource_col):
    """
    Creates a bit flag: `mhm_HasWaterSource` where 1 denotes if there is a watersource and 0 denotes the contrary.
    This modifies the DataFrame itself.

    Parameters
    ----------
    df : pd.DataFrame
        A mosquito habitat mapper DataFrame
    watersource_col : str
        The column name in the mosquito habitat mapper DataFrame that contains the watersource records.
    """

    has_watersource = np.vectorize(lambda watersource: int(not pd.isna(watersource)))
    df["mhm_HasWaterSource"] = has_watersource(df[watersource_col].to_numpy())


def photo_bit_flags(df, watersource_photos, larvae_photos, abdomen_photos):
    """
    Creates the following flags:
    - `mhm_PhotoCount`: The number of valid photos per record.
    - `mhm_RejectedCount`: The number of photos that were rejected per record.
    - `mhm_PendingCount`: The number of photos that are pending approval per record.
    - `mhm_PhotoBitBinary`: A string that represents the presence of a photo in the order of watersource, larvae, and abdomen. For example, if the entry is `110`, that indicates that there is a water source photo and a larvae photo, but no abdomen photo.
    - `mhm_PhotoBitDecimal`: The numerical representation of the mhm_PhotoBitBinary string.

    This modifies the DataFrame itself.

    Parameters
    ----------
    df : pd.DataFrame
        A mosquito habitat mapper DataFrame
    watersource_photos : str
        The column name in the mosquito habitat mapper DataFrame that contains the watersource photo url records.
    larvae_photos : str
        The column name in the mosquito habitat mapper DataFrame that contains the larvae photo url records.
    abdomen_photos : str
        The column name in the mosquito habitat mapper DataFrame that contains the abdomen photo url records.
    """

    def pic_data(*args):
        pic_count = 0
        rejected_count = 0
        pending_count = 0
        valid_photo_bit_mask = ""

        # bit_power = len(args) - 1
        # For url string -- if we see ANY http, add 1
        # also count all valid photos, rejected photos,
        # If there are NO http then add 0, to empty photo field
        for url_string in args:
            if not pd.isna(url_string):
                if "http" not in url_string:
                    valid_photo_bit_mask += "0"
                else:
                    valid_photo_bit_mask += "1"

                pic_count += url_string.count("http")
                pending_count += url_string.count("pending")
                rejected_count += url_string.count("rejected")
            else:
                valid_photo_bit_mask += "0"

        return (
            pic_count,
            rejected_count,
            pending_count,
            valid_photo_bit_mask,
            int(valid_photo_bit_mask, 2),
        )

    get_photo_data = np.vectorize(pic_data)
    (
        df["mhm_PhotoCount"],
        df["mhm_RejectedCount"],
        df["mhm_PendingCount"],
        df["mhm_PhotoBitBinary"],
        df["mhm_PhotoBitDecimal"],
    ) = get_photo_data(
        df[watersource_photos].to_numpy(),
        df[larvae_photos].to_numpy(),
        df[abdomen_photos].to_numpy(),
    )


def completion_score_flag(df):
    """
    Adds the following completness score flags:
    - `mhm_SubCompletenessScore`: The percentage of the watersource photos, larvae photos, abdomen photos, and genus columns that are filled out.
    - `mhm_CumulativeCompletenessScore`: The percentage of non null values out of all the columns.

    This modifies the DataFrame itself.

    Parameters
    ----------
    df : pd.DataFrame
        A mosquito habitat mapper DataFrame with the [`mhm_PhotoBitDecimal`](#photo_bit_flags) and [`mhm_HasGenus`](#has_genus_flags) flags.
    """

    def sum_bit_mask(bit_mask="0"):
        total = 0.0
        for char in bit_mask:
            total += int(char)
        return total

    scores = {}
    scores["sub_score"] = []
    # Cummulative Completion Score
    scores["cumulative_score"] = round(df.count(axis=1) / len(df.columns), 2)
    # Sub-Score
    for index in range(len(df)):
        bit_mask = df["mhm_PhotoBitBinary"][index]
        sub_score = df["mhm_HasGenus"][index] + sum_bit_mask(bit_mask=bit_mask)
        sub_score /= 4.0
        scores["sub_score"].append(sub_score)

    df["mhm_SubCompletenessScore"], df["mhm_CumulativeCompletenessScore"] = (
        scores["sub_score"],
        scores["cumulative_score"],
    )


def apply_cleanup(mhm_df):
    """Applies a full cleanup procedure to the mosquito habitat mapper data.
    It follows the following steps:
    - Removes Homogenous Columns
    - Renames Latitude and Longitudes
    - Cleans the Column Naming
    - Converts Larvae Count to Numbers
    - Rounds Columns
    - Standardizes Null Values

    Parameters
    ----------
    mhm_df : pd.DataFrame
        A DataFrame containing **raw** Mosquito Habitat Mapper Data from the API.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the cleaned up Mosquito Habitat Mapper Data
    """
    mhm_df = mhm_df.copy()
    remove_homogenous_cols(mhm_df)
    rename_latlon_cols(mhm_df)
    cleanup_column_prefix(mhm_df)
    larvae_to_num(mhm_df)
    round_cols(mhm_df)
    standardize_null_vals(mhm_df)
    return mhm_df


def add_flags(mhm_df):
    """Adds the following flags to the Mosquito Habitat Mapper Data:
    - Has Genus
    - Is Infectious Genus/Genus of Interest
    - Is Container
    - Has WaterSource
    - Photo Bit Flags
    - Completion Score Flag

    Parameters
    ----------
    mhm_df : pd.DataFrame
        A DataFrame containing cleaned up Mosquito Habitat Mapper Data ideally from the [add_flags](#add_flags) method.
    """

    has_genus_flag(mhm_df, "mhm_Genus")
    infectious_genus_flag(mhm_df, "mhm_Genus")
    is_container_flag(mhm_df, "mhm_WaterSource")
    has_watersource_flag(mhm_df, "mhm_WaterSource")
    photo_bit_flags(
        mhm_df,
        "mhm_WaterSourcePhotoUrls",
        "mhm_LarvaFullBodyPhotoUrls",
        "mhm_AbdomenCloseupPhotoUrls",
    )
    completion_score_flag(mhm_df)


def plot_photo_entries(df):
    """
    Plots the number of entries with photos and the number of entries without photos

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing Mosquito Habitat Mapper Data with the PhotoBitDecimal Flag.
    """
    plt.figure()
    num_valid = len(df[df["mhm_PhotoBitDecimal"] > 0])
    plt.title("Entries with Photos vs No Photos")
    plt.ylabel("Number of Entries")
    plt.bar("Valid Photos", num_valid, color="#e34a33")
    plt.bar("No Photos", len(df) - num_valid, color="#fdcc8a")


def photo_subjects(mhm_df):
    """
    Plots the amount of photos for each photo area (Larvae, Abdomen, Watersource)

    Parameters
    ----------
    mhm_df : pd.DataFrame
        The DataFrame containing Mosquito Habitat Mapper Data with the PhotoBitDecimal Flag.
    """

    total_dict = {"Larvae Photos": 0, "Abdomen Photos": 0, "Watersource Photos": 0}

    for number in mhm_df["mhm_PhotoBitDecimal"]:
        total_dict["Watersource Photos"] += number & 4
        total_dict["Larvae Photos"] += number & 2
        total_dict["Abdomen Photos"] += number & 1

    for key in total_dict.keys():
        total_dict[key] = math.log10(total_dict[key])

    plt.figure(figsize=(10, 5))
    plt.title("Mosquito Habitat Mapper - Photo Subject Frequencies (Log Scale)")
    plt.xlabel("Photo Type")
    plt.ylabel("Frequency (Log Scale)")
    plt.bar(total_dict.keys(), total_dict.values(), color="lightblue")


def diagnostic_plots(mhm_df):
    """
    Generates (but doesn't display) diagnostic plots to gain insight into the current data.

    Plots:
    - Larvae Count Distribution (where a negative entry denotes null data)
    - Photo Subject Distribution
    - Number of valid photos vs no photos
    - Completeness Score Distribution
    - Subcompleteness Score Distribution

    Parameters
    ----------
    mhm_df : pd.DataFrame
        The DataFrame containing Flagged and Cleaned Mosquito Habitat Mapper Data.
    """
    plot_int_distribution(mhm_df, "mhm_LarvaeCount", "Larvae Count")
    photo_subjects(mhm_df)
    plot_photo_entries(mhm_df)
    completeness_histogram(
        mhm_df,
        "Mosquito Habitat Mapper",
        "mhm_CumulativeCompletenessScore",
        "Cumulative Completeness",
    )
    completeness_histogram(
        mhm_df,
        "Mosquito Habitat Mapper",
        "mhm_SubCompletenessScore",
        "Sub Completeness",
    )
