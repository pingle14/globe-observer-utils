import math
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
            and "e+" in mhm_df[larvae_count_col][i]
        ):
            mhm_df.at[i, larvae_count_col] = "100000"

    larvae_conversion = np.vectorize(entry_to_num)
    (
        mhm_df[larvae_count_col],
        mhm_df[f"{prefix}LarvaeCountMagnitude"],
        mhm_df[f"{prefix}LarvaeCountIsRangeFlag"],
    ) = larvae_conversion(mhm_df[larvae_count_col].to_numpy())


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
    """
    remove_homogenous_cols(mhm_df)
    rename_latlon_cols(mhm_df)
    cleanup_column_prefix(mhm_df)
    larvae_to_num(mhm_df)
    round_cols(mhm_df)
    standardize_null_vals(mhm_df)
