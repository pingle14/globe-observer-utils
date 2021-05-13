import numpy as np
import pandas as pd
import re

from go_utils.cleanup import replace_column_prefix

classifications = [
    "lc_WestClassifications",
    "lc_EastClassifications",
    "lc_NorthClassifications",
    "lc_SouthClassifications",
]


def cleanup_column_prefix(df):
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
    return re.search(r"(?<=\[).*(?=\])", entry).group()


def extract_classification_percentage(entry):
    return float(re.search(".*(?=%)", entry).group())


def extract_classifications(info):
    entries = info.split(";")
    return [extract_classification_name(entry) for entry in entries]


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
