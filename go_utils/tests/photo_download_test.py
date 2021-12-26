import shutil

import pandas as pd
import pytest

from go_utils.download import convert_dates_to_datetime

from go_utils.photo_download import (  # isort: skip
    download_lc_photos,
    download_mhm_photos,
    get_lc_download_targets,
    get_mhm_download_targets,
    remove_bad_characters,
)

out_directory = "test_photos"
desired_mhm_names = [
    "mhm_Larvae_adult mosquito trap_-22.9582_-43.1994_2021-01-05_26415_Aedes incerta_2076283.png",
    "mhm_Watersource_adult mosquito trap_-22.9582_-43.1994_2021-01-05_26415_Aedes incerta_2076282.png",
    "mhm_Watersource_adult mosquito trap_-22.9582_-43.1994_2021-01-05_26416_Aedes incerta_2076285.png",
    "mhm_Watersource_adult mosquito trap_-22.9582_-43.1994_2021-01-05_26416_Aedes incerta_2076286.png",
    "mhm_Watersource_adult mosquito trap_-22.9582_-43.1994_2021-01-05_26416_Aedes incerta_2076284.png",
    "mhm_Watersource_dish or pot_13.9162_-15.6708_2021-01-07_26422_Aedes_2082763.png",
    "mhm_Larvae_dish or pot_13.9162_-15.6708_2021-01-07_26422_Aedes_2082764.png",
    "mhm_Larvae_other_33.6961_-117.9615_2021-01-21_26455_Culex_2094410.png",
    "mhm_Larvae_other_33.6961_-117.9615_2021-01-21_26455_Culex_2094411.png",
    "mhm_Larvae_other_33.6961_-117.9615_2021-01-21_26455_Culex_2094412.png",
    "mhm_Larvae_other_33.6961_-117.9615_2021-01-21_26455_Culex_2094413.png",
    "mhm_Larvae_other_33.6961_-117.9615_2021-01-21_26455_Culex_2094414.png",
    "mhm_Larvae_other_33.6961_-117.9615_2021-01-21_26455_Culex_2094416.png",
    "mhm_Watersource_other_33.6961_-117.9615_2021-01-21_26455_Culex_2094409.png",
    "mhm_Watersource_ditch_9.1515_7.3363_2021-01-23_26456_None_2096252.png",
    "mhm_Larvae_ditch_9.1515_7.3363_2021-01-23_26456_None_2096253.png",
    "mhm_Larvae_ditch_9.1515_7.3363_2021-01-23_26456_None_2096254.png",
    "mhm_Larvae_ditch_9.1515_7.3363_2021-01-23_26456_None_2096255.png",
    "mhm_Larvae_ditch_9.1515_7.3363_2021-01-23_26456_None_2096256.png",
    "mhm_Larvae_ditch_9.1515_7.3363_2021-01-23_26456_None_2096257.png",
    "mhm_Larvae_ditch_9.1515_7.3363_2021-01-23_26456_None_2096258.png",
    "mhm_Larvae_ditch_9.1515_7.3363_2021-01-23_26456_None_2096259.png",
    "mhm_Larvae_ditch_9.1515_7.3363_2021-01-23_26456_None_2096260.png",
]

desired_lc_names = [
    "lc_Down_39.1857_-86.7782_2021-01-01_38513_2072274.png",
    "lc_East_39.1857_-86.7782_2021-01-01_38513_2072270.png",
    "lc_Up_39.1857_-86.7782_2021-01-01_38513_2072273.png",
    "lc_West_39.1857_-86.7782_2021-01-01_38513_2072272.png",
    "lc_South_39.1857_-86.7782_2021-01-01_38513_2072271.png",
    "lc_North_39.1857_-86.7782_2021-01-01_38513_2072269.png",
    "lc_North_39.1857_-86.7782_2021-01-03_38532_2074022.png",
    "lc_Up_39.1857_-86.7782_2021-01-03_38532_2074026.png",
    "lc_West_39.1857_-86.7782_2021-01-03_38532_2074025.png",
    "lc_East_39.1857_-86.7782_2021-01-03_38532_2074023.png",
    "lc_North_39.1857_-86.7782_2021-01-05_38550_2076062.png",
    "lc_South_39.1857_-86.7782_2021-01-05_38550_2076064.png",
    "lc_Up_39.1857_-86.7782_2021-01-05_38550_2076066.png",
    "lc_West_39.1857_-86.7782_2021-01-05_38550_2076065.png",
    "lc_North_31.883_-106.4354_2021-01-04_38535_2075102.png",
    "lc_South_31.883_-106.4354_2021-01-04_38535_2075104.png",
    "lc_Up_31.883_-106.4354_2021-01-04_38535_2075106.png",
    "lc_West_31.883_-106.4354_2021-01-04_38535_2075105.png",
    "lc_East_31.883_-106.4354_2021-01-04_38535_2075103.png",
    "lc_Down_31.883_-106.4354_2021-01-04_38535_2075107.png",
    "lc_North_-39.9572_-71.0726_2021-01-05_38547_2075767.png",
    "lc_South_-39.9572_-71.0726_2021-01-05_38547_2075769.png",
    "lc_Up_-39.9572_-71.0726_2021-01-05_38547_2075771.png",
    "lc_West_-39.9572_-71.0726_2021-01-05_38547_2075770.png",
    "lc_East_-39.9572_-71.0726_2021-01-05_38547_2075768.png",
]


@pytest.mark.photodownload
@pytest.mark.util
def test_bad_characters():
    output = remove_bad_characters('<bad-/test|"\\filename:?>*')
    assert output == "bad-testfilename"


@pytest.mark.photodownload
@pytest.mark.parametrize(
    "input_file, func, desired",
    [
        (
            "go_utils/tests/sample_data/mhm_small.csv",
            get_mhm_download_targets,
            desired_mhm_names,
        ),
        (
            "go_utils/tests/sample_data/lc_small.csv",
            get_lc_download_targets,
            desired_lc_names,
        ),
    ],
)
def test_naming(input_file, func, desired):
    df = pd.read_csv(input_file)
    convert_dates_to_datetime(df)

    targets = func(df, "")
    success = True
    output_filenames = [target[2] for target in targets]
    for filename in desired:
        if filename not in output_filenames:  # pragma: no cover
            success = False
            print(filename)

    for filename in output_filenames:
        if filename not in desired:  # pragma: no cover
            success = False
            print(filename)

    assert success
    assert len(output_filenames) == len(desired)


@pytest.fixture(
    scope="module",
    params=[
        ("go_utils/tests/sample_data/mhm_small.csv", download_mhm_photos),
        ("go_utils/tests/sample_data/lc_small.csv", download_lc_photos),
    ],
)
def photodownload_setup(request):
    input_file, func = request.param
    df = pd.read_csv(input_file)
    convert_dates_to_datetime(df)
    yield df, func
    shutil.rmtree(out_directory)


@pytest.mark.downloadtest
@pytest.mark.photodownload
def test_photodownload(photodownload_setup):
    df, func = photodownload_setup
    func(df, out_directory)
