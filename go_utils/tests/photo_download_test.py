import pytest
import pandas as pd
import shutil

from go_utils.photo_download import (
    get_mhm_download_targets,
    get_lc_download_targets,
    download_cleaned_mhm_photos,
    download_cleaned_lc_photos,
)


out_directory = "test_photos"
desired_mhm_names = [
    "mhm-Larvae-adult mosquito trap-2021-01-05-26415-Aedes incerta-2076283.png",
    "mhm-Watersource-adult mosquito trap-2021-01-05-26415-Aedes incerta-2076282.png",
    "mhm-Watersource-adult mosquito trap-2021-01-05-26416-Aedes incerta-2076285.png",
    "mhm-Watersource-adult mosquito trap-2021-01-05-26416-Aedes incerta-2076286.png",
    "mhm-Watersource-adult mosquito trap-2021-01-05-26416-Aedes incerta-2076284.png",
    "mhm-Watersource-dish or pot-2021-01-07-26422-Aedes-2082763.png",
    "mhm-Larvae-dish or pot-2021-01-07-26422-Aedes-2082764.png",
    "mhm-Larvae-other-2021-01-21-26455-Culex-2094410.png",
    "mhm-Larvae-other-2021-01-21-26455-Culex-2094411.png",
    "mhm-Larvae-other-2021-01-21-26455-Culex-2094412.png",
    "mhm-Larvae-other-2021-01-21-26455-Culex-2094413.png",
    "mhm-Larvae-other-2021-01-21-26455-Culex-2094414.png",
    "mhm-Larvae-other-2021-01-21-26455-Culex-2094416.png",
    "mhm-Watersource-other-2021-01-21-26455-Culex-2094409.png",
    "mhm-Watersource-ditch-2021-01-23-26456-None-2096252.png",
    "mhm-Larvae-ditch-2021-01-23-26456-None-2096253.png",
    "mhm-Larvae-ditch-2021-01-23-26456-None-2096254.png",
    "mhm-Larvae-ditch-2021-01-23-26456-None-2096255.png",
    "mhm-Larvae-ditch-2021-01-23-26456-None-2096256.png",
    "mhm-Larvae-ditch-2021-01-23-26456-None-2096257.png",
    "mhm-Larvae-ditch-2021-01-23-26456-None-2096258.png",
    "mhm-Larvae-ditch-2021-01-23-26456-None-2096259.png",
    "mhm-Larvae-ditch-2021-01-23-26456-None-2096260.png",
]

desired_lc_names = [
    "lc-Down-2021-01-01-38513-2072274.png",
    "lc-East-2021-01-01-38513-2072270.png",
    "lc-Up-2021-01-01-38513-2072273.png",
    "lc-West-2021-01-01-38513-2072272.png",
    "lc-South-2021-01-01-38513-2072271.png",
    "lc-North-2021-01-01-38513-2072269.png",
    "lc-North-2021-01-03-38532-2074022.png",
    "lc-Up-2021-01-03-38532-2074026.png",
    "lc-West-2021-01-03-38532-2074025.png",
    "lc-East-2021-01-03-38532-2074023.png",
    "lc-Down-2021-01-03-38532-2074027.png",
    "lc-North-2021-01-05-38550-2076062.png",
    "lc-South-2021-01-05-38550-2076064.png",
    "lc-Up-2021-01-05-38550-2076066.png",
    "lc-West-2021-01-05-38550-2076065.png",
    "lc-Down-2021-01-05-38550-2076067.png",
    "lc-North-2021-01-04-38535-2075102.png",
    "lc-South-2021-01-04-38535-2075104.png",
    "lc-Up-2021-01-04-38535-2075106.png",
    "lc-West-2021-01-04-38535-2075105.png",
    "lc-East-2021-01-04-38535-2075103.png",
    "lc-Down-2021-01-04-38535-2075107.png",
    "lc-North-2021-01-05-38547-2075767.png",
    "lc-South-2021-01-05-38547-2075769.png",
    "lc-Up-2021-01-05-38547-2075771.png",
    "lc-West-2021-01-05-38547-2075770.png",
    "lc-East-2021-01-05-38547-2075768.png",
    "lc-Down-2021-01-05-38547-2075772.png",
]

mhm_cols = [
    "mhm_WaterSource",
    "mhm_measuredDate",
    "mhm_MosquitoHabitatMapperId",
    "mhm_Genus",
    "mhm_Species",
    "mhm_LarvaFullBodyPhotoUrls",
    "mhm_WaterSourcePhotoUrls",
    "mhm_AbdomenCloseupPhotoUrls",
]
lc_cols = [
    "lc_measuredDate",
    "lc_LandCoverId",
    "lc_UpwardPhotoUrl",
    "lc_DownwardPhotoUrl",
    "lc_NorthPhotoUrl",
    "lc_SouthPhotoUrl",
    "lc_EastPhotoUrl",
    "lc_WestPhotoUrl",
]


@pytest.mark.photodownload
@pytest.mark.parametrize(
    "input_file, func, cols, desired",
    [
        (
            "go_utils/tests/sample_data/mhm_small.csv",
            get_mhm_download_targets,
            mhm_cols,
            desired_mhm_names,
        ),
        (
            "go_utils/tests/sample_data/lc_small.csv",
            get_lc_download_targets,
            lc_cols,
            desired_lc_names,
        ),
    ],
)
def test_naming(input_file, func, cols, desired):
    df = pd.read_csv(input_file)

    targets = func(df, "", *cols)
    print(targets)
    success = True
    for target in targets:
        if target[2] not in desired:
            print(target[2])
            success = False

    assert success
    assert len(targets) == len(desired)


@pytest.fixture(
    scope="module",
    params=[
        ("go_utils/tests/sample_data/mhm_small.csv", download_cleaned_mhm_photos),
        ("go_utils/tests/sample_data/lc_small.csv", download_cleaned_lc_photos),
    ],
)
def photodownload_setup(request):
    input_file, func = request.param
    df = pd.read_csv(input_file)
    yield df, func
    shutil.rmtree(out_directory)


@pytest.mark.downloadtest
@pytest.mark.photodownload
def test_photodownload(photodownload_setup):
    df, func = photodownload_setup
    func(df, out_directory)
