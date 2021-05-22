import argparse
import matplotlib.pyplot as plt
import pandas as pd

from go_utils.download import get_api_data, get_country_api_data
from go_utils import mhm, lc
from go_utils.photo_download import (
    download_cleaned_mhm_photos,
    download_cleaned_lc_photos,
)

protocol_map = {"mosquito": "mosquito_habitat_mapper", "landcover": "land_covers"}


def add_download_args(parser):
    parser.add_argument("--out", "-o", help="Output Directory of the command")
    parser.add_argument(
        "--start",
        "-s",
        help="Start date (if you want data from the api, so don't specify -i)",
    )
    parser.add_argument(
        "--end",
        "-e",
        help="End date (if you want data from the api, so don't specify -i)",
    )
    parser.add_argument(
        "--countries",
        "-co",
        help="Desired Countries separated by commas (if you want data from the api, so don't specify -i)",
    )
    parser.add_argument(
        "--regions",
        "-r",
        help="Desired Regions separated by commas (if you want data from the api, so don't specify -i)",
    )
    parser.add_argument(
        "--box",
        "-b",
        help="Bounding Box (if you want data from the api, so don't specify -i). Put coordinates in order of 'min lat, min lon, max lat, max lon'",
        type=list,
    )


def download_data(protocol, args):
    func_args = {"protocol": protocol}
    if args.start:
        func_args["start_date"] = args.start
    if args.end:
        func_args["end_date"] = args.end
    if args.box:
        coords = [float(coord.strip()) for coord in args.box.split(",")]
        box = {
            "min_lat": coords[0],
            "min_lon": coords[1],
            "max_lat": coords[1],
            "max_lon": coords[3],
        }
        func_args["latlon_box"] = box
    if args.countries:
        func_args["countries"] = [
            country.strip() for country in args.countries.split(",")
        ]
    if args.regions:
        func_args["regions"] = [region.strip() for region in args.regions.split(",")]

    if "countries" in func_args or "regions" in func_args:
        df = get_country_api_data(**func_args)
    else:
        df = get_api_data(**func_args)

    if "mosquito" in protocol:
        df = mhm.apply_cleanup(df)
        mhm.add_flags(df)
    else:
        df = lc.apply_cleanup(df)
        lc.add_flags(df)

    return df


def mhm_data_download():
    parser = argparse.ArgumentParser(
        description="GLOBE Observer Mosquito Habitat Mapper API Download CLI"
    )
    add_download_args(parser)
    parser.add_argument(
        "--hasgenus",
        "-hg",
        help="Filter data if it has a genus as a record",
        action="store_true",
    )
    parser.add_argument(
        "--iscontainer",
        "-ic",
        help="Filter data if the record's watersource is a container",
        action="store_true",
    )
    parser.add_argument(
        "--hasphotos",
        "-hp",
        help="Filter data if it has photo records",
        action="store_true",
    )
    parser.add_argument(
        "--minlarvae", "-ml", help="Filter data by minimum larvae count", type=int
    )
    args = parser.parse_args()
    df = download_data("mosquito_habitat_mapper", args)

    filter_args = {}
    filter_args["has_genus"] = args.hasgenus
    filter_args["is_container"] = args.iscontainer
    filter_args["has_photos"] = args.hasphotos

    if args.minlarvae:
        filter_args["min_larvae_count"] = args.minlarvae

    df = mhm.qa_filter(df, **filter_args)

    if args.out:
        df.to_csv(args.out)
    else:
        mhm.diagnostic_plots(df)
        plt.show()


def lc_data_download():
    parser = argparse.ArgumentParser(
        description="GLOBE Observer Land Cover API Download CLI"
    )
    add_download_args(parser)

    # TODO: Group args to reduce user confusion
    parser.add_argument(
        "--hasclassification",
        "-hc",
        help="Filter data if it has atleast one classification as a record",
        action="store_true",
    )
    parser.add_argument(
        "--hasphoto",
        "-hp",
        help="Filter data if it has atleast one photo as a record",
        action="store_true",
    )
    parser.add_argument(
        "--hasallclassifications",
        "-hac",
        help="Filter data if it has all classifications for each record",
        action="store_true",
    )
    parser.add_argument(
        "--hasallphotos",
        "-hap",
        help="Filter data if it has all photos for each record",
        action="store_true",
    )
    args = parser.parse_args()
    df = download_data("land_covers", args)

    filter_args = {}
    filter_args["has_classification"] = args.hasclassification
    filter_args["has_photo"] = args.hasphoto
    filter_args["has_all_classifications"] = args.hasallclassifications
    filter_args["has_all_photos"] = args.hasallphotos
    df = lc.qa_filter(df, **filter_args)

    if args.out:
        df.to_csv(args.out)
    else:
        lc.diagnostic_plots(df)
        plt.show()


def mhm_photo_download():
    parser = argparse.ArgumentParser(
        description="GLOBE Observer Mosquito Habitat Mapper Photo Downloader CLI"
    )
    parser.add_argument("input", help="Input Directory of the command")
    parser.add_argument("out", help="Output Directory of the command")
    parser.add_argument(
        "--larvae", "-l", help="Include Larvae Photos", action="store_true"
    )
    parser.add_argument(
        "--watersource", "-w", help="Include Watersource Photos", action="store_true"
    )
    parser.add_argument(
        "--abdomen", "-a", help="Include Abdomen Photos", action="store_true"
    )
    parser.add_argument(
        "--species",
        "-s",
        help="Include Species in Filename if it is classified",
        action="store_true",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    download_cleaned_mhm_photos(
        df,
        args.out,
        args.species,
        args.larvae,
        args.watersource,
        args.abdomen,
    )


def lc_photo_download():
    parser = argparse.ArgumentParser(
        description="GLOBE Observer Land Cover Photo Downloader CLI"
    )
    parser.add_argument("input", help="Input Directory of the command")
    parser.add_argument("out", help="Output Directory of the command")

    parser.add_argument("--up", "-u", help="Include Upward Photos", action="store_true")
    parser.add_argument(
        "--down", "-d", help="Include Downward Photos", action="store_true"
    )
    parser.add_argument(
        "--north", "-n", help="Include Northern Photos", action="store_true"
    )
    parser.add_argument(
        "--south", "-s", help="Include Southern Photos", action="store_true"
    )
    parser.add_argument(
        "--east", "-e", help="Include Eastern Photos", action="store_true"
    )
    parser.add_argument(
        "--west", "-w", help="Include Western Photos", action="store_true"
    )
    parser.add_argument("--all", "-a", help="Include All Photos", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if args.all:
        download_cleaned_lc_photos(
            df, args.out, True, True, True, True, True, True, args.verbose
        )
    else:
        download_cleaned_lc_photos(
            df,
            args.out,
            args.up,
            args.down,
            args.north,
            args.south,
            args.east,
            args.west,
        )
