"""
Imputes static data of all patients and saves back to file-system
"""

import argparse
import os
import os.path
import sys

import circews.classes.imputer_static as tf_impute_static
import circews.functions.util.io as mlhc_io
import ipdb
import numpy as np
import pandas as pd


def impute_static_data(dim_reduced_data, split_key, configs=None):
    data_split = mlhc_io.load_pickle(configs["data_split_binary"])[split_key]
    train = data_split["train"]
    val = data_split["val"]
    test = data_split["test"]

    if configs["dataset"] == "bern":
        all_pids = train + val + test
        reduced_output_base_dir = configs["bern_imputed_reduced_path"]
        output_base_dir = configs["bern_imputed_path"]
    elif configs["dataset"] == "mimic":
        all_pids = map(
            int, mlhc_io.read_list_from_file(configs["mimic_all_pid_list_path"])
        )
        reduced_output_base_dir = configs["mimic_imputed_reduced_path"]
        output_base_dir = configs["mimic_imputed_path"]

    # df_static_bern = pd.read_hdf(configs["bern_static_info_path"], mode="r")
    df_static_mimic = pd.read_hdf(configs["mimic_static_info_path"], mode="r")
    df_train = df_static_mimic[df_static_mimic["PatientID"].isin(train)]

    # if configs["dataset"] == "bern":
    # df_all = df_static_bern[df_static_bern["PatientID"].isin(all_pids)]
    if configs["dataset"] == "mimic":
        df_all = df_static_mimic[df_static_mimic["PatientID"].isin(all_pids)]

    tf_model = tf_impute_static.StaticDataImputer(dataset=configs["dataset"])
    df_static_imputed = tf_model.transform(df_all, df_train=df_train)

    if dim_reduced_data:
        base_dir = os.path.join(reduced_output_base_dir, split_key)
    else:
        base_dir = os.path.join(output_base_dir, split_key)

    assert df_static_imputed.isnull().sum().values.sum() == 0

    if not configs["debug_mode"]:
        os.makedirs(base_dir, exist_ok=True)
        df_static_imputed.to_hdf(
            os.path.join(base_dir, "static.h5"),
            "data",
            complevel=configs["hdf_comp_level"],
            complib=configs["hdf_comp_alg"],
        )


def parse_cmd_args():
    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument("--version", required=True, help="Version to run")
    parser.add_argument(
        "--hdf_comp_level", type=int, default=5, help="HDF compression level"
    )
    parser.add_argument(
        "--hdf_comp_alg", default="blosc:lz4", help="HDF compression algorithm"
    )
    parser.add_argument(
        "--debug_mode",
        default=False,
        action="store_true",
        help="Debug mode stores nothing to disk",
    )
    parser.add_argument(
        "--dataset",
        default="mimic",
        help="For which data-set should static imputation be run?",
    )

    args = parser.parse_args()
    configs = vars(args)

    configs["data_split_binary"] = (
        f"/data/qmia/mimiciii/validation/misc_derived/split_{configs['version']}.pickle"
    )
    configs["mimic_static_info_path"] = (
        f"/data/qmia/mimiciii/validation/external_validation/merged/{configs['version']}/static.h5"
    )
    configs["mimic_all_pid_list_path"] = (
        f"/data/qmia/mimiciii/validation/external_validation/pids_with_endpoint_data.csv.{configs['version']}"
    )
    configs["mimic_imputed_path"] = (
        f"/data/qmia/mimiciii/validation/external_validation/imputed/imputed_{configs['version']}"
    )
    os.makedirs(configs["mimic_imputed_path"], exist_ok=True)
    configs["mimic_imputed_reduced_path"] = f"{configs['mimic_imputed_path']}/reduced"
    os.makedirs(configs["mimic_imputed_reduced_path"], exist_ok=True)

    configs["DATA_MODES"] = ["reduced"]
    configs["SPLIT_MODES"] = [
        "held_out",
        "random_1",
        "random_2",
        "random_3",
        "random_4",
        "random_5",
    ]
    return configs


if __name__ == "__main__":
    configs = parse_cmd_args()

    for dim_reduced_str in configs["DATA_MODES"]:
        for split_key in configs["SPLIT_MODES"]:
            dim_reduced_data = dim_reduced_str == "reduced"
            print(
                "Imputing static data for reduced data: {} on split: {}".format(
                    dim_reduced_data, split_key
                )
            )
            impute_static_data(dim_reduced_data, split_key, configs=configs)
