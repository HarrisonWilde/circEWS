"""
Saves the statistics of the sampling frequencies which is later used during adaptive imputation
"""

import argparse
import glob
import os
import os.path
import pickle
import random
import sys
import timeit

import circews.functions.util.io as mlhc_io
import ipdb
import numpy as np
import pandas as pd
import scipy.stats as sp_stats


def save_imputation_params(configs):
    """Saves global imputation parameters conditional on the split key"""
    dim_reduced_data = configs["data_mode"] == "reduced"
    split_key = configs["split_key"]
    print(
        "Saving imputation parameters for split: {}, reduced data: {}".format(
            split_key, dim_reduced_data
        ),
        flush=True,
    )
    pid_batch_dict = mlhc_io.load_pickle(configs["pid_batch_map_binary"])[
        "pid_to_chunk"
    ]
    split_desc = mlhc_io.load_pickle(configs["data_split_binary"])

    if dim_reduced_data:
        var_encoding_dict = mlhc_io.load_pickle(configs["meta_varencoding_map_path"])
    else:
        var_encoding_dict = mlhc_io.load_pickle(configs["varencoding_map_path"])

    relevant_pids = split_desc[split_key]["train"]
    random.seed(configs["random_seed"])
    random.shuffle(relevant_pids)
    batch_schedule = {}
    var_statistics_per_patient = {}
    median_interval_pp = {}
    iqr_interval_pp = {}

    for pid in relevant_pids:
        batch_pid = pid_batch_dict[pid]
        if batch_pid not in batch_schedule:
            batch_schedule[batch_pid] = []
        batch_schedule[batch_pid].append(pid)

    print("Computed batch schedule on-the-fly...", flush=True)
    t_begin = timeit.default_timer()
    n_skipped_patients = 0

    for pidx, pid in enumerate(relevant_pids):
        if (pidx + 1) % 100 == 0:
            print("Data split: {}".format(split_key))
            print("#patients: {}/{}".format(pidx + 1, len(relevant_pids)), flush=True)
            print("Skipped patients: {}".format(n_skipped_patients), flush=True)
            t_end = timeit.default_timer()
            time_pp = (t_end - t_begin) / pidx
            eta_time = time_pp * (len(relevant_pids) - pidx) / 60.0
            print("ETA in minutes: {:.3f}".format(eta_time), flush=True)

        batch_pid = pid_batch_dict[pid]

        if dim_reduced_data:
            cand_files = glob.glob(
                os.path.join(
                    configs["reduced_merged_path"],
                    "reduced_fmat_{}_*.h5".format(batch_pid),
                )
            )
        else:
            cand_files = glob.glob(
                os.path.join(configs["merged_path"], "fmat_{}_*.h5".format(batch_pid))
            )

        assert len(cand_files) == 1
        df_patient = pd.read_hdf(
            cand_files[0], mode="r", where="PatientID={}".format(pid)
        )

        if df_patient.shape[0] == 0:
            n_skipped_patients += 1

        df_patient.sort_values(by="Datetime", kind="mergesort", inplace=True)

        if pidx == 0:
            all_vars = df_patient.columns.values.tolist()
            relevant_vars = set(all_vars).difference(
                set(["Datetime", "PatientID", "a_temp", "m_pm_1", "m_pm_2"])
            )

        for var in relevant_vars:
            if var[0] == "p":
                continue

            var_encoding = var_encoding_dict[var]
            value_vect = df_patient[var].dropna()

            if value_vect.size == 0:
                continue

            if var_encoding == "continuous":
                summary_statistic = np.median(np.array(value_vect))
            elif var_encoding == "categorical":
                summary_statistic = int(value_vect.mode().iloc[0])
            elif var_encoding == "ordinal":
                summary_statistic = int(round(np.median(np.array(value_vect))))
            elif var_encoding == "binary":
                summary_statistic = int(value_vect.mode().iloc[0])

            if var not in var_statistics_per_patient:
                var_statistics_per_patient[var] = []

            var_statistics_per_patient[var].append(summary_statistic)
            ts_value_vect = df_patient[["Datetime", var]].dropna()

            if ts_value_vect.shape[0] >= 3:
                diff_series = (
                    np.array(
                        ts_value_vect["Datetime"].diff().dropna(), dtype=np.float64
                    )
                    / 1000000000.0
                )
                median_interval = np.median(diff_series)
                iqr_interval = sp_stats.iqr(diff_series)

                if var not in median_interval_pp:
                    median_interval_pp[var] = []
                    iqr_interval_pp[var] = []

                median_interval_pp[var].append(median_interval)
                iqr_interval_pp[var].append(iqr_interval)

    global_impute_dict = {}
    interval_median_dict = {}
    interval_iqr_dict = {}

    for var in relevant_vars:
        if var[0] == "p":
            continue

        var_encoding = var_encoding_dict[var]

        if var not in var_statistics_per_patient:
            continue

        all_vals = var_statistics_per_patient[var]

        if var_encoding == "continuous":
            global_impute_val = np.median(all_vals)
        elif var_encoding in ["categorical", "binary"]:
            global_impute_val = int(sp_stats.mode(all_vals)[0])
        elif var_encoding == "ordinal":
            global_impute_val = int(round(np.median(all_vals)))
        else:
            print("ERROR: Invalid variable encoding...", flush=True)
            assert False

        global_impute_dict[var] = global_impute_val

    for var in relevant_vars:
        if var[0] == "p":
            continue

        if var not in median_interval_pp:
            interval_median_dict[var] = np.nan
            interval_iqr_dict[var] = np.nan
            continue

        interval_median_dict[var] = np.median(np.array(median_interval_pp[var]))
        interval_iqr_dict[var] = np.median(np.array(iqr_interval_pp[var]))

    if dim_reduced_data:
        output_impute_dict = configs["imputation_param_dict_reduced"]
    else:
        output_impute_dict = configs["imputation_param_dict"]

    if not configs["debug_mode"]:
        mlhc_io.save_pickle(
            global_impute_dict,
            os.path.join(
                output_impute_dict, "global_vals_from_data_{}.pickle".format(split_key)
            ),
        )
        mlhc_io.save_pickle(
            interval_median_dict,
            os.path.join(
                output_impute_dict, "interval_median_{}.pickle".format(split_key)
            ),
        )
        mlhc_io.save_pickle(
            interval_iqr_dict,
            os.path.join(
                output_impute_dict, "interval_iqr_{}.pickle".format(split_key)
            ),
        )


def parse_cmd_args():
    # Input paths
    META_VARENCODING_MAP_PATH = os.path.abspath(
        "./external_validation/resource/meta_varencoding_map.pickle"
    )

    # Output paths
    LOG_DIR = "/data/qmia/mimiciii/validation/misc_derived/log"

    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--version", type=str, required=True, help="Version to run")
    parser.add_argument(
        "--meta_varencoding_map_path",
        default=META_VARENCODING_MAP_PATH,
        help="Dictionary mapping variable IDs to variable encoding to distinguish between processing categories",
    )
    parser.add_argument("--log_dir", default=LOG_DIR, help="Logging directory")

    # Arguments
    parser.add_argument("--random_seed", type=int, default=2018, help="Random seed")
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        default=False,
        help="Debug mode, no output written to file-system",
    )
    parser.add_argument(
        "--run_mode",
        default="INTERACTIVE",
        help="Should the code be executed in batch or interactive mode?",
    )
    parser.add_argument(
        "--split_key",
        default="random_5",
        help="For which split should imputation parameters be computed?",
    )
    parser.add_argument(
        "--data_mode", default="reduced", help="Should dim-reduced data be processed?"
    )

    args = parser.parse_args()
    configs = vars(args)

    configs["data_split_binary"] = (
        f"/data/qmia/mimiciii/validation/misc_derived/split_{configs['version']}.pickle"
    )
    configs["pid_batch_map_binary"] = (
        f"/data/qmia/mimiciii/validation/external_validation/misc_derived/id_lists/chunks_{configs['version']}.pickle"
    )
    configs["reduced_merged_path"] = (
        f"/data/qmia/mimiciii/validation/external_validation/merged/{configs['version']}/reduced"
    )
    configs["imputation_param_dict_reduced"] = (
        f"/data/qmia/mimiciii/validation/misc_derived/imputation_reduced_{configs['version']}"
    )
    os.makedirs(configs["imputation_param_dict_reduced"], exist_ok=True)
    configs["imputation_param_dict"] = (
        f"/data/qmia/mimiciii/validation/misc_derived/imputation_{configs['version']}"
    )
    os.makedirs(configs["imputation_param_dict"], exist_ok=True)
    return configs


if __name__ == "__main__":
    configs = parse_cmd_args()

    run_mode = configs["run_mode"]
    assert run_mode in ["CLUSTER", "INTERACTIVE"]
    dim_reduced_str = configs["data_mode"]
    assert dim_reduced_str in ["reduced", "non_reduced"]
    split_key = configs["split_key"]

    if run_mode == "CLUSTER":
        sys.stdout = open(
            os.path.join(
                configs["log_dir"],
                "IMPUTEPARAMS_{}_{}.stdout".format(split_key, dim_reduced_str),
            ),
            "w",
        )
        sys.stderr = open(
            os.path.join(
                configs["log_dir"],
                "IMPUTEPARAMS_{}_{}.stderr".format(split_key, dim_reduced_str),
            ),
            "w",
        )

    save_imputation_params(configs)
