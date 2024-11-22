"""Label generation from endpoints/imputed data"""

import argparse
import gc
import glob
import os
import os.path
import sys

import circews.classes.label_gen as tf_labels
import circews.functions.util.filesystem as mlhc_fs
import circews.functions.util.io as mlhc_io
import numpy as np
import pandas as pd


def is_df_sorted(df, colname):
    return (np.array(df[colname].diff().dropna(), dtype=np.float64) >= 0).all()


def label_gen(configs):
    """Creation of base labels directly defined on the imputed data / endpoints"""

    label_key = configs["label_key"]
    split_key = configs["split_key"]
    lhours = configs["lhours"]
    rhours = configs["rhours"]
    data_mode = configs["data_mode"]

    if data_mode == "reduced":
        dim_reduced_data = True
    else:
        dim_reduced_data = False

    if configs["verbose"]:
        print(
            "Creating label: {} [{},{}] for reduced data: {}".format(
                label_key, lhours, rhours, dim_reduced_data
            ),
            flush=True,
        )

    if configs["dataset"] == "bern":
        label_base_dir = configs["bern_output_base_dir"]
        endpoint_base_dir = configs["bern_endpoints_dir"]
        imputed_base_dir = configs["bern_imputed_dir"]
    elif configs["dataset"] == "mimic":
        label_base_dir = configs["mimic_output_base_dir"]
        endpoint_base_dir = configs["mimic_endpoints_dir"]
        imputed_base_dir = configs["mimic_imputed_path"]

    if dim_reduced_data:
        base_dir = os.path.join(
            label_base_dir,
            "reduced",
            split_key,
            label_key,
            "{}To{}Hours".format(lhours, rhours),
        )
    else:
        base_dir = os.path.join(
            label_base_dir, split_key, label_key, "{}To{}Hours".format(lhours, rhours)
        )

    try:
        if not configs["debug_mode"]:
            mlhc_fs.create_dir_if_not_exist(base_dir, recursive=True)
    except:
        print("WARNING: Race condition when creating directory from different jobs...")

    data_split = mlhc_io.load_pickle(configs["data_split_binary"])[split_key]

    if configs["dataset"] == "bern":
        all_pids = data_split["train"] + data_split["val"] + data_split["test"]
    elif configs["dataset"] == "mimic":
        all_pids = list(
            map(int, mlhc_io.read_list_from_file(configs["mimic_all_pid_list_path"]))
        )

    if configs["verbose"]:
        print("Number of patient IDs: {}".format(len(all_pids)), flush=True)

    if configs["dataset"] == "bern":
        batch_map = mlhc_io.load_pickle(configs["bern_pid_batch_map_binary"])[
            "chunk_to_pids"
        ]
    elif configs["dataset"] == "mimic":
        batch_map = mlhc_io.load_pickle(configs["mimic_pid_batch_map_binary"])[
            "chunk_to_pids"
        ]

    batch_idx = configs["batch_idx"]

    if not configs["debug_mode"]:
        mlhc_fs.delete_if_exist(os.path.join(base_dir, "batch_{}.h5".format(batch_idx)))

    pids_batch = batch_map[batch_idx]
    selected_pids = list(set(pids_batch).intersection(all_pids))
    n_skipped_patients = 0
    first_write = True

    if label_key == "Deterioration":
        tf_model = tf_labels.DeteriorationLabel(lhours, rhours)
    elif label_key == "WorseState":
        tf_model = tf_labels.WorseStateLabel(lhours, rhours)
    elif label_key == "WorseStateSoft":
        tf_model = tf_labels.WorseStateSoftLabel(lhours, rhours)
    elif label_key == "AllLabels":
        tf_model = tf_labels.AllLabel(lhours, rhours, dataset=configs["dataset"])
    else:
        print("ERROR: Invalid label requested...", flush=True)
        sys.exit(1)

    print("Number of selected PIDs: {}".format(len(selected_pids)), flush=True)

    for pidx, pid in enumerate(selected_pids):
        if dim_reduced_data:
            patient_path = os.path.join(
                imputed_base_dir, "reduced", split_key, "batch_{}.h5".format(batch_idx)
            )
            cand_files = glob.glob(
                os.path.join(
                    endpoint_base_dir,
                    "reduced",
                    "reduced_endpoints_{}_*.h5".format(batch_idx),
                )
            )
            assert len(cand_files) == 1
            endpoint_path = cand_files[0]
            output_dir = os.path.join(
                label_base_dir,
                "reduced",
                split_key,
                label_key,
                "{}To{}Hours".format(lhours, rhours),
            )
        else:
            patient_path = os.path.join(
                imputed_base_dir, split_key, "batch_{}.h5".format(batch_idx)
            )
            cand_files = glob.glob(
                os.path.join(endpoint_base_dir, "endpoints_{}_*.h5".format(batch_idx))
            )
            assert len(cand_files) == 1
            endpoint_path = cand_files[0]
            output_dir = os.path.join(
                label_base_dir,
                split_key,
                label_key,
                "{}To{}Hours".format(lhours, rhours),
            )

        if not os.path.exists(patient_path):
            print(
                "WARNING: Patient {} does not exists, skipping...".format(pid),
                flush=True,
            )
            n_skipped_patients += 1
            continue

        try:
            df_endpoint = pd.read_hdf(
                endpoint_path, mode="r", where="PatientID={}".format(pid)
            )
        except:
            print(
                "WARNING: Issue while reading endpoints of patient {}".format(pid),
                flush=True,
            )
            n_skipped_patients += 1
            continue

        df_pat = pd.read_hdf(patient_path, mode="r", where="PatientID={}".format(pid))

        if df_pat.shape[0] == 0 or df_endpoint.shape[0] == 0:
            print(
                "WARNING: Empty endpoints or empty imputed data in patient {}".format(
                    pid
                ),
                flush=True,
            )
            n_skipped_patients += 1
            continue

        if not is_df_sorted(df_endpoint, "Datetime"):
            df_endpoint = df_endpoint.sort_values(by="Datetime", kind="mergesort")

        df_label = tf_model.transform(df_pat, df_endpoint, pid=pid)

        if df_label is None:
            print(
                "WARNING: Label could not be created for PID: {}".format(pid),
                flush=True,
            )
            n_skipped_patients += 1
            continue

        assert df_label.shape[0] == df_pat.shape[0]
        output_path = os.path.join(output_dir, "batch_{}.h5".format(batch_idx))

        if first_write:
            append_mode = False
            open_mode = "w"
        else:
            append_mode = True
            open_mode = "a"

        if not configs["debug_mode"]:
            df_label.to_hdf(
                output_path,
                key=configs["label_dset_id"],
                complevel=configs["hdf_comp_level"],
                complib=configs["hdf_comp_alg"],
                format="table",
                append=append_mode,
                mode=open_mode,
                data_columns=["PatientID"],
            )

        gc.collect()
        first_write = False

        if (pidx + 1) % 20 == 0 and configs["verbose"]:
            print(
                "Progress for batch {}: {:.2f} %".format(
                    batch_idx, (pidx + 1) / len(selected_pids) * 100
                ),
                flush=True,
            )
            print("Number of skipped patients: {}".format(n_skipped_patients))


def parse_cmd_args():
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--version", type=str, required=True, help="Version to run")
    parser.add_argument(
        "--run_mode",
        default="INTERACTIVE",
        help="Should the code be run in cluster mode?",
    )
    parser.add_argument(
        "--split_key",
        default="random_5",
        help="For which split should labels be produced?",
    )
    parser.add_argument(
        "--data_mode",
        default="reduced",
        help="Which data version should be used? [non_reduced or reduced]",
    )
    parser.add_argument(
        "--label_key",
        default="AllLabels",
        help="Which label function should be created?",
    )
    parser.add_argument(
        "--lhours",
        type=float,
        default=0,
        help="Left boundary of the future horizon in hours",
    )
    parser.add_argument(
        "--rhours",
        type=float,
        default=8,
        help="Right boundary of the future horizon in hours",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Should status messages be printed?",
    )
    parser.add_argument(
        "--batch_idx",
        type=int,
        default=41,
        help="On which batch should this process operate?",
    )
    parser.add_argument(
        "--label_dset_id", default="data", help="Data-set ID for labels"
    )
    parser.add_argument(
        "--hdf_comp_level",
        type=int,
        default=5,
        help="HDF compression level for output files",
    )
    parser.add_argument(
        "--hdf_comp_alg", default="blosc:lz4", help="HDF compression algorithm"
    )
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        default=False,
        help="Debug mode for testing, no output created to file-system",
    )
    parser.add_argument(
        "--dataset",
        default="mimic",
        help="For which data-set should label generation be performed?",
    )

    args = parser.parse_args()
    configs = vars(args)

    # Input paths
    configs["data_split_binary"] = (
        f"/data/qmia/mimiciii/validation/misc_derived/split_{configs['version']}.pickle"
    )
    configs["mimic_pid_batch_map_binary"] = (
        f"/data/qmia/mimiciii/validation/external_validation/misc_derived/id_lists/chunks_{configs['version']}.pickle"
    )
    configs["mimic_all_pid_list_path"] = (
        f"/data/qmia/mimiciii/validation/external_validation/pids_with_endpoint_data.csv.{configs['version']}"
    )
    configs["mimic_imputed_path"] = (
        f"/data/qmia/mimiciii/validation/external_validation/imputed/imputed_{configs['version']}"
    )
    configs["mimic_endpoints_dir"] = (
        f"/data/qmia/mimiciii/validation/external_validation/endpoints/{configs['version']}"
    )

    # Output paths
    configs["mimic_output_base_dir"] = (
        f"/data/qmia/mimiciii/validation/external_validation/labels/targets_{configs['version']}"
    )
    os.makedirs(configs["mimic_output_base_dir"], exist_ok=True)
    configs["log_dir"] = (
        f"/data/qmia/mimiciii/validation/external_validation/misc_derived/{configs['version']}/log"
    )
    os.makedirs(configs["log_dir"], exist_ok=True)

    return configs


if __name__ == "__main__":
    configs = parse_cmd_args()
    split_key = configs["split_key"]
    reduced_str = configs["run_mode"]
    label_key = configs["label_key"]
    lhours = configs["lhours"]
    rhours = configs["rhours"]
    batch_idx = configs["batch_idx"]

    if configs["run_mode"] == "CLUSTER":
        sys.stdout = open(
            os.path.join(
                configs["log_dir"],
                "LABELGEN_{}_{}_{}_{}_{}_{}.stdout".format(
                    split_key, reduced_str, label_key, lhours, rhours, batch_idx
                ),
            ),
            "w",
        )
        sys.stderr = open(
            os.path.join(
                configs["log_dir"],
                "LABELGEN_{}_{}_{}_{}_{}_{}.stderr".format(
                    split_key, reduced_str, label_key, lhours, rhours, batch_idx
                ),
            ),
            "w",
        )

    label_gen(configs)
