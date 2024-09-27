import argparse
import glob
import os

import pandas as pd

parser = argparse.ArgumentParser(description="Generate list of PIDs with endpoints.")
parser.add_argument(
    "--version", type=str, required=True, help="Version of the preprocessing to run"
)
args = parser.parse_args()
version = args.version

MIMIC_PIDS_PATH = (
    f"/data/qmia/mimiciii/validation/external_validation/endpoints/{version}/reduced"
)
OUT_PATH = "/data/qmia/mimiciii/validation/external_validation"

# cycle through all of the h5 files in the directory (there could be any number) and get a unique list of pids
pids = set()

files = os.path.join(MIMIC_PIDS_PATH, "*.h5")

for f in glob.glob(files):
    df = pd.read_hdf(f)
    pids.update(df.PatientID.unique())

# write the pids to a file
with open(os.path.join(OUT_PATH, f"pids_with_endpoint_data.csv.{version}"), "w") as f:
    for pid in pids:
        f.write(str(pid) + "\n")

print(
    "Wrote {} pids to {}".format(
        len(pids), os.path.join(OUT_PATH, f"pids_with_endpoint_data.csv.{version}")
    )
)
