import glob
import os

import pandas as pd

MIMIC_PIDS_PATH = (
    "/data/qmia/mimiciii/validation/external_validation/endpoints/181103/reduced"
)
OUT_PATH = "/data/qmia/mimiciii/validation/external_validation"

# cycle through all of the h5 files in the directory (there could be any number) and get a unique list of pids
pids = set()

files = os.path.join(MIMIC_PIDS_PATH, "*.h5")

for f in glob.glob(files):
    df = pd.read_hdf(f)
    pids.update(df.PatientID.unique())

# write the pids to a file
with open(os.path.join(OUT_PATH, "pids_with_endpoint_data.csv.181103"), "w") as f:
    for pid in pids:
        f.write(str(pid) + "\n")

print(
    "Wrote {} pids to {}".format(
        len(pids), os.path.join(OUT_PATH, "pids_with_endpoint_data.csv.181103")
    )
)
