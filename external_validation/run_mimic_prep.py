#!/usr/bin/env ipython
# author: stephanie hyland
# this is the wrapper for running the mimic preprocessing
import argparse

import extract_data_from_mimic as em

parser = argparse.ArgumentParser(description="Run MIMIC preprocessing.")
parser.add_argument(
    "--version", type=str, default="181023", help="Version of the preprocessing to run"
)
args = parser.parse_args()

version = args.version

# build static
em.build_static_table(version=version)
# subset all the tables
em.meta_subset_tables(version=version)
# convert to hdf5
em.trim_unify_csvs(version=version)
# pivot and merge the hdf5s
em.merge_tables(version=version)
# remove
em.remove_impossible_values(version=version)

# chunk
import chunk_val

chunk_val.build_chunk_file(version=version)
chunk_val.chunk_up_merged_file(version=version)

# next, do endpoints etc...
