# --- internal paths --- #
mimic_root_dir = "/data/qmia/mimiciii/"
root_dir = "/data/qmia/mimiciii/validation/"

# --- all about mimic --- #
source_data = mimic_root_dir
derived = mimic_root_dir + "derived_qmia/"
chartevents_path = mimic_root_dir + "CHARTEVENTS.csv"
labevents_path = mimic_root_dir + "LABEVENTS.csv"
outputevents_path = mimic_root_dir + "OUTPUTEVENTS.csv"
inputevents_cv_path = mimic_root_dir + "INPUTEVENTS_CV.csv"
inputevents_mv_path = mimic_root_dir + "INPUTEVENTS_MV.csv"
datetimeevents_path = mimic_root_dir + "DATETIMEEVENTS.csv"
procedureevents_mv_path = mimic_root_dir + "PROCEDUREEVENTS_MV.csv"
admissions_path = mimic_root_dir + "ADMISSIONS.csv"
patients_path = mimic_root_dir + "PATIENTS.csv"
icustays_path = mimic_root_dir + "ICUSTAYS.csv"
services_path = mimic_root_dir + "SERVICES.csv"
csv_folder = derived + "derived_csvs/"

# --- all about our data on leomed --- #
validation_dir = root_dir + "external_validation/"
misc_dir = root_dir + "misc_derived/qmia/"
vis_dir = validation_dir + "vis/"

D_ITEMS_path = validation_dir + "ref_lists/D_ITEMS.csv"
D_LABITEMS_path = validation_dir + "ref_lists/D_LABITEMS.csv"
GDOC_path = validation_dir + "ref_lists/mimic_vars.csv"

chunks_file = validation_dir + "chunks.csv"

csvs_dir = validation_dir + "csvs/"
hdf5_dir = validation_dir + "hdf5/"
# mimic is always reduced
merged_dir = validation_dir + "merged/"
endpoints_dir = validation_dir + "endpoints/"

predictions_dir = validation_dir + "predictions/"

id2string = root_dir + "misc_derived/visualisation/id2string_v6.npy"
mid2string = root_dir + "misc_derived/visualisation/mid2string_v6.npy"
