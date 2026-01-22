# Source: Local pipeline entry point for the capstone

""" ***************************************************************************
# * File Description:                                                         *
# * Pipeline entry point for capstone stages                                  *
# *                                                                           *
# * The contents of this script are:                                          *
# * 1. Importing Libraries                                                    *
# * 2. Configurations                                                         *
# * 3. Functions                                                              *
# * 4. Main                                                                   *
# *                                                                           *
# * --------------------------------------------------------------------------*
# * AUTHORS(S): Aidan Ryther                                                  *
# * --------------------------------------------------------------------------*
# * DATE CREATED: 2026-01-22                                                  *
# * --------------------------------------------------------------------------*
# * NOTES: None                                                               *
# * ************************************************************************"""


###############################################################################
#                          1. Importing Libraries                             #
###############################################################################
import os
import subprocess
import sys

import joblib
import pandas as pd


###############################################################################
#                             2. Configurations                               #
###############################################################################
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_SWEEP_PATH = os.path.join(ROOT_DIR, "model_sweep.py")
DATA_SCRIPT_PATH = os.path.join(ROOT_DIR, "generate_data.py")

FEATURES_PATH = os.path.join(ROOT_DIR, "data", "cleaned", "selected_features_f1_final_new_common_nate.txt")
TRAIN_PATH = os.path.join(ROOT_DIR, "data", "data_ood_id", "train_data.csv")
TEST_OOD_1_PATH = os.path.join(ROOT_DIR, "data", "data_ood_id", "test_set_ood_1.csv")
TEST_OOD_2_PATH = os.path.join(ROOT_DIR, "data", "data_ood_id", "test_set_ood_2.csv")
TEST_ID_PATH = os.path.join(ROOT_DIR, "data", "data_ood_id", "test_set_id.csv")

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
ACCURACY_PATH = os.path.join(RESULTS_DIR, "accuracy_summary.json")

DATASETS = [
    ("test_set_ood_1", TEST_OOD_1_PATH),
    ("test_set_ood_2", TEST_OOD_2_PATH),
    ("test_set_id", TEST_ID_PATH)
]

PROBABILITY_THRESHOLD = 0.5
OUTPUT_COLUMNS = ["filename","patient_id","left_circumflex","right_coronary_artery","left_anterior_descending","occuluded_artery"]

req_files = [
    FEATURES_PATH,
    TRAIN_PATH,
    TEST_OOD_1_PATH,
    TEST_OOD_2_PATH,
    TEST_ID_PATH
]

###############################################################################
#                             3. Functions                             #
###############################################################################

def ensure_inputs():
    missing = [path for path in req_files if not os.path.exists(path)]
    if not missing:
        print("Data generation skipped.")
        return

    result = subprocess.run([sys.executable, DATA_SCRIPT_PATH], cwd=ROOT_DIR)
    if result.returncode != 0:
        print("ERROR: PTB-XL data generation failed.")
        sys.exit(result.returncode)

    missing = [path for path in req_files if not os.path.exists(path)]
    if missing:
        print("ERROR: missing required files after generation:")
        for path in missing:
            print(f"- {path}")
        sys.exit(1)
    print("Data generation ran.")


def run_model_sweep():
    result = subprocess.run([sys.executable, MODEL_SWEEP_PATH], cwd=ROOT_DIR)
    if result.returncode != 0:
        print("ERROR: model sweep failed.")
        sys.exit(result.returncode)
    print("Model sweep completed.")


def export_probabilities():
    accuracy_rows = []
    bundles = []
    if os.path.isdir(RESULTS_DIR):
        for dirpath, _, filenames in os.walk(RESULTS_DIR):
            for filename in filenames:
                if filename != "final_model_f1_new_overlap.joblib":
                    continue
                model_name = os.path.basename(dirpath)
                bundles.append((model_name, os.path.join(dirpath, filename)))

    bundles.sort(key=lambda item: item[0])
    if not bundles:
        print("ERROR: no model bundles found under results/")
        sys.exit(1)

    for model_name, bundle_path in bundles:
        bundle = joblib.load(bundle_path)
        model = bundle["model"]
        scaler = bundle["scaler"]
        cols = bundle["cols"]
        features = bundle["features"]

        for split_name, split_path in DATASETS:
            df = pd.read_csv(split_path)
            X_scaled = scaler.transform(df[cols])
            X_scaled = pd.DataFrame(X_scaled, columns=cols, index=df.index)
            probabilities = model.predict_proba(X_scaled[features])[:, 1]

            output = df[OUTPUT_COLUMNS].copy()
            output["mi_probability"] = probabilities
            output["mi_pred"] = (output["mi_probability"] >= PROBABILITY_THRESHOLD).astype(
                int
            )

            out_dir = os.path.join(RESULTS_DIR, model_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"probabilities_{split_name}.csv")
            output.to_csv(out_path, index=False)
            accuracy = (output["mi_pred"] == output["occuluded_artery"]).mean() * 100.0
            print(f"{model_name} {split_name} accuracy: {accuracy:.2f}%")
            accuracy_rows.append(
                {"model": model_name, "split": split_name, "accuracy_percent": accuracy}
            )

    if accuracy_rows:
        accuracy_df = pd.DataFrame(accuracy_rows)
        accuracy_df.to_json(ACCURACY_PATH, orient="records", indent=2)
    print("Probability exports done.")





###############################################################################
#                                  4. Main                                    #
###############################################################################
def main():
    ensure_inputs()
    run_model_sweep()
    export_probabilities()


if __name__ == "__main__":
    main()
