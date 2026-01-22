# Source: PTB-XL data ingestion for model_sweep compatibility

""" ***************************************************************************
# * File Description:                                                         *
# * Generate PTB-XL derived feature data compatible with model_sweep.py        *
# *                                                                           *
# * The contents of this script are:                                          *
# * 1. Importing Libraries                                                    *
# * 2. Configurations                                                         *
# * 3. Functions                                                              *
# * 4. Generate Data                                                          *                                                      *
# *                                                                           *
# * --------------------------------------------------------------------------*
# * AUTHORS(S): Aidan Ryther - arr9180@g.rit.edu                              *
# * --------------------------------------------------------------------------*
# * DATE CREATED: 2026-01-23                                                  *
# * --------------------------------------------------------------------------*
# * NOTES: None                                                               *
# * ************************************************************************"""


###############################################################################
#                          1. Importing Libraries                             #
###############################################################################
import ast
import os
import sys

import numpy as np
import pandas as pd
import wfdb


###############################################################################
#                             2. Configurations                               #
###############################################################################
N_FEATURES = 30
MAX_PER_SPLIT = 500

FEATURE_NAMES = [f"feature_{ii:02d}" for ii in range(1, N_FEATURES + 1)]
MI_CODES = {"AMI", "ASMI", "IMI", "LMI"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PTBXL_DIR = os.path.join(BASE_DIR, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
PTBXL_META_PATH = os.path.join(PTBXL_DIR, "ptbxl_database.csv")

FEATURES_PATH = os.path.join(
    BASE_DIR, "data", "cleaned", "selected_features_f1_final_new_common_nate.txt"
)
TRAIN_PATH = os.path.join(BASE_DIR, "data", "data_ood_id", "train_data.csv")
TEST_OOD_1_PATH = os.path.join(BASE_DIR, "data", "data_ood_id", "test_set_ood_1.csv")
TEST_OOD_2_PATH = os.path.join(BASE_DIR, "data", "data_ood_id", "test_set_ood_2.csv")
TEST_ID_PATH = os.path.join(BASE_DIR, "data", "data_ood_id", "test_set_id.csv")

ORDERED_COLUMNS = (
    ["filename", "patient_id"]
    + FEATURE_NAMES
    + ["age", "sex"]
    + ["left_circumflex", "right_coronary_artery", "left_anterior_descending"]
    + ["occuluded_artery"]
)


###############################################################################
#                             3. Functions                                    #
###############################################################################
def parse_scp_codes(value):
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def build_label_columns(df):
    codes = df["scp_codes"].apply(parse_scp_codes)
    code_sets = codes.apply(lambda item: set(item.keys()))

    df["occuluded_artery"] = code_sets.apply(
        lambda codeset: int(any(code in codeset for code in MI_CODES))
    )
    df["left_anterior_descending"] = code_sets.apply(
        lambda codeset: int("AMI" in codeset or "ASMI" in codeset)
    )
    df["right_coronary_artery"] = code_sets.apply(lambda codeset: int("IMI" in codeset))
    df["left_circumflex"] = code_sets.apply(lambda codeset: int("LMI" in codeset))
    return df


def normalize_record_path(record_value):
    if record_value is None:
        return None
    record_str = str(record_value).strip()
    if record_str.endswith(".dat") or record_str.endswith(".hea"):
        record_str = os.path.splitext(record_str)[0]
    return record_str


def load_signal(record_path):
    if record_path is None:
        raise ValueError("Missing record path in PTB-XL metadata.")
    full_path = os.path.join(PTBXL_DIR, record_path)
    try:
        signals, _ = wfdb.rdsamp(full_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read PTB-XL record: {record_path}") from exc
    if signals.ndim == 1:
        signals = signals[:, None]
    if signals.shape[1] < 12:
        raise ValueError(f"Expected 12 leads, got {signals.shape[1]} for {record_path}")
    return signals[:, :12]


def compute_features(signals):
    means = np.nanmean(signals, axis=0)
    stds = np.nanstd(signals, axis=0)

    abs_signal = np.abs(signals)
    mean_abs = np.nanmean(abs_signal)
    std_abs = np.nanstd(abs_signal)
    max_abs = np.nanmax(abs_signal)
    rms = float(np.sqrt(np.nanmean(np.square(signals))))

    lead_ptp = np.nanmax(signals, axis=0) - np.nanmin(signals, axis=0)
    mean_ptp = np.nanmean(lead_ptp)
    std_ptp = np.nanstd(lead_ptp)

    features = np.concatenate(
        [
            means,
            stds,
            np.array([mean_abs, std_abs, max_abs, rms, mean_ptp, std_ptp], dtype=float),
        ]
    )
    if features.shape[0] != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} features, got {features.shape[0]}")
    return features


def build_dataset(df, filename_col):
    rows = []
    for row in df.itertuples(index=False):
        record_rel = normalize_record_path(getattr(row, filename_col))
        signals = load_signal(record_rel)
        features = compute_features(signals)

        row_data = {
            "filename": record_rel,
            "patient_id": int(row.patient_id),
            "age": float(row.age),
            "sex": int(row.sex),
            "left_circumflex": int(row.left_circumflex),
            "right_coronary_artery": int(row.right_coronary_artery),
            "left_anterior_descending": int(row.left_anterior_descending),
            "occuluded_artery": int(row.occuluded_artery),
        }
        for name, value in zip(FEATURE_NAMES, features):
            row_data[name] = float(value)
        rows.append(row_data)

    df_out = pd.DataFrame(rows)
    return df_out[ORDERED_COLUMNS]


###############################################################################
#                             4. Generate Data                                #
###############################################################################
def main():
    if not os.path.isdir(PTBXL_DIR):
        print(f"ERROR: PTB-XL directory not found: {PTBXL_DIR}")
        sys.exit(1)

    if not os.path.exists(PTBXL_META_PATH):
        print(f"ERROR: PTB-XL metadata not found: {PTBXL_META_PATH}")
        sys.exit(1)

    print("Generating PTB-XL data...")

    meta = pd.read_csv(PTBXL_META_PATH)
    required_cols = [
        "patient_id",
        "age",
        "sex",
        "scp_codes",
        "strat_fold",
        "filename_lr",
        "filename_hr",
    ]
    missing = [col for col in required_cols if col not in meta.columns]
    if missing:
        print(f"ERROR: Missing columns in ptbxl_database.csv: {', '.join(missing)}")
        sys.exit(1)

    meta["age"] = pd.to_numeric(meta["age"], errors="coerce")
    meta["sex"] = pd.to_numeric(meta["sex"], errors="coerce")
    age_median = float(meta["age"].median())
    meta["age"] = meta["age"].fillna(age_median)
    meta["sex"] = meta["sex"].fillna(0).astype(int)
    meta["patient_id"] = (
        pd.to_numeric(meta["patient_id"], errors="coerce").fillna(0).astype(int)
    )

    meta = build_label_columns(meta)

    train_meta = meta[meta["strat_fold"].isin(range(1, 9))].reset_index(drop=True)
    test_id_meta = meta[meta["strat_fold"] == 9].reset_index(drop=True)
    fold10_meta = meta[meta["strat_fold"] == 10].reset_index(drop=True)

    if MAX_PER_SPLIT is not None:
        train_meta = train_meta.sample(
            n=min(len(train_meta), MAX_PER_SPLIT), random_state=42
        ).reset_index(drop=True)
        test_id_meta = test_id_meta.sample(
            n=min(len(test_id_meta), MAX_PER_SPLIT), random_state=42
        ).reset_index(drop=True)
        fold10_meta = fold10_meta.sample(
            n=min(len(fold10_meta), MAX_PER_SPLIT), random_state=42
        ).reset_index(drop=True)


    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)

    with open(FEATURES_PATH, "w") as f:
        f.write("\n".join(FEATURE_NAMES))

    train_df = build_dataset(train_meta, "filename_lr")
    test_id_df = build_dataset(test_id_meta, "filename_lr")
    test_ood_1_df = build_dataset(fold10_meta, "filename_lr")
    test_ood_2_df = build_dataset(fold10_meta, "filename_hr")

    train_df.to_csv(TRAIN_PATH, index=False)
    test_id_df.to_csv(TEST_ID_PATH, index=False)
    test_ood_1_df.to_csv(TEST_OOD_1_PATH, index=False)
    test_ood_2_df.to_csv(TEST_OOD_2_PATH, index=False)

    written_paths = [
        FEATURES_PATH,
        TRAIN_PATH,
        TEST_ID_PATH,
        TEST_OOD_1_PATH,
        TEST_OOD_2_PATH,
    ]
    written_paths = [os.path.relpath(path, BASE_DIR) for path in written_paths]
    print("Done. Wrote:")
    for path in written_paths:
        print(f"- {path}")

if __name__ == "__main__":
    main()
