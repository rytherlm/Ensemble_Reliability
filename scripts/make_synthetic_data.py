# Source: Synthetic data generator for model_sweep compatibility

""" ***************************************************************************
# * File Description:                                                         *
# * Generate synthetic data compatible with model_sweep.py                     *
# *                                                                           *
# * The contents of this script are:                                          *
# * 1. Importing Libraries                                                    *
# * 2. Configurations                                                         *
# * 3. Helper Functions                                                       *
# * 4. Generate Data                                                          *
# * 5. Write Outputs                                                          *
# *                                                                           *
# * --------------------------------------------------------------------------*
# * AUTHORS(S): Aidan Ryther - arr9180@g.rit.edu                              *
# * --------------------------------------------------------------------------*
# * DATE CREATED: 2026-01-21                                                  *
# * --------------------------------------------------------------------------*
# * NOTES: None                                                               *
# * ************************************************************************"""


###############################################################################
#                          1. Importing Libraries                             #
###############################################################################
# For reading, writing, and generating data
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

import os


###############################################################################
#                             2. Configurations                               #
###############################################################################
SEED = 42
POSITIVE_RATE = 0.20

N_FEATURES = 30
N_INFORMATIVE = 5
N_REDUNDANT = 15
N_REPEATED = 5
N_NOISE = 5

TRAIN_SAMPLES = 1000
TEST_ID_SAMPLES = 400
TEST_OOD_1_SAMPLES = 400
TEST_OOD_2_SAMPLES = 400

TRAIN_PATIENTS = 200
TEST_PATIENTS = 150

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(
    BASE_DIR, "data", "cleaned", "selected_features_f1_final_new_common_nate.txt"
)
TRAIN_PATH = os.path.join(BASE_DIR, "data", "data_ood_id", "train_data.csv")
TEST_OOD_1_PATH = os.path.join(BASE_DIR, "data", "data_ood_id", "test_set_ood_1.csv")
TEST_OOD_2_PATH = os.path.join(BASE_DIR, "data", "data_ood_id", "test_set_ood_2.csv")
TEST_ID_PATH = os.path.join(BASE_DIR, "data", "data_ood_id", "test_set_id.csv")

FEATURE_NAMES = [f"feature_{ii:02d}" for ii in range(1, N_FEATURES + 1)]
ARTERIES = ["left_circumflex", "right_coronary_artery", "left_anterior_descending"]


###############################################################################
#                             3. Helper Functions                             #
###############################################################################
def build_patient_ids(rng, n_samples, n_patients, start_id):
    base_ids = np.arange(start_id, start_id + n_patients)
    patient_ids = np.repeat(base_ids, 2)
    if n_samples > patient_ids.size:
        extra = rng.choice(base_ids, size=n_samples - patient_ids.size, replace=True)
        patient_ids = np.concatenate([patient_ids, extra])
    else:
        patient_ids = rng.choice(patient_ids, size=n_samples, replace=False)
    rng.shuffle(patient_ids)
    return patient_ids


def make_artery_columns(rng, y):
    arteries = np.zeros((len(y), 3), dtype=int)
    positive_idx = np.flatnonzero(y == 1)
    if positive_idx.size > 0:
        choices = rng.integers(0, 3, size=positive_idx.size)
        arteries[positive_idx, choices] = 1
    return arteries


def make_dataset(
    seed,
    n_samples,
    n_patients,
    patient_id_start,
    class_sep,
    flip_y,
    mean_shift,
    noise_scale,
    set_name,
):
    rng = np.random.default_rng(seed)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=N_REDUNDANT,
        n_repeated=N_REPEATED,
        n_clusters_per_class=2,
        weights=[1.0 - POSITIVE_RATE, POSITIVE_RATE],
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=seed,
    )

    if mean_shift != 0.0:
        shift = np.zeros(N_FEATURES, dtype=float)
        shift[:10] = mean_shift
        X = X + shift

    if noise_scale > 0.0:
        X = X + rng.normal(0.0, noise_scale, size=X.shape)

    patient_ids = build_patient_ids(rng, n_samples, n_patients, patient_id_start)
    ages = rng.normal(60.0, 12.0, size=n_samples).clip(18.0, 90.0)
    sexes = rng.integers(0, 2, size=n_samples)
    arteries = make_artery_columns(rng, y)

    filenames = [f"record_{set_name}_{ii:05d}.dat" for ii in range(n_samples)]

    columns = {
        "filename": filenames,
        "patient_id": patient_ids.astype(int),
    }
    for idx, name in enumerate(FEATURE_NAMES):
        columns[name] = X[:, idx].astype(float)
    columns["age"] = ages.astype(float)
    columns["sex"] = sexes.astype(int)
    columns[ARTERIES[0]] = arteries[:, 0]
    columns[ARTERIES[1]] = arteries[:, 1]
    columns[ARTERIES[2]] = arteries[:, 2]
    columns["occuluded_artery"] = y.astype(int)

    df = pd.DataFrame(columns)
    ordered_columns = (
        ["filename", "patient_id"]
        + FEATURE_NAMES
        + ["age", "sex"]
        + ARTERIES
        + ["occuluded_artery"]
    )
    df = df[ordered_columns]

    numeric_cols = FEATURE_NAMES + ["age", "sex"] + ARTERIES + ["occuluded_artery"]
    if not np.isfinite(df[numeric_cols].to_numpy()).all():
        raise ValueError("Non-finite values found in generated data.")

    return df


###############################################################################
#                             4. Generate Data                                #
###############################################################################
def main():
    if N_FEATURES != (N_INFORMATIVE + N_REDUNDANT + N_REPEATED + N_NOISE):
        raise ValueError("Feature counts do not sum to N_FEATURES.")

    if TRAIN_SAMPLES < 2 * TRAIN_PATIENTS:
        raise ValueError("TRAIN_SAMPLES must be at least 2 * TRAIN_PATIENTS.")

    if TEST_ID_SAMPLES < 2 * TEST_PATIENTS:
        raise ValueError("TEST_ID_SAMPLES must be at least 2 * TEST_PATIENTS.")

    train_df = make_dataset(
        seed=SEED,
        n_samples=TRAIN_SAMPLES,
        n_patients=TRAIN_PATIENTS,
        patient_id_start=1000,
        class_sep=1.0,
        flip_y=0.02,
        mean_shift=0.0,
        noise_scale=0.0,
        set_name="train",
    )

    test_id_df = make_dataset(
        seed=SEED + 1,
        n_samples=TEST_ID_SAMPLES,
        n_patients=TEST_PATIENTS,
        patient_id_start=5000,
        class_sep=1.0,
        flip_y=0.02,
        mean_shift=0.0,
        noise_scale=0.0,
        set_name="id",
    )

    test_ood_1_df = make_dataset(
        seed=SEED + 2,
        n_samples=TEST_OOD_1_SAMPLES,
        n_patients=TEST_PATIENTS,
        patient_id_start=7000,
        class_sep=1.0,
        flip_y=0.02,
        mean_shift=1.5,
        noise_scale=0.0,
        set_name="ood1",
    )

    test_ood_2_df = make_dataset(
        seed=SEED + 3,
        n_samples=TEST_OOD_2_SAMPLES,
        n_patients=TEST_PATIENTS,
        patient_id_start=9000,
        class_sep=0.8,
        flip_y=0.05,
        mean_shift=0.0,
        noise_scale=1.5,
        set_name="ood2",
    )

    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)

    with open(FEATURES_PATH, "w") as f:
        f.write("\n".join(FEATURE_NAMES))

    train_df.to_csv(TRAIN_PATH, index=False)
    test_ood_1_df.to_csv(TEST_OOD_1_PATH, index=False)
    test_ood_2_df.to_csv(TEST_OOD_2_PATH, index=False)
    test_id_df.to_csv(TEST_ID_PATH, index=False)

    written_paths = [
        FEATURES_PATH,
        TRAIN_PATH,
        TEST_OOD_1_PATH,
        TEST_OOD_2_PATH,
        TEST_ID_PATH,
    ]
    written_paths = [os.path.relpath(path, BASE_DIR) for path in written_paths]
    print("Synthetic data written:")
    for path in written_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
