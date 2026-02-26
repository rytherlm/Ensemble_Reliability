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
import json
import os
import shutil
import subprocess
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold


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
VALIDATION_PATH = os.path.join(ROOT_DIR, "data", "data_ood_id", "validation_data.csv")

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODEL_OUTPUTS_DIR = os.path.join(RESULTS_DIR, "model_outputs")
ACCURACY_PATH = os.path.join(RESULTS_DIR, "accuracy_summary.json")
ENSEMBLES_DIR = os.path.join(RESULTS_DIR, "ensembles")
CALIBRATION_DIR = os.path.join(RESULTS_DIR, "calibration")
ABSTAIN_DIR = os.path.join(RESULTS_DIR, "abstain_uncertainty")
FEATURE_IMPORTANCE_DIR = os.path.join(RESULTS_DIR, "feature_importance")
FINAL_PLOTS_DIR = os.path.join(RESULTS_DIR, "final_plots")

DATASETS = [
    ("validation", VALIDATION_PATH),
    ("test_set_ood_1", TEST_OOD_1_PATH),
    ("test_set_ood_2", TEST_OOD_2_PATH),
    ("test_set_id", TEST_ID_PATH)
]

PROBABILITY_THRESHOLD = 0.5
TOP_K_MODELS = 5
CALIBRATION_SPLIT = "validation"
CALIBRATION_BINS = 10
ABSTAIN_TARGET_ERROR_RATE = 0.10
ABSTAIN_THRESHOLD_STEPS = 101
ABSTAIN_ENSEMBLE_PRIORITY = [
    "stacking_logistic",
    "performance_weighted_soft",
    "soft_voting",
    "k_filter_voting",
    "hard_voting",
]
FEATURE_SUBSET_SIZE = 15
FEATURE_VOTE_TOP_FRACTION = 0.25
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


def ensure_validation_split():
    if os.path.exists(VALIDATION_PATH):
        print("Validation split exists.")
        return

    if not os.path.exists(TRAIN_PATH):
        print("ERROR: train_data.csv missing; cannot create validation split.")
        sys.exit(1)

    train_df = pd.read_csv(TRAIN_PATH)
    required_columns = {"patient_id", "occuluded_artery"}
    if not required_columns.issubset(train_df.columns):
        print("ERROR: train_data.csv missing required columns for validation split.")
        sys.exit(1)

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    y = train_df["occuluded_artery"]
    groups = train_df["patient_id"]

    try:
        _, validation_indices = next(splitter.split(train_df, y, groups))
    except ValueError as exc:
        print(f"ERROR: failed to create validation split: {exc}")
        sys.exit(1)

    validation_df = train_df.iloc[validation_indices].copy()
    validation_df.to_csv(VALIDATION_PATH, index=False)
    print("Validation split created.")


def run_model_sweep():
    result = subprocess.run([sys.executable, MODEL_SWEEP_PATH], cwd=ROOT_DIR)
    if result.returncode != 0:
        print("ERROR: model sweep failed.")
        sys.exit(result.returncode)
    print("Model sweep completed.")


def export_probabilities():
    if not os.path.exists(VALIDATION_PATH):
        ensure_validation_split()

    accuracy_rows = []
    bundles = []
    if os.path.isdir(MODEL_OUTPUTS_DIR):
        for model_name in sorted(os.listdir(MODEL_OUTPUTS_DIR)):
            model_dir = os.path.join(MODEL_OUTPUTS_DIR, model_name)
            if not os.path.isdir(model_dir):
                continue
            bundle_path = os.path.join(model_dir, "final_model_f1_new_overlap.joblib")
            if os.path.exists(bundle_path):
                bundles.append((model_name, bundle_path))

    bundles.sort(key=lambda item: item[0])
    if not bundles:
        print("ERROR: no model bundles found under results/model_outputs/")
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

            out_dir = os.path.join(MODEL_OUTPUTS_DIR, model_name)
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



def hard_voting_ensemble(probabilities):
    if isinstance(probabilities, dict):
        prob_df = pd.DataFrame(probabilities)
    else:
        prob_df = pd.DataFrame(probabilities)

    if prob_df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=int)

    probs = prob_df.to_numpy()
    ensemble_probability = probs.mean(axis=1)
    votes = probs >= PROBABILITY_THRESHOLD
    vote_sum = votes.sum(axis=1)
    n_models = probs.shape[1]
    ensemble_pred = (vote_sum > (n_models / 2)).astype(int)

    if n_models % 2 == 0:
        tie_mask = vote_sum == (n_models / 2)
        if tie_mask.any():
            ensemble_pred[tie_mask] = (ensemble_probability[tie_mask] >= PROBABILITY_THRESHOLD).astype(int)

    return ensemble_probability, ensemble_pred


def soft_voting_ensemble(probabilities):
    if isinstance(probabilities, dict):
        prob_df = pd.DataFrame(probabilities)
    else:
        prob_df = pd.DataFrame(probabilities)

    if prob_df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=int)

    ensemble_probability = prob_df.mean(axis=1).to_numpy()
    ensemble_pred = (ensemble_probability >= PROBABILITY_THRESHOLD).astype(int)
    return ensemble_probability, ensemble_pred


def performance_weighted_soft_voting_ensemble(probabilities, accuracy_path=ACCURACY_PATH, weight_split=CALIBRATION_SPLIT):
    if isinstance(probabilities, dict):
        prob_df = pd.DataFrame(probabilities)
        model_names = list(probabilities.keys())
    else:
        prob_df = pd.DataFrame(probabilities)
        model_names = list(prob_df.columns)

    if prob_df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=int), {}

    accuracy_rows = []
    if os.path.exists(accuracy_path):
        with open(accuracy_path, "r") as handle:
            accuracy_rows = json.load(handle)

    model_entries = {}
    for row in accuracy_rows:
        model = row.get("model")
        split = row.get("split")
        accuracy = row.get("accuracy_percent")
        if model is None or accuracy is None:
            continue
        model_entries.setdefault(model, []).append((split, accuracy))

    raw_weights = {}
    for model in model_names:
        entries = model_entries.get(model, [])
        split_scores = [acc for split, acc in entries if split == weight_split]
        if split_scores:
            raw_weights[model] = sum(split_scores) / len(split_scores)
        else:
            raw_weights[model] = 0.0

    total = sum(raw_weights.values())
    if total <= 0:
        normalized_weights = {model: 1.0 / len(model_names) for model in model_names}
    else:
        normalized_weights = {model: weight / total for model, weight in raw_weights.items()}

    weight_series = pd.Series(normalized_weights)
    ensemble_probability = prob_df[model_names].mul(weight_series, axis=1).sum(axis=1).to_numpy()
    ensemble_pred = (ensemble_probability >= PROBABILITY_THRESHOLD).astype(int)
    return ensemble_probability, ensemble_pred, normalized_weights


def k_filter_voting_ensemble(probabilities, accuracy_path=ACCURACY_PATH, weight_split=CALIBRATION_SPLIT, k=TOP_K_MODELS):
    if isinstance(probabilities, dict):
        prob_df = pd.DataFrame(probabilities)
        model_names = list(probabilities.keys())
    else:
        prob_df = pd.DataFrame(probabilities)
        model_names = list(prob_df.columns)

    if prob_df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=int), []

    accuracy_rows = []
    if os.path.exists(accuracy_path):
        with open(accuracy_path, "r") as handle:
            accuracy_rows = json.load(handle)

    model_entries = {}
    for row in accuracy_rows:
        model = row.get("model")
        split = row.get("split")
        accuracy = row.get("accuracy_percent")
        if model is None or accuracy is None:
            continue
        model_entries.setdefault(model, []).append((split, accuracy))

    model_scores = {}
    for model in model_names:
        entries = model_entries.get(model, [])
        split_scores = [acc for split, acc in entries if split == weight_split]
        if split_scores:
            model_scores[model] = sum(split_scores) / len(split_scores)
        else:
            model_scores[model] = 0.0

    ranked_models = sorted(model_names, key=lambda name: (-model_scores.get(name, 0.0), name))
    selected_models = ranked_models[: min(k, len(ranked_models))]
    selected_probs = {model: probabilities[model] for model in selected_models} if isinstance(probabilities, dict) else prob_df[selected_models]

    ensemble_probability, ensemble_pred = hard_voting_ensemble(selected_probs)
    return ensemble_probability, ensemble_pred, selected_models


def stacking_logistic_ensemble(
    validation_probabilities, validation_labels, split_probabilities
):
    validation_df = pd.DataFrame(validation_probabilities)
    split_df = pd.DataFrame(split_probabilities)
    if validation_df.empty or split_df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=int), [], PROBABILITY_THRESHOLD

    common_models = [name for name in validation_df.columns if name in split_df.columns]
    if not common_models:
        return pd.Series(dtype=float), pd.Series(dtype=int), [], PROBABILITY_THRESHOLD

    X_validation = validation_df[common_models].to_numpy(dtype=float)
    X_split = split_df[common_models].to_numpy(dtype=float)
    y_validation = np.asarray(validation_labels, dtype=int)
    if len(y_validation) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=int), [], PROBABILITY_THRESHOLD

    if len(np.unique(y_validation)) < 2:
        split_probability = X_split.mean(axis=1)
        split_pred = (split_probability >= PROBABILITY_THRESHOLD).astype(int)
        return split_probability, split_pred, common_models, PROBABILITY_THRESHOLD

    meta_model = LogisticRegression(max_iter=1000, solver="liblinear")
    meta_model.fit(X_validation, y_validation)
    validation_probability = meta_model.predict_proba(X_validation)[:, 1]

    best_threshold = PROBABILITY_THRESHOLD
    best_accuracy = ((validation_probability >= best_threshold) == y_validation).mean()
    for threshold in np.linspace(0.30, 0.70, 41):
        threshold_accuracy = ((validation_probability >= threshold) == y_validation).mean()
        if threshold_accuracy > best_accuracy:
            best_accuracy = threshold_accuracy
            best_threshold = float(threshold)

    split_probability = meta_model.predict_proba(X_split)[:, 1]
    split_pred = (split_probability >= best_threshold).astype(int)
    return split_probability, split_pred, common_models, best_threshold


def run_ensemble_methods():
    if not os.path.isdir(MODEL_OUTPUTS_DIR):
        print("ERROR: results/model_outputs directory not found; skipping ensembles.")
        return

    os.makedirs(ENSEMBLES_DIR, exist_ok=True)

    summary_rows = []
    weights_used = {}
    selected_models = {}
    stacking_info = {}

    model_dirs = [
        name
        for name in os.listdir(MODEL_OUTPUTS_DIR)
        if os.path.isdir(os.path.join(MODEL_OUTPUTS_DIR, name))
    ]
    model_dirs.sort()

    validation_base_df = None
    validation_keys = None
    validation_probabilities = {}
    for model_name in model_dirs:
        validation_path = os.path.join(
            MODEL_OUTPUTS_DIR, model_name, f"probabilities_{CALIBRATION_SPLIT}.csv"
        )
        if not os.path.exists(validation_path):
            continue

        validation_df = pd.read_csv(validation_path)
        if (
            not set(OUTPUT_COLUMNS).issubset(validation_df.columns)
            or "mi_probability" not in validation_df.columns
        ):
            continue

        validation_df = validation_df.sort_values(["patient_id", "filename"]).reset_index(drop=True)
        if validation_base_df is None:
            validation_base_df = validation_df[OUTPUT_COLUMNS].copy()
            validation_keys = validation_base_df[["patient_id", "filename"]].copy().reset_index(drop=True)
        else:
            current_keys = validation_df[["patient_id", "filename"]].copy().reset_index(drop=True)
            if not current_keys.equals(validation_keys):
                continue

        validation_probabilities[model_name] = validation_df["mi_probability"].to_numpy()

    if validation_base_df is not None and validation_probabilities:
        validation_labels = validation_base_df["occuluded_artery"].to_numpy(dtype=int)
    else:
        validation_labels = np.array([], dtype=int)

    for split_name, _ in DATASETS:
        base_df = None
        base_keys = None
        model_probabilities = {}

        for model_name in model_dirs:
            prob_path = os.path.join(MODEL_OUTPUTS_DIR, model_name, f"probabilities_{split_name}.csv")
            if not os.path.exists(prob_path):
                continue

            df = pd.read_csv(prob_path)
            if not set(OUTPUT_COLUMNS).issubset(df.columns) or "mi_probability" not in df.columns:
                print(f"WARNING: {model_name} {split_name} missing required columns; skipping.")
                continue

            df = df.sort_values(["patient_id", "filename"]).reset_index(drop=True)
            if base_df is None:
                base_df = df[OUTPUT_COLUMNS].copy()
                base_keys = base_df[["patient_id", "filename"]].copy().reset_index(drop=True)
            else:
                current_keys = df[["patient_id", "filename"]].copy().reset_index(drop=True)
                if not current_keys.equals(base_keys):
                    print(f"WARNING: {model_name} {split_name} keys do not match base; skipping model.")
                    continue

            model_probabilities[model_name] = df["mi_probability"].to_numpy()

        if base_df is None:
            print(f"ERROR: no probability files found for split {split_name}; skipping.")
            continue

        if not model_probabilities:
            print(f"ERROR: no usable models for split {split_name}; skipping.")
            continue

        if len(model_probabilities) < 2:
            print(f"WARNING: only {len(model_probabilities)} model(s) available for {split_name}; ensembles will match single model.")

        methods = [
            ("hard_voting", hard_voting_ensemble, {}),
            ("soft_voting", soft_voting_ensemble, {}),
            ("performance_weighted_soft", performance_weighted_soft_voting_ensemble, {"accuracy_path": ACCURACY_PATH, "weight_split": CALIBRATION_SPLIT}),
            ("k_filter_voting", k_filter_voting_ensemble, {"accuracy_path": ACCURACY_PATH, "weight_split": CALIBRATION_SPLIT, "k": TOP_K_MODELS}),
            ("stacking_logistic", stacking_logistic_ensemble, {}),
        ]

        for method_name, method_fn, method_kwargs in methods:
            if method_name == "performance_weighted_soft":
                ensemble_probability, ensemble_pred, used_weights = method_fn(model_probabilities, **method_kwargs)
                weights_used[split_name] = used_weights
                models_used = list(used_weights.keys())
            elif method_name == "k_filter_voting":
                ensemble_probability, ensemble_pred, selected = method_fn(model_probabilities, **method_kwargs)
                selected_models[split_name] = selected
                models_used = selected
            elif method_name == "stacking_logistic":
                if len(validation_probabilities) == 0 or len(validation_labels) == 0:
                    print("WARNING: validation probabilities missing for stacking; using soft voting fallback.")
                    ensemble_probability, ensemble_pred = soft_voting_ensemble(model_probabilities)
                    models_used = list(model_probabilities.keys())
                    stacking_info[split_name] = {
                        "models_used": models_used,
                        "threshold": PROBABILITY_THRESHOLD,
                        "fallback": "soft_voting",
                        "calibration_split": CALIBRATION_SPLIT,
                    }
                else:
                    (
                        ensemble_probability,
                        ensemble_pred,
                        models_used,
                        best_threshold,
                    ) = method_fn(validation_probabilities, validation_labels, model_probabilities)
                    if len(models_used) == 0:
                        ensemble_probability, ensemble_pred = soft_voting_ensemble(model_probabilities)
                        models_used = list(model_probabilities.keys())
                        best_threshold = PROBABILITY_THRESHOLD
                        fallback = "soft_voting"
                    else:
                        fallback = None
                    stacking_info[split_name] = {
                        "models_used": models_used,
                        "threshold": round(float(best_threshold), 4),
                        "fallback": fallback,
                        "calibration_split": CALIBRATION_SPLIT,
                    }
            else:
                ensemble_probability, ensemble_pred = method_fn(model_probabilities)
                models_used = list(model_probabilities.keys())

            out_dir = os.path.join(ENSEMBLES_DIR, method_name)
            os.makedirs(out_dir, exist_ok=True)

            output = base_df.copy()
            output["ensemble_probability"] = ensemble_probability
            output["ensemble_pred"] = pd.Series(ensemble_pred, index=output.index).astype(int)
            out_path = os.path.join(out_dir, f"probabilities_{split_name}.csv")
            output.to_csv(out_path, index=False)

            accuracy = (output["ensemble_pred"] == output["occuluded_artery"]).mean() * 100.0
            accuracy = round(accuracy, 2)
            summary_rows.append(
                {
                    "method": method_name,
                    "split": split_name,
                    "accuracy_percent": accuracy,
                    "n_models": len(models_used),
                    "models_used": models_used,
                }
            )

    if summary_rows:
        summary_path = os.path.join(ENSEMBLES_DIR, "ensemble_accuracy_summary.json")
        with open(summary_path, "w") as handle:
            json.dump(summary_rows, handle, indent=2)

    if weights_used:
        weights_path = os.path.join(ENSEMBLES_DIR, "performance_weighted_soft", "weights_used.json")
        with open(weights_path, "w") as handle:
            json.dump(weights_used, handle, indent=2)

    if selected_models:
        selected_path = os.path.join(ENSEMBLES_DIR, "k_filter_voting", "selected_models.json")
        with open(selected_path, "w") as handle:
            json.dump(selected_models, handle, indent=2)

    if stacking_info:
        stacking_path = os.path.join(ENSEMBLES_DIR, "stacking_logistic", "stacking_info.json")
        os.makedirs(os.path.dirname(stacking_path), exist_ok=True)
        with open(stacking_path, "w") as handle:
            json.dump(stacking_info, handle, indent=2)

    print("Ensemble methods completed.")


def run_probability_calibration():
    if not os.path.isdir(RESULTS_DIR):
        print("ERROR: results/ directory not found; skipping calibration.")
        return

    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    split_names = [split_name for split_name, _ in DATASETS]
    summary_rows = []

    def expected_calibration_error(y_true, probabilities, n_bins=CALIBRATION_BINS):
        y_true = np.asarray(y_true, dtype=float)
        probabilities = np.asarray(probabilities, dtype=float)
        if len(probabilities) == 0:
            return 0.0

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(probabilities, bins[1:-1], right=False)
        ece = 0.0

        for bin_index in range(n_bins):
            mask = bin_ids == bin_index
            if not mask.any():
                continue
            bin_probability_mean = probabilities[mask].mean()
            bin_outcome_mean = y_true[mask].mean()
            ece += (mask.sum() / len(probabilities)) * abs(
                bin_outcome_mean - bin_probability_mean
            )

        return float(ece)

    def fit_calibrator(probabilities, y_true):
        y_true = np.asarray(y_true, dtype=int)
        probabilities = np.asarray(probabilities, dtype=float)
        if len(np.unique(y_true)) < 2:
            return None, "identity_single_class"

        calibrator = LogisticRegression(max_iter=1000, solver="lbfgs")
        calibrator.fit(probabilities.reshape(-1, 1), y_true)
        return calibrator, "platt_sigmoid"

    def apply_calibration(probabilities, calibrator):
        probabilities = np.asarray(probabilities, dtype=float)
        probabilities = np.clip(probabilities, 0.0, 1.0)
        if calibrator is None:
            return probabilities
        calibrated = calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]
        return np.clip(calibrated, 0.0, 1.0)

    sources = []
    if os.path.isdir(MODEL_OUTPUTS_DIR):
        for source_name in sorted(os.listdir(MODEL_OUTPUTS_DIR)):
            source_path = os.path.join(MODEL_OUTPUTS_DIR, source_name)
            if not os.path.isdir(source_path):
                continue
            sources.append(
                {
                    "source_type": "model",
                    "source_name": source_name,
                    "source_path": source_path,
                    "probability_column": "mi_probability",
                }
            )

    if os.path.isdir(ENSEMBLES_DIR):
        for source_name in sorted(os.listdir(ENSEMBLES_DIR)):
            source_path = os.path.join(ENSEMBLES_DIR, source_name)
            if not os.path.isdir(source_path):
                continue
            sources.append(
                {
                    "source_type": "ensemble",
                    "source_name": source_name,
                    "source_path": source_path,
                    "probability_column": "ensemble_probability",
                }
            )

    if not sources:
        print("ERROR: no model or ensemble probability sources found; skipping calibration.")
        return

    for source in sources:
        source_type = source["source_type"]
        source_name = source["source_name"]
        source_path = source["source_path"]
        probability_column = source["probability_column"]

        calibration_path = os.path.join(
            source_path, f"probabilities_{CALIBRATION_SPLIT}.csv"
        )
        if not os.path.exists(calibration_path):
            print(
                f"WARNING: {source_type} {source_name} missing {CALIBRATION_SPLIT}; skipping."
            )
            continue

        calibration_df = pd.read_csv(calibration_path)
        required_columns = set(OUTPUT_COLUMNS + [probability_column])
        if not required_columns.issubset(calibration_df.columns):
            print(
                f"WARNING: {source_type} {source_name} missing required columns; skipping."
            )
            continue

        calibrator, calibration_method = fit_calibrator(
            calibration_df[probability_column].to_numpy(),
            calibration_df["occuluded_artery"].to_numpy(),
        )

        out_dir = os.path.join(CALIBRATION_DIR, f"{source_type}s", source_name)
        os.makedirs(out_dir, exist_ok=True)

        for split_name in split_names:
            split_path = os.path.join(source_path, f"probabilities_{split_name}.csv")
            if not os.path.exists(split_path):
                print(
                    f"WARNING: {source_type} {source_name} missing split {split_name}; skipping split."
                )
                continue

            split_df = pd.read_csv(split_path)
            if not required_columns.issubset(split_df.columns):
                print(
                    f"WARNING: {source_type} {source_name} split {split_name} missing required columns; skipping split."
                )
                continue

            y_true = split_df["occuluded_artery"].to_numpy(dtype=int)
            raw_probability = split_df[probability_column].to_numpy(dtype=float)
            calibrated_probability = apply_calibration(raw_probability, calibrator)

            output = split_df[OUTPUT_COLUMNS].copy()
            output["raw_probability"] = raw_probability
            output["calibrated_probability"] = calibrated_probability
            output["calibrated_pred"] = (
                output["calibrated_probability"] >= PROBABILITY_THRESHOLD
            ).astype(int)
            output_path = os.path.join(out_dir, f"probabilities_{split_name}.csv")
            output.to_csv(output_path, index=False)

            raw_probability_clipped = np.clip(raw_probability, 1e-15, 1.0 - 1e-15)
            calibrated_probability_clipped = np.clip(
                calibrated_probability, 1e-15, 1.0 - 1e-15
            )

            raw_brier = brier_score_loss(y_true, raw_probability)
            calibrated_brier = brier_score_loss(y_true, calibrated_probability)
            raw_log_loss = log_loss(y_true, raw_probability_clipped, labels=[0, 1])
            calibrated_log_loss = log_loss(
                y_true, calibrated_probability_clipped, labels=[0, 1]
            )
            raw_ece = expected_calibration_error(y_true, raw_probability)
            calibrated_ece = expected_calibration_error(y_true, calibrated_probability)
            calibrated_accuracy = (
                (output["calibrated_pred"] == output["occuluded_artery"]).mean() * 100.0
            )

            try:
                raw_true, raw_pred = calibration_curve(
                    y_true, raw_probability, n_bins=CALIBRATION_BINS, strategy="uniform"
                )
            except ValueError:
                raw_true, raw_pred = np.array([]), np.array([])

            try:
                calibrated_true, calibrated_pred = calibration_curve(
                    y_true,
                    calibrated_probability,
                    n_bins=CALIBRATION_BINS,
                    strategy="uniform",
                )
            except ValueError:
                calibrated_true, calibrated_pred = np.array([]), np.array([])

            plt.figure(figsize=(5, 5))
            plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Perfect")
            if len(raw_true):
                plt.plot(raw_pred, raw_true, marker="o", linewidth=1.5, label="Raw")
            if len(calibrated_true):
                plt.plot(
                    calibrated_pred,
                    calibrated_true,
                    marker="o",
                    linewidth=1.5,
                    label="Calibrated",
                )
            plt.title(f"{source_name} {split_name}")
            plt.xlabel("Predicted probability")
            plt.ylabel("Observed frequency")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best")
            plt.tight_layout()
            plot_path = os.path.join(out_dir, f"calibration_plot_{split_name}.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()

            summary_rows.append(
                {
                    "source_type": source_type,
                    "source_name": source_name,
                    "split": split_name,
                    "calibration_split": CALIBRATION_SPLIT,
                    "calibration_method": calibration_method,
                    "accuracy_percent_calibrated": round(calibrated_accuracy, 2),
                    "brier_raw": round(float(raw_brier), 6),
                    "brier_calibrated": round(float(calibrated_brier), 6),
                    "log_loss_raw": round(float(raw_log_loss), 6),
                    "log_loss_calibrated": round(float(calibrated_log_loss), 6),
                    "ece_raw": round(float(raw_ece), 6),
                    "ece_calibrated": round(float(calibrated_ece), 6),
                    "n_samples": int(len(split_df)),
                }
            )

    if summary_rows:
        summary_path = os.path.join(CALIBRATION_DIR, "calibration_summary.json")
        with open(summary_path, "w") as handle:
            json.dump(summary_rows, handle, indent=2)
        print("Probability calibration completed.")
    else:
        print("WARNING: no calibration outputs generated.")


def run_abstain_uncertainty_logic():
    if not os.path.isdir(CALIBRATION_DIR):
        print("ERROR: calibration directory not found; skipping abstain logic.")
        return

    os.makedirs(ABSTAIN_DIR, exist_ok=True)

    ensembles_calibration_dir = os.path.join(CALIBRATION_DIR, "ensembles")
    models_calibration_dir = os.path.join(CALIBRATION_DIR, "models")
    if not os.path.isdir(ensembles_calibration_dir):
        print("ERROR: calibrated ensemble outputs missing; skipping abstain logic.")
        return
    if not os.path.isdir(models_calibration_dir):
        print("ERROR: calibrated model outputs missing; skipping abstain logic.")
        return

    selected_ensemble_method = None
    for method_name in ABSTAIN_ENSEMBLE_PRIORITY:
        method_path = os.path.join(
            ensembles_calibration_dir, method_name, f"probabilities_{CALIBRATION_SPLIT}.csv"
        )
        if os.path.exists(method_path):
            selected_ensemble_method = method_name
            break

    if selected_ensemble_method is None:
        print("ERROR: no calibrated ensemble probabilities found for abstain logic.")
        return

    model_names = sorted(
        [
            name
            for name in os.listdir(models_calibration_dir)
            if os.path.isdir(os.path.join(models_calibration_dir, name))
        ]
    )
    if not model_names:
        print("ERROR: no calibrated model directories found for abstain logic.")
        return

    def load_sorted_split(path, required_probability_column):
        df = pd.read_csv(path)
        required_columns = set(OUTPUT_COLUMNS + [required_probability_column])
        if not required_columns.issubset(df.columns):
            return None
        df = df.sort_values(["patient_id", "filename"]).reset_index(drop=True)
        return df

    def build_split_bundle(split_name):
        ensemble_path = os.path.join(
            ensembles_calibration_dir,
            selected_ensemble_method,
            f"probabilities_{split_name}.csv",
        )
        if not os.path.exists(ensemble_path):
            print(
                f"WARNING: missing calibrated ensemble probabilities for {split_name}; skipping split."
            )
            return None

        ensemble_df = load_sorted_split(ensemble_path, "calibrated_probability")
        if ensemble_df is None:
            print(
                f"WARNING: calibrated ensemble file missing columns for {split_name}; skipping split."
            )
            return None

        base_keys = ensemble_df[["patient_id", "filename"]].copy().reset_index(drop=True)
        model_pred_columns = []
        model_prob_columns = []
        models_used = []

        for model_name in model_names:
            model_path = os.path.join(
                models_calibration_dir, model_name, f"probabilities_{split_name}.csv"
            )
            if not os.path.exists(model_path):
                continue

            model_df = load_sorted_split(model_path, "calibrated_probability")
            if model_df is None:
                print(
                    f"WARNING: calibrated model file missing columns: {model_name} {split_name}; skipping model."
                )
                continue

            model_keys = model_df[["patient_id", "filename"]].copy().reset_index(drop=True)
            if not model_keys.equals(base_keys):
                print(
                    f"WARNING: calibrated model keys mismatch: {model_name} {split_name}; skipping model."
                )
                continue

            model_probability = model_df["calibrated_probability"].to_numpy(dtype=float)
            model_prediction = (model_probability >= PROBABILITY_THRESHOLD).astype(int)
            model_prob_columns.append(model_probability)
            model_pred_columns.append(model_prediction)
            models_used.append(model_name)

        if not model_pred_columns:
            print(f"ERROR: no aligned calibrated models available for {split_name}; skipping split.")
            return None

        model_pred_matrix = np.column_stack(model_pred_columns)
        model_prob_matrix = np.column_stack(model_prob_columns)
        return {
            "ensemble_df": ensemble_df,
            "model_pred_matrix": model_pred_matrix,
            "model_prob_matrix": model_prob_matrix,
            "models_used": models_used,
        }

    def disagreement_from_predictions(model_pred_matrix):
        positive_fraction = model_pred_matrix.mean(axis=1)
        return 1.0 - np.abs((2.0 * positive_fraction) - 1.0)

    validation_bundle = build_split_bundle(CALIBRATION_SPLIT)
    if validation_bundle is None:
        print("ERROR: validation bundle unavailable for abstain threshold selection.")
        return

    validation_df = validation_bundle["ensemble_df"]
    validation_probability = validation_df["calibrated_probability"].to_numpy(dtype=float)
    validation_pred = (validation_probability >= PROBABILITY_THRESHOLD).astype(int)
    validation_y = validation_df["occuluded_artery"].to_numpy(dtype=int)
    validation_confidence = np.abs((validation_probability * 2.0) - 1.0)
    validation_disagreement = disagreement_from_predictions(
        validation_bundle["model_pred_matrix"]
    )

    confidence_threshold_grid = np.linspace(0.0, 1.0, ABSTAIN_THRESHOLD_STEPS)
    disagreement_threshold_grid = np.linspace(0.0, 1.0, ABSTAIN_THRESHOLD_STEPS)
    feasible_choice = None
    fallback_choice = None

    for confidence_threshold in confidence_threshold_grid:
        confidence_mask = validation_confidence >= confidence_threshold
        for disagreement_threshold in disagreement_threshold_grid:
            keep_mask = confidence_mask & (validation_disagreement <= disagreement_threshold)
            coverage = keep_mask.mean()
            if coverage <= 0:
                continue

            error_rate = (
                (validation_pred[keep_mask] != validation_y[keep_mask]).mean()
            )

            if fallback_choice is None:
                fallback_choice = (coverage, error_rate, confidence_threshold, disagreement_threshold)
            else:
                best_coverage, best_error, _, _ = fallback_choice
                if (error_rate < best_error) or (
                    abs(error_rate - best_error) < 1e-12 and coverage > best_coverage
                ):
                    fallback_choice = (
                        coverage,
                        error_rate,
                        confidence_threshold,
                        disagreement_threshold,
                    )

            if error_rate <= ABSTAIN_TARGET_ERROR_RATE:
                if feasible_choice is None:
                    feasible_choice = (
                        coverage,
                        error_rate,
                        confidence_threshold,
                        disagreement_threshold,
                    )
                else:
                    best_coverage, best_error, _, _ = feasible_choice
                    if (coverage > best_coverage) or (
                        abs(coverage - best_coverage) < 1e-12 and error_rate < best_error
                    ):
                        feasible_choice = (
                            coverage,
                            error_rate,
                            confidence_threshold,
                            disagreement_threshold,
                        )

    if feasible_choice is None and fallback_choice is None:
        print("ERROR: failed to determine abstain thresholds.")
        return

    if feasible_choice is not None:
        (
            selected_coverage,
            selected_error_rate,
            confidence_threshold,
            disagreement_threshold,
        ) = feasible_choice
        threshold_selection_mode = "target_error_constrained"
    else:
        (
            selected_coverage,
            selected_error_rate,
            confidence_threshold,
            disagreement_threshold,
        ) = fallback_choice
        threshold_selection_mode = "fallback_min_error"

    summary_rows = []
    split_names = [split_name for split_name, _ in DATASETS]
    for split_name in split_names:
        split_bundle = build_split_bundle(split_name)
        if split_bundle is None:
            continue

        ensemble_df = split_bundle["ensemble_df"]
        model_pred_matrix = split_bundle["model_pred_matrix"]
        model_prob_matrix = split_bundle["model_prob_matrix"]
        models_used = split_bundle["models_used"]

        y_true = ensemble_df["occuluded_artery"].to_numpy(dtype=int)
        calibrated_probability = ensemble_df["calibrated_probability"].to_numpy(dtype=float)
        confidence = np.abs((calibrated_probability * 2.0) - 1.0)
        disagreement = disagreement_from_predictions(model_pred_matrix)
        model_probability_std = model_prob_matrix.std(axis=1)
        binary_pred = (calibrated_probability >= PROBABILITY_THRESHOLD).astype(int)

        abstain_mask = (confidence < confidence_threshold) | (
            disagreement > disagreement_threshold
        )
        final_pred = np.where(abstain_mask, -1, binary_pred)
        final_decision = np.where(
            abstain_mask,
            "not_sure",
            np.where(binary_pred == 1, "yes", "no"),
        )

        output = ensemble_df[OUTPUT_COLUMNS].copy()
        output["ensemble_method"] = selected_ensemble_method
        output["calibrated_probability"] = calibrated_probability
        output["confidence_score"] = confidence
        output["model_disagreement"] = disagreement
        output["model_probability_std"] = model_probability_std
        output["abstain"] = abstain_mask.astype(int)
        output["final_pred"] = final_pred.astype(int)
        output["final_decision"] = final_decision

        out_path = os.path.join(ABSTAIN_DIR, f"decisions_{split_name}.csv")
        output.to_csv(out_path, index=False)

        coverage_mask = ~abstain_mask
        coverage = coverage_mask.mean()
        abstain_rate = abstain_mask.mean()
        if coverage_mask.any():
            covered_accuracy = (binary_pred[coverage_mask] == y_true[coverage_mask]).mean()
        else:
            covered_accuracy = 0.0
        overall_correct_non_abstain = (
            ((binary_pred == y_true) & coverage_mask).sum() / len(y_true)
        )

        summary_rows.append(
            {
                "split": split_name,
                "ensemble_method": selected_ensemble_method,
                "n_samples": int(len(ensemble_df)),
                "n_models": len(models_used),
                "models_used": models_used,
                "confidence_threshold": round(float(confidence_threshold), 4),
                "disagreement_threshold": round(float(disagreement_threshold), 4),
                "target_error_rate": ABSTAIN_TARGET_ERROR_RATE,
                "coverage_percent": round(float(coverage * 100.0), 2),
                "abstain_percent": round(float(abstain_rate * 100.0), 2),
                "covered_accuracy_percent": round(float(covered_accuracy * 100.0), 2),
                "covered_error_percent": round(float((1.0 - covered_accuracy) * 100.0), 2),
                "overall_non_abstain_correct_percent": round(
                    float(overall_correct_non_abstain * 100.0), 2
                ),
            }
        )

    thresholds_used = {
        "ensemble_method": selected_ensemble_method,
        "selection_mode": threshold_selection_mode,
        "calibration_split": CALIBRATION_SPLIT,
        "target_error_rate": ABSTAIN_TARGET_ERROR_RATE,
        "confidence_threshold": round(float(confidence_threshold), 4),
        "disagreement_threshold": round(float(disagreement_threshold), 4),
        "validation_coverage_percent": round(float(selected_coverage * 100.0), 2),
        "validation_error_percent": round(float(selected_error_rate * 100.0), 2),
        "n_models_validation": len(validation_bundle["models_used"]),
        "models_used_validation": validation_bundle["models_used"],
    }

    thresholds_path = os.path.join(ABSTAIN_DIR, "thresholds_used.json")
    with open(thresholds_path, "w") as handle:
        json.dump(thresholds_used, handle, indent=2)

    if summary_rows:
        summary_path = os.path.join(ABSTAIN_DIR, "abstain_uncertainty_summary.json")
        with open(summary_path, "w") as handle:
            json.dump(summary_rows, handle, indent=2)
        print("Abstain and uncertainty outputs completed.")
    else:
        print("WARNING: no abstain and uncertainty outputs generated.")


def run_feature_importance_analysis():
    if not os.path.isdir(MODEL_OUTPUTS_DIR):
        print("ERROR: results/model_outputs directory not found; skipping feature importance analysis.")
        return

    os.makedirs(FEATURE_IMPORTANCE_DIR, exist_ok=True)
    if not os.path.exists(VALIDATION_PATH):
        ensure_validation_split()
    if not os.path.exists(VALIDATION_PATH):
        print("ERROR: validation split missing; skipping feature importance analysis.")
        return

    validation_df = pd.read_csv(VALIDATION_PATH)
    if "occuluded_artery" not in validation_df.columns:
        print("ERROR: validation split missing occuluded_artery; skipping feature importance analysis.")
        return

    bundles = []
    if os.path.isdir(MODEL_OUTPUTS_DIR):
        for model_name in sorted(os.listdir(MODEL_OUTPUTS_DIR)):
            model_dir = os.path.join(MODEL_OUTPUTS_DIR, model_name)
            if not os.path.isdir(model_dir):
                continue
            bundle_path = os.path.join(model_dir, "final_model_f1_new_overlap.joblib")
            if os.path.exists(bundle_path):
                bundles.append((model_name, bundle_path))

    bundles.sort(key=lambda item: item[0])
    if not bundles:
        print("ERROR: no model bundles found for feature importance analysis.")
        return

    accuracy_rows = []
    if os.path.exists(ACCURACY_PATH):
        with open(ACCURACY_PATH, "r") as handle:
            accuracy_rows = json.load(handle)

    validation_accuracy = {}
    for row in accuracy_rows:
        model_name = row.get("model")
        split_name = row.get("split")
        accuracy = row.get("accuracy_percent")
        if model_name is None or accuracy is None:
            continue
        if split_name != CALIBRATION_SPLIT:
            continue
        validation_accuracy[model_name] = float(accuracy)

    model_importance_maps = {}
    model_rank_maps = {}
    model_importance_source = {}
    model_feature_sets = {}

    def build_rank_map(features, scores):
        ranking = sorted(
            zip(features, scores),
            key=lambda item: (-float(item[1]), item[0]),
        )
        rank_map = {}
        for rank, (feature_name, _) in enumerate(ranking, start=1):
            rank_map[feature_name] = rank
        return rank_map

    def normalize_importance(raw_importance):
        values = np.asarray(raw_importance, dtype=float)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = np.abs(values)
        total = values.sum()
        if total <= 0:
            return np.ones_like(values) / len(values)
        return values / total

    def permutation_importance(bundle):
        model = bundle["model"]
        scaler = bundle["scaler"]
        cols = bundle["cols"]
        features = list(bundle["features"])
        try:
            X_scaled = scaler.transform(validation_df[cols])
            X_scaled = pd.DataFrame(X_scaled, columns=cols, index=validation_df.index)
            X_input = X_scaled[features].copy()
            y_true = validation_df["occuluded_artery"].to_numpy(dtype=int)
            baseline_probability = model.predict_proba(X_input)[:, 1]
            baseline_probability = np.clip(baseline_probability, 1e-15, 1.0 - 1e-15)
            baseline_loss = log_loss(y_true, baseline_probability, labels=[0, 1])

            rng = np.random.default_rng(42)
            importances = np.zeros(len(features), dtype=float)
            for index, feature_name in enumerate(features):
                X_permuted = X_input.copy()
                X_permuted[feature_name] = rng.permutation(X_permuted[feature_name].to_numpy())
                permuted_probability = model.predict_proba(X_permuted)[:, 1]
                permuted_probability = np.clip(permuted_probability, 1e-15, 1.0 - 1e-15)
                permuted_loss = log_loss(y_true, permuted_probability, labels=[0, 1])
                importances[index] = max(permuted_loss - baseline_loss, 0.0)
            return importances, "permutation_log_loss"
        except Exception as exc:
            print(f"WARNING: permutation importance failed; using uniform fallback. {exc}")
            return np.ones(len(features), dtype=float), "uniform_fallback"

    for model_name, bundle_path in bundles:
        bundle = joblib.load(bundle_path)
        model = bundle["model"]
        features = list(bundle["features"])
        raw_importance = None
        importance_source = None

        if hasattr(model, "feature_importances_"):
            raw_importance = np.asarray(model.feature_importances_, dtype=float)
            importance_source = "feature_importances_"
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=float)
            if coef.ndim == 1:
                raw_importance = np.abs(coef)
            else:
                raw_importance = np.abs(coef).mean(axis=0)
            importance_source = "coef_"
        elif hasattr(model, "estimators_"):
            tree_importances = []
            for estimator in model.estimators_:
                if hasattr(estimator, "feature_importances_"):
                    estimator_importance = np.asarray(
                        estimator.feature_importances_, dtype=float
                    )
                    if len(estimator_importance) == len(features):
                        tree_importances.append(estimator_importance)
            if tree_importances:
                raw_importance = np.mean(tree_importances, axis=0)
                importance_source = "estimators_feature_importances_"

        if raw_importance is None or len(raw_importance) != len(features):
            raw_importance, importance_source = permutation_importance(bundle)

        normalized_importance = normalize_importance(raw_importance)
        rank_map = build_rank_map(features, normalized_importance)

        model_importance_source[model_name] = importance_source
        model_feature_sets[model_name] = features
        model_importance_maps[model_name] = dict(zip(features, normalized_importance))
        model_rank_maps[model_name] = rank_map

        model_out_dir = os.path.join(FEATURE_IMPORTANCE_DIR, "models", model_name)
        os.makedirs(model_out_dir, exist_ok=True)
        model_rows = []
        for feature_name in features:
            model_rows.append(
                {
                    "feature": feature_name,
                    "importance": float(model_importance_maps[model_name][feature_name]),
                    "rank": int(model_rank_maps[model_name][feature_name]),
                    "importance_source": importance_source,
                }
            )
        model_df = pd.DataFrame(model_rows).sort_values(
            ["rank", "feature"], ascending=[True, True]
        )
        model_df.to_csv(os.path.join(model_out_dir, "feature_importance.csv"), index=False)

    all_features = sorted(
        {feature_name for features in model_feature_sets.values() for feature_name in features}
    )
    if not all_features:
        print("ERROR: no features available for aggregation.")
        return

    n_features = len(all_features)
    model_names = sorted(model_importance_maps.keys())
    if not model_names:
        print("ERROR: no model importances available for aggregation.")
        return

    complete_importance = {}
    complete_rank = {}
    for model_name in model_names:
        importance_map = {}
        rank_map = {}
        for feature_name in all_features:
            importance_map[feature_name] = float(
                model_importance_maps[model_name].get(feature_name, 0.0)
            )
            rank_map[feature_name] = int(model_rank_maps[model_name].get(feature_name, n_features + 1))
        complete_importance[model_name] = importance_map
        complete_rank[model_name] = rank_map

    raw_weights = {}
    for model_name in model_names:
        raw_weights[model_name] = validation_accuracy.get(model_name, 0.0)
    weight_total = sum(raw_weights.values())
    if weight_total <= 0:
        normalized_weights = {name: 1.0 / len(model_names) for name in model_names}
    else:
        normalized_weights = {
            name: raw_weights[name] / weight_total for name in model_names
        }

    ranked_by_accuracy = sorted(
        model_names,
        key=lambda name: (-raw_weights.get(name, 0.0), name),
    )
    top_k_models = ranked_by_accuracy[: min(TOP_K_MODELS, len(ranked_by_accuracy))]
    vote_top_k = max(1, int(np.ceil(FEATURE_VOTE_TOP_FRACTION * n_features)))

    aggregated_scores = {}
    aggregated_metadata = {}

    hard_scores = {}
    for feature_name in all_features:
        vote_count = sum(
            1
            for model_name in model_names
            if complete_rank[model_name][feature_name] <= vote_top_k
        )
        hard_scores[feature_name] = vote_count / len(model_names)
    aggregated_scores["hard_voting"] = hard_scores
    aggregated_metadata["hard_voting"] = {
        "models_used": model_names,
        "vote_top_k": vote_top_k,
    }

    soft_scores = {}
    for feature_name in all_features:
        soft_scores[feature_name] = float(
            np.mean([complete_importance[model_name][feature_name] for model_name in model_names])
        )
    aggregated_scores["soft_voting"] = soft_scores
    aggregated_metadata["soft_voting"] = {
        "models_used": model_names,
    }

    weighted_scores = {}
    for feature_name in all_features:
        weighted_scores[feature_name] = sum(
            normalized_weights[model_name] * complete_importance[model_name][feature_name]
            for model_name in model_names
        )
    aggregated_scores["performance_weighted_soft"] = weighted_scores
    aggregated_metadata["performance_weighted_soft"] = {
        "models_used": model_names,
        "weights_used": normalized_weights,
        "weight_split": CALIBRATION_SPLIT,
    }

    k_filter_scores = {}
    for feature_name in all_features:
        k_filter_scores[feature_name] = float(
            np.mean(
                [
                    complete_importance[model_name][feature_name]
                    for model_name in top_k_models
                ]
            )
        )
    aggregated_scores["k_filter_voting"] = k_filter_scores
    aggregated_metadata["k_filter_voting"] = {
        "models_used": top_k_models,
        "k": TOP_K_MODELS,
        "weight_split": CALIBRATION_SPLIT,
    }

    stacking_models = []
    stacking_weights = {}
    validation_probabilities = {}
    validation_prob_keys = None
    validation_prob_labels = None
    for model_name in model_names:
        validation_prob_path = os.path.join(
            MODEL_OUTPUTS_DIR, model_name, f"probabilities_{CALIBRATION_SPLIT}.csv"
        )
        if not os.path.exists(validation_prob_path):
            continue

        validation_prob_df = pd.read_csv(validation_prob_path)
        required_columns = {"patient_id", "filename", "occuluded_artery", "mi_probability"}
        if not required_columns.issubset(validation_prob_df.columns):
            continue

        validation_prob_df = validation_prob_df.sort_values(
            ["patient_id", "filename"]
        ).reset_index(drop=True)
        current_keys = validation_prob_df[["patient_id", "filename"]].copy().reset_index(drop=True)
        if validation_prob_keys is None:
            validation_prob_keys = current_keys
            validation_prob_labels = validation_prob_df["occuluded_artery"].to_numpy(dtype=int)
        elif not current_keys.equals(validation_prob_keys):
            continue

        validation_probabilities[model_name] = validation_prob_df["mi_probability"].to_numpy(dtype=float)

    if validation_probabilities:
        stacking_models = [name for name in model_names if name in validation_probabilities]

    if stacking_models and validation_prob_labels is not None:
        X_meta = np.column_stack([validation_probabilities[name] for name in stacking_models])
        y_meta = np.asarray(validation_prob_labels, dtype=int)
        if len(np.unique(y_meta)) >= 2:
            stacker = LogisticRegression(max_iter=1000, solver="liblinear")
            stacker.fit(X_meta, y_meta)
            coef_values = np.abs(np.asarray(stacker.coef_, dtype=float).ravel())
            coef_sum = float(coef_values.sum())
            if coef_sum > 0:
                stacking_weights = {
                    name: float(weight / coef_sum)
                    for name, weight in zip(stacking_models, coef_values)
                }

    if not stacking_weights:
        if stacking_models:
            stacking_weights = {
                name: 1.0 / len(stacking_models) for name in stacking_models
            }
        else:
            stacking_models = list(model_names)
            stacking_weights = {
                name: normalized_weights.get(name, 1.0 / len(model_names))
                for name in stacking_models
            }

    stacking_scores = {}
    for feature_name in all_features:
        stacking_scores[feature_name] = sum(
            stacking_weights[name] * complete_importance[name][feature_name]
            for name in stacking_models
        )
    aggregated_scores["stacking_logistic"] = stacking_scores
    aggregated_metadata["stacking_logistic"] = {
        "models_used": stacking_models,
        "weights_used": stacking_weights,
        "weight_split": CALIBRATION_SPLIT,
    }

    rra_scores = {}
    for feature_name in all_features:
        rank_percentiles = np.array(
            [
                complete_rank[model_name][feature_name] / n_features
                for model_name in model_names
            ],
            dtype=float,
        )
        rank_percentiles = np.clip(rank_percentiles, 1e-12, 1.0)
        geometric_mean_rank = float(np.exp(np.mean(np.log(rank_percentiles))))
        rra_scores[feature_name] = 1.0 - geometric_mean_rank
    aggregated_scores["robust_rank_aggregation"] = rra_scores
    aggregated_metadata["robust_rank_aggregation"] = {
        "models_used": model_names,
    }

    mean_rank_scores = {}
    for feature_name in all_features:
        rank_percentiles = np.array(
            [
                complete_rank[model_name][feature_name] / n_features
                for model_name in model_names
            ],
            dtype=float,
        )
        mean_rank_scores[feature_name] = 1.0 - float(np.mean(rank_percentiles))
    aggregated_scores["mean_rank_aggregation"] = mean_rank_scores
    aggregated_metadata["mean_rank_aggregation"] = {
        "models_used": model_names,
    }

    aggregation_summary_rows = []
    ranked_features_by_method = {}
    for method_name, score_map in aggregated_scores.items():
        method_dir = os.path.join(FEATURE_IMPORTANCE_DIR, "aggregated", method_name)
        os.makedirs(method_dir, exist_ok=True)
        ranking = sorted(
            score_map.items(),
            key=lambda item: (-float(item[1]), item[0]),
        )
        ranked_features_by_method[method_name] = [feature_name for feature_name, _ in ranking]

        rows = []
        for rank, (feature_name, score) in enumerate(ranking, start=1):
            rows.append(
                {
                    "feature": feature_name,
                    "score": float(score),
                    "rank": int(rank),
                }
            )

        pd.DataFrame(rows).to_csv(
            os.path.join(method_dir, "feature_ranking.csv"), index=False
        )
        aggregation_summary_rows.append(
            {
                "method": method_name,
                "n_features": n_features,
                "metadata": aggregated_metadata[method_name],
            }
        )

    aggregation_summary_path = os.path.join(
        FEATURE_IMPORTANCE_DIR, "aggregation_summary.json"
    )
    with open(aggregation_summary_path, "w") as handle:
        json.dump(aggregation_summary_rows, handle, indent=2)

    def evaluate_subset_with_sweep(method_name, selected_features):
        subset_eval_dir = os.path.join(
            FEATURE_IMPORTANCE_DIR, "subset_evaluations", method_name
        )
        os.makedirs(subset_eval_dir, exist_ok=True)

        selected_path = os.path.join(subset_eval_dir, "selected_features.txt")
        with open(selected_path, "w") as handle:
            handle.write("\n".join(selected_features))
            handle.write("\n")

        subset_summary_path = os.path.join(
            subset_eval_dir, "subset_evaluation_summary.json"
        )
        if os.path.exists(subset_summary_path):
            with open(subset_summary_path, "r") as handle:
                cached_summary = json.load(handle)
            cached_summary["status"] = "cached"
            return cached_summary

        workdir = os.path.join(subset_eval_dir, "workdir")
        os.makedirs(workdir, exist_ok=True)
        work_data_dir = os.path.join(workdir, "data")
        work_cleaned_dir = os.path.join(work_data_dir, "cleaned")
        os.makedirs(work_cleaned_dir, exist_ok=True)

        linked_ood_dir = os.path.join(work_data_dir, "data_ood_id")
        source_ood_dir = os.path.join(ROOT_DIR, "data", "data_ood_id")
        if not os.path.exists(linked_ood_dir):
            os.symlink(source_ood_dir, linked_ood_dir)

        work_features_path = os.path.join(
            work_cleaned_dir, "selected_features_f1_final_new_common_nate.txt"
        )
        with open(work_features_path, "w") as handle:
            handle.write("\n".join(selected_features))
            handle.write("\n")

        sweep_script_path = os.path.join(workdir, "model_sweep.py")
        shutil.copy2(MODEL_SWEEP_PATH, sweep_script_path)

        sweep_env = os.environ.copy()
        sweep_env["PYTHONWARNINGS"] = "ignore"
        result = subprocess.run(
            [sys.executable, sweep_script_path],
            cwd=workdir,
            env=sweep_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            summary = {
                "method": method_name,
                "status": "failed",
                "subset_size": len(selected_features),
                "selected_features": selected_features,
            }
            with open(subset_summary_path, "w") as handle:
                json.dump(summary, handle, indent=2)
            return summary

        model_results = []
        local_results_dir = os.path.join(workdir, "results")
        if os.path.isdir(local_results_dir):
            for model_name in sorted(os.listdir(local_results_dir)):
                model_dir = os.path.join(local_results_dir, model_name)
                if not os.path.isdir(model_dir):
                    continue
                result_json_path = os.path.join(
                    model_dir, "final_model_f1_new_overlap.json"
                )
                if not os.path.exists(result_json_path):
                    continue
                with open(result_json_path, "r") as handle:
                    model_json = json.load(handle)
                auc_test_1 = (
                    model_json.get("test_1", {}).get("overall", {}).get("AUC")
                )
                auc_test_2 = (
                    model_json.get("test_2", {}).get("overall", {}).get("AUC")
                )
                auc_test_3 = (
                    model_json.get("test_3", {}).get("overall", {}).get("AUC")
                )
                model_results.append(
                    {
                        "model": model_name,
                        "auc_test_set_ood_1": auc_test_1,
                        "auc_test_set_ood_2": auc_test_2,
                        "auc_test_set_id": auc_test_3,
                    }
                )

        def safe_mean(values):
            clean_values = [value for value in values if value is not None]
            if not clean_values:
                return None
            return float(np.mean(clean_values))

        summary = {
            "method": method_name,
            "status": "success",
            "subset_size": len(selected_features),
            "selected_features": selected_features,
            "n_models_evaluated": len(model_results),
            "mean_auc_test_set_ood_1": safe_mean(
                [row["auc_test_set_ood_1"] for row in model_results]
            ),
            "mean_auc_test_set_ood_2": safe_mean(
                [row["auc_test_set_ood_2"] for row in model_results]
            ),
            "mean_auc_test_set_id": safe_mean(
                [row["auc_test_set_id"] for row in model_results]
            ),
            "model_results": model_results,
        }
        with open(subset_summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        return summary

    subset_summary_rows = []
    subset_cache = {}
    excluded_subset_features = {"age", "sex"}
    for method_name in [
        "hard_voting",
        "soft_voting",
        "performance_weighted_soft",
        "k_filter_voting",
        "stacking_logistic",
        "robust_rank_aggregation",
        "mean_rank_aggregation",
    ]:
        ranked_features = ranked_features_by_method.get(method_name, [])
        subset_features = [
            feature_name
            for feature_name in ranked_features
            if feature_name not in excluded_subset_features
        ][:FEATURE_SUBSET_SIZE]

        if not subset_features:
            subset_summary_rows.append(
                {
                    "method": method_name,
                    "status": "skipped",
                    "reason": "no subset features available",
                }
            )
            continue

        subset_key = tuple(subset_features)
        if subset_key in subset_cache:
            cached_summary = dict(subset_cache[subset_key])
            cached_summary["method"] = method_name
            cached_summary["status"] = "reused_subset_result"
            cached_summary["reused_from_method"] = subset_cache[subset_key]["method"]
            summary = cached_summary
        else:
            summary = evaluate_subset_with_sweep(method_name, subset_features)
            subset_cache[subset_key] = dict(summary)
        subset_summary_rows.append(summary)

    subset_summary_path = os.path.join(
        FEATURE_IMPORTANCE_DIR, "subset_evaluation_summary.json"
    )
    with open(subset_summary_path, "w") as handle:
        json.dump(subset_summary_rows, handle, indent=2)

    model_importance_metadata = []
    for model_name in model_names:
        model_importance_metadata.append(
            {
                "model": model_name,
                "importance_source": model_importance_source[model_name],
                "n_features": len(model_feature_sets[model_name]),
            }
        )
    model_importance_metadata_path = os.path.join(
        FEATURE_IMPORTANCE_DIR, "model_importance_summary.json"
    )
    with open(model_importance_metadata_path, "w") as handle:
        json.dump(model_importance_metadata, handle, indent=2)

    lead_names = [
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]
    feature_definition_rows = []
    for feature_name in all_features:
        definition = "Custom feature"
        if feature_name == "age":
            definition = "Patient age"
        elif feature_name == "sex":
            definition = "Patient sex"
        elif feature_name.startswith("feature_"):
            try:
                feature_index = int(feature_name.split("_")[1])
            except (IndexError, ValueError):
                feature_index = None
            if feature_index is not None:
                if 1 <= feature_index <= 12:
                    lead = lead_names[feature_index - 1]
                    definition = f"Mean amplitude ({lead})"
                elif 13 <= feature_index <= 24:
                    lead = lead_names[feature_index - 13]
                    definition = f"Std amplitude ({lead})"
                elif feature_index == 25:
                    definition = "Mean absolute amplitude (all leads)"
                elif feature_index == 26:
                    definition = "Std absolute amplitude (all leads)"
                elif feature_index == 27:
                    definition = "Max absolute amplitude (all leads)"
                elif feature_index == 28:
                    definition = "RMS amplitude (all leads)"
                elif feature_index == 29:
                    definition = "Mean peak-to-peak (all leads)"
                elif feature_index == 30:
                    definition = "Std peak-to-peak (all leads)"

        feature_definition_rows.append(
            {"feature": feature_name, "definition": definition}
        )

    feature_definition_df = pd.DataFrame(feature_definition_rows).sort_values("feature")
    feature_definition_df.to_csv(
        os.path.join(FEATURE_IMPORTANCE_DIR, "feature_definitions.csv"),
        index=False,
    )

    print("Feature importance analysis completed.")


def run_final_summary_plots():
    if not os.path.isdir(RESULTS_DIR):
        print("ERROR: results directory not found; skipping final plots.")
        return

    os.makedirs(FINAL_PLOTS_DIR, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    single_model_accuracy_df = None
    best_single_accuracy_by_split = {}
    best_single_model_name = None
    best_single_model_mean_auc = None
    model_auc_rows = []
    ensemble_df_for_summary = None
    feature_label_map = {}
    feature_definition_path = os.path.join(
        FEATURE_IMPORTANCE_DIR, "feature_definitions.csv"
    )
    if os.path.exists(feature_definition_path):
        try:
            feature_definition_df = pd.read_csv(feature_definition_path)
            if {"feature", "definition"}.issubset(feature_definition_df.columns):
                for row in feature_definition_df.itertuples(index=False):
                    feature_label_map[str(row.feature)] = str(row.definition)
        except Exception:
            feature_label_map = {}

    def format_feature_label(feature_name):
        feature_name = str(feature_name)
        definition = feature_label_map.get(feature_name)
        if definition is None or definition == "Custom feature":
            return feature_name
        return f"{feature_name} - {definition}"

    if os.path.exists(ACCURACY_PATH):
        with open(ACCURACY_PATH, "r") as handle:
            single_model_rows = json.load(handle)
        if single_model_rows:
            single_model_accuracy_df = pd.DataFrame(single_model_rows)
            for split_name in ["test_set_ood_1", "test_set_ood_2", "test_set_id"]:
                split_df = single_model_accuracy_df[
                    single_model_accuracy_df["split"] == split_name
                ]
                if not split_df.empty:
                    best_single_accuracy_by_split[split_name] = float(
                        split_df["accuracy_percent"].max()
                    )

    if os.path.isdir(MODEL_OUTPUTS_DIR):
        model_names_for_summary = sorted(os.listdir(MODEL_OUTPUTS_DIR))
    else:
        model_names_for_summary = []

    for model_name in model_names_for_summary:
        model_dir = os.path.join(MODEL_OUTPUTS_DIR, model_name)
        if not os.path.isdir(model_dir):
            continue
        metrics_path = os.path.join(model_dir, "final_model_f1_new_overlap.json")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path, "r") as handle:
            model_metrics = json.load(handle)
        auc_values = [
            model_metrics.get("test_1", {}).get("overall", {}).get("AUC"),
            model_metrics.get("test_2", {}).get("overall", {}).get("AUC"),
            model_metrics.get("test_3", {}).get("overall", {}).get("AUC"),
        ]
        auc_values = [value for value in auc_values if value is not None]
        if not auc_values:
            continue
        model_mean_auc = float(np.mean(auc_values))
        model_auc_rows.append({"model": model_name, "mean_auc": model_mean_auc})

    if model_auc_rows:
        model_auc_rows = sorted(
            model_auc_rows, key=lambda row: (-row["mean_auc"], row["model"])
        )
        best_single_model_name = model_auc_rows[0]["model"]
        best_single_model_mean_auc = model_auc_rows[0]["mean_auc"]

    ensemble_summary_path = os.path.join(ENSEMBLES_DIR, "ensemble_accuracy_summary.json")
    if os.path.exists(ensemble_summary_path):
        with open(ensemble_summary_path, "r") as handle:
            ensemble_rows = json.load(handle)

        if ensemble_rows:
            ensemble_df = pd.DataFrame(ensemble_rows)
            ensemble_df = ensemble_df[
                ensemble_df["split"].isin(["test_set_ood_1", "test_set_ood_2", "test_set_id"])
            ].copy()
            ensemble_df_for_summary = ensemble_df.copy()
            if not ensemble_df.empty:
                method_order = (
                    ensemble_df.groupby("method")["accuracy_percent"]
                    .mean()
                    .sort_values(ascending=False)
                    .index.tolist()
                )
                split_order = ["test_set_ood_1", "test_set_ood_2", "test_set_id"]
                category_order = list(method_order)
                if len(best_single_accuracy_by_split) == len(split_order):
                    category_order.append("best_single_model")

                bar_width = 0.22
                x = np.arange(len(category_order))
                colors = ["#4C72B0", "#55A868", "#C44E52"]
                split_display = {
                    "test_set_ood_1": "OOD-1",
                    "test_set_ood_2": "OOD-2",
                    "test_set_id": "ID",
                }
                method_display = {
                    "hard_voting": "Hard",
                    "soft_voting": "Soft",
                    "performance_weighted_soft": "Weighted",
                    "k_filter_voting": "Top-K",
                    "stacking_logistic": "Stacking LR",
                    "best_single_model": "Best Single",
                }

                plt.figure(figsize=(16, 7.6))
                for index, split_name in enumerate(split_order):
                    split_values = []
                    for category_name in category_order:
                        if category_name == "best_single_model":
                            split_values.append(best_single_accuracy_by_split.get(split_name, np.nan))
                            continue
                        match = ensemble_df[
                            (ensemble_df["method"] == category_name)
                            & (ensemble_df["split"] == split_name)
                        ]
                        split_values.append(float(match["accuracy_percent"].iloc[0]) if not match.empty else np.nan)
                    positions = x + (index - 1) * bar_width
                    bars = plt.bar(
                        positions,
                        split_values,
                        width=bar_width,
                        label=split_display.get(split_name, split_name),
                        color=colors[index],
                        alpha=0.9,
                    )
                    for bar, value in zip(bars, split_values):
                        if np.isnan(value):
                            continue
                        label_y = value - 1.8 if value >= 8 else value + 0.2
                        label_va = "top" if value >= 8 else "bottom"
                        label_color = "white" if value >= 8 else "black"
                        plt.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            label_y,
                            f"{value:.1f}",
                            ha="center",
                            va=label_va,
                            fontsize=7,
                            color=label_color,
                        )

                if "best_single_model" in category_order:
                    divider_index = category_order.index("best_single_model") - 0.5
                    plt.axvline(divider_index, color="#666666", linestyle="--", linewidth=1.0)

                method_labels = [method_display.get(name, name) for name in category_order]
                plt.xticks(x, method_labels, rotation=0, ha="center")
                plt.ylabel("Accuracy (%)")
                plt.title("Ensemble Accuracy Comparison (higher is better)")
                plt.legend(
                    title="Split",
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.14),
                    ncol=3,
                    frameon=False,
                )
                plt.figtext(
                    0.5,
                    0.01,
                    "Split key: OOD-1/OOD-2 = out-of-distribution test sets, ID = in-distribution test set.",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
                plt.ylim(0.0, 102.0)
                plt.tight_layout(rect=[0, 0.05, 1, 0.91])
                plt.savefig(
                    os.path.join(FINAL_PLOTS_DIR, "ensemble_methods_bar.png"),
                    dpi=180,
                )
                plt.close()
            else:
                print("WARNING: no ensemble test rows available for final ensemble plot.")
        else:
            print("WARNING: ensemble summary is empty; skipping ensemble plot.")
    else:
        print("WARNING: ensemble summary file missing; skipping ensemble plot.")

    abstain_summary_path = os.path.join(ABSTAIN_DIR, "abstain_uncertainty_summary.json")
    if os.path.exists(abstain_summary_path):
        with open(abstain_summary_path, "r") as handle:
            abstain_rows = json.load(handle)

        if abstain_rows:
            abstain_df = pd.DataFrame(abstain_rows)
            split_order = [split_name for split_name, _ in DATASETS]
            abstain_df["split"] = pd.Categorical(
                abstain_df["split"], categories=split_order, ordered=True
            )
            abstain_df = abstain_df.sort_values("split")

            x = np.arange(len(abstain_df))
            split_display = {
                "validation": "Validation",
                "test_set_ood_1": "OOD-1",
                "test_set_ood_2": "OOD-2",
                "test_set_id": "ID",
            }
            threshold_text = None
            thresholds_path = os.path.join(ABSTAIN_DIR, "thresholds_used.json")
            if os.path.exists(thresholds_path):
                with open(thresholds_path, "r") as handle:
                    thresholds_data = json.load(handle)
                threshold_text = (
                    f"Method={thresholds_data.get('ensemble_method', 'NA')}, "
                    f"conf>={thresholds_data.get('confidence_threshold', 'NA')}, "
                    f"disagree<={thresholds_data.get('disagreement_threshold', 'NA')}"
                )

            coverage = abstain_df["coverage_percent"].astype(float).to_numpy()
            covered_accuracy = abstain_df["covered_accuracy_percent"].astype(float).to_numpy()
            not_sure = abstain_df["abstain_percent"].astype(float).to_numpy()
            auto_correct = (coverage * covered_accuracy) / 100.0
            auto_incorrect = np.maximum(coverage - auto_correct, 0.0)
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(12, 8.6),
                sharex=True,
                gridspec_kw={"height_ratios": [2.0, 1.35]},
            )
            ax_outcome, ax_metrics = axes

            bars_correct = ax_outcome.bar(
                x,
                auto_correct,
                width=0.58,
                label="Auto Correct",
                color="#55A868",
                alpha=0.95,
            )
            bars_incorrect = ax_outcome.bar(
                x,
                auto_incorrect,
                width=0.58,
                bottom=auto_correct,
                label="Auto Incorrect",
                color="#C44E52",
                alpha=0.95,
            )
            bars_not_sure = ax_outcome.bar(
                x,
                not_sure,
                width=0.58,
                bottom=auto_correct + auto_incorrect,
                label="Not Sure",
                color="#8C8C8C",
                alpha=0.95,
            )

            for bars in [bars_correct, bars_incorrect, bars_not_sure]:
                for bar in bars:
                    height = bar.get_height()
                    if height < 7.0:
                        continue
                    y_center = bar.get_y() + (height / 2.0)
                    ax_outcome.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        y_center,
                        f"{height:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                    )

            ax_outcome.set_ylabel("All cases (%)")
            ax_outcome.set_ylim(0.0, 100.0)
            ax_outcome.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                frameon=False,
            )
            ax_outcome.set_title("Abstain / Uncertainty Outcomes by Split")

            bar_width = 0.34
            coverage_bars = ax_metrics.bar(
                x - (bar_width / 2.0),
                coverage,
                width=bar_width,
                label="Coverage (auto yes/no %)",
                color="#4C72B0",
                alpha=0.9,
            )
            covered_acc_bars = ax_metrics.bar(
                x + (bar_width / 2.0),
                covered_accuracy,
                width=bar_width,
                label="Covered Accuracy (%)",
                color="#E17C05",
                alpha=0.9,
            )
            for bars in [coverage_bars, covered_acc_bars]:
                for bar in bars:
                    value = bar.get_height()
                    ax_metrics.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        value + 1.0,
                        f"{value:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

            ax_metrics.set_ylabel("Percent (%)")
            ax_metrics.set_ylim(0.0, 100.0)
            ax_metrics.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.18),
                ncol=2,
                frameon=False,
            )
            x_labels = [split_display.get(value, value) for value in abstain_df["split"].tolist()]
            ax_metrics.set_xticks(x)
            ax_metrics.set_xticklabels(x_labels, rotation=0, ha="center")
            ax_metrics.set_xlabel("Dataset split")

            info_lines = [
                "Coverage = % of cases where model gives auto yes/no",
                "Covered accuracy = accuracy only within those covered cases",
                "Not sure = % abstained",
                "Split key: OOD-1/OOD-2 = out-of-distribution test sets, ID = in-distribution test set.",
            ]
            if threshold_text is not None:
                info_lines.append(threshold_text)
            fig.text(0.5, 0.01, "\n".join(info_lines), ha="center", va="bottom", fontsize=8)

            fig.tight_layout(rect=[0, 0.14, 1, 0.95])
            fig.savefig(
                os.path.join(FINAL_PLOTS_DIR, "abstain_uncertainty_bar.png"),
                dpi=180,
            )
            plt.close(fig)
        else:
            print("WARNING: abstain summary is empty; skipping abstain plot.")
    else:
        print("WARNING: abstain summary file missing; skipping abstain plot.")

    subset_summary_path = os.path.join(
        FEATURE_IMPORTANCE_DIR, "subset_evaluation_summary.json"
    )
    if os.path.exists(subset_summary_path):
        with open(subset_summary_path, "r") as handle:
            subset_rows = json.load(handle)

        subset_df = pd.DataFrame(subset_rows)
        if not subset_df.empty:
            valid_mask = subset_df["status"].isin(
                ["success", "cached", "reused_subset_result"]
            )
            valid_df = subset_df[valid_mask].copy()
            if not valid_df.empty:
                valid_df["overall_auc"] = valid_df[
                    ["mean_auc_test_set_ood_1", "mean_auc_test_set_ood_2", "mean_auc_test_set_id"]
                ].mean(axis=1, skipna=True)
                valid_df = valid_df.sort_values("overall_auc", ascending=False)

                values = valid_df["overall_auc"].astype(float).to_numpy()
                colors = ["#4C72B0" if index == 0 else "#8AA8D6" for index in range(len(values))]
                method_display = {
                    "hard_voting": "Hard Voting",
                    "soft_voting": "Soft Voting",
                    "performance_weighted_soft": "Weighted Soft",
                    "k_filter_voting": "Top-K Voting",
                    "stacking_logistic": "Stacking (LR)",
                    "robust_rank_aggregation": "RRA",
                    "mean_rank_aggregation": "Mean Rank",
                }
                method_labels = [
                    method_display.get(value, value) for value in valid_df["method"].tolist()
                ]

                fig, ax = plt.subplots(figsize=(12, 6.8))
                y = np.arange(len(method_labels))
                bars = ax.barh(y, values, color=colors, alpha=0.95)
                ax.set_yticks(y)
                ax.set_yticklabels(method_labels)
                ax.invert_yaxis()

                baseline_value = None
                if best_single_model_mean_auc is not None:
                    baseline_value = float(best_single_model_mean_auc)
                    baseline_label = (
                        f"Single-best full model AUC ({best_single_model_name}) = "
                        f"{baseline_value:.3f}"
                    )
                    ax.axvline(
                        baseline_value,
                        color="#C44E52",
                        linestyle="--",
                        linewidth=1.4,
                        label=baseline_label,
                    )

                x_max = float(np.nanmax(values))
                if baseline_value is not None:
                    x_max = max(x_max, baseline_value)
                x_max = min(1.0, x_max + 0.12)
                ax.set_xlim(0.0, x_max)

                for index, (bar, value) in enumerate(zip(bars, values)):
                    text_x = min(value + 0.008, x_max - 0.01)
                    delta_text = ""
                    if baseline_value is not None:
                        delta_text = f" ({value - baseline_value:+.3f} vs baseline)"
                    ax.text(
                        text_x,
                        index,
                        f"{value:.3f}{delta_text}",
                        ha="left",
                        va="center",
                        fontsize=8,
                    )

                ax.set_xlabel("Mean AUC across OOD-1, OOD-2, and ID")
                ax.set_title("Feature Aggregation Methods: Subset Evaluation Quality")
                if baseline_value is not None:
                    ax.legend(loc="lower right", frameon=False)

                fig.text(
                    0.5,
                    0.01,
                    (
                        f"Evaluation logic: each method ranks features -> top {FEATURE_SUBSET_SIZE} are selected -> "
                        "model_sweep is rerun on that subset -> mean AUC is reported. "
                        "Split key: OOD-1/OOD-2 = out-of-distribution, ID = in-distribution."
                    ),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
                fig.tight_layout(rect=[0, 0.07, 1, 0.98])
                fig.savefig(
                    os.path.join(FINAL_PLOTS_DIR, "feature_importance_methods_bar.png"),
                    dpi=180,
                )
                plt.close(fig)
            else:
                placeholder_methods = subset_df["method"].astype(str).tolist()
                if not placeholder_methods:
                    placeholder_methods = [
                        "hard_voting",
                        "soft_voting",
                        "performance_weighted_soft",
                        "k_filter_voting",
                        "stacking_logistic",
                        "robust_rank_aggregation",
                        "mean_rank_aggregation",
                    ]
                method_display = {
                    "hard_voting": "Hard Voting",
                    "soft_voting": "Soft Voting",
                    "performance_weighted_soft": "Weighted Soft",
                    "k_filter_voting": "Top-K Voting",
                    "stacking_logistic": "Stacking (LR)",
                    "robust_rank_aggregation": "RRA",
                    "mean_rank_aggregation": "Mean Rank",
                }
                x = np.arange(len(placeholder_methods))
                values = np.zeros(len(placeholder_methods), dtype=float)
                plt.figure(figsize=(10, 5))
                bars = plt.bar(x, values, color="#A0A0A0", alpha=0.9)
                for bar in bars:
                    plt.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        0.01,
                        "NA",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
                placeholder_labels = [
                    method_display.get(value, value) for value in placeholder_methods
                ]
                plt.xticks(x, placeholder_labels, rotation=15, ha="right")
                plt.ylabel("Mean AUC across test splits")
                plt.title("Feature Importance Method Comparison (Awaiting Subset Evaluation)")
                plt.ylim(0.0, 1.0)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(FINAL_PLOTS_DIR, "feature_importance_methods_bar.png"),
                    dpi=180,
                )
                plt.close()
                print("WARNING: subset evaluations not available; generated placeholder feature method plot.")
        else:
            placeholder_methods = [
                "hard_voting",
                "soft_voting",
                "performance_weighted_soft",
                "k_filter_voting",
                "stacking_logistic",
                "robust_rank_aggregation",
                "mean_rank_aggregation",
            ]
            method_display = {
                "hard_voting": "Hard Voting",
                "soft_voting": "Soft Voting",
                "performance_weighted_soft": "Weighted Soft",
                "k_filter_voting": "Top-K Voting",
                "stacking_logistic": "Stacking (LR)",
                "robust_rank_aggregation": "RRA",
                "mean_rank_aggregation": "Mean Rank",
            }
            x = np.arange(len(placeholder_methods))
            values = np.zeros(len(placeholder_methods), dtype=float)
            plt.figure(figsize=(10, 5))
            bars = plt.bar(x, values, color="#A0A0A0", alpha=0.9)
            for bar in bars:
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    0.01,
                    "NA",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            placeholder_labels = [
                method_display.get(value, value) for value in placeholder_methods
            ]
            plt.xticks(x, placeholder_labels, rotation=15, ha="right")
            plt.ylabel("Mean AUC across test splits")
            plt.title("Feature Importance Method Comparison (Awaiting Subset Evaluation)")
            plt.ylim(0.0, 1.0)
            plt.tight_layout()
            plt.savefig(
                os.path.join(FINAL_PLOTS_DIR, "feature_importance_methods_bar.png"),
                dpi=180,
            )
            plt.close()
            print("WARNING: subset summary is empty; generated placeholder feature method plot.")
    else:
        placeholder_methods = [
            "hard_voting",
            "soft_voting",
            "performance_weighted_soft",
            "k_filter_voting",
            "stacking_logistic",
            "robust_rank_aggregation",
            "mean_rank_aggregation",
        ]
        method_display = {
            "hard_voting": "Hard Voting",
            "soft_voting": "Soft Voting",
            "performance_weighted_soft": "Weighted Soft",
            "k_filter_voting": "Top-K Voting",
            "stacking_logistic": "Stacking (LR)",
            "robust_rank_aggregation": "RRA",
            "mean_rank_aggregation": "Mean Rank",
        }
        x = np.arange(len(placeholder_methods))
        values = np.zeros(len(placeholder_methods), dtype=float)
        plt.figure(figsize=(10, 5))
        bars = plt.bar(x, values, color="#A0A0A0", alpha=0.9)
        for bar in bars:
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                0.01,
                "NA",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        placeholder_labels = [
            method_display.get(value, value) for value in placeholder_methods
        ]
        plt.xticks(x, placeholder_labels, rotation=15, ha="right")
        plt.ylabel("Mean AUC across test splits")
        plt.title("Feature Importance Method Comparison (Awaiting Subset Evaluation)")
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(
            os.path.join(FINAL_PLOTS_DIR, "feature_importance_methods_bar.png"),
            dpi=180,
        )
        plt.close()
        print("WARNING: subset evaluation summary missing; generated placeholder feature method plot.")

    if single_model_accuracy_df is not None and ensemble_df_for_summary is not None:
        split_filter = ["test_set_ood_1", "test_set_ood_2", "test_set_id"]
        before_df = single_model_accuracy_df[
            single_model_accuracy_df["split"].isin(split_filter)
        ].copy()
        after_df = ensemble_df_for_summary[
            ensemble_df_for_summary["split"].isin(split_filter)
        ].copy()

        if not before_df.empty and not after_df.empty:
            method_display = {
                "hard_voting": "Hard Voting",
                "soft_voting": "Soft Voting",
                "performance_weighted_soft": "Weighted Soft",
                "k_filter_voting": "Top-K Voting",
                "stacking_logistic": "Stacking (LR)",
            }

            split_order = ["test_set_ood_1", "test_set_ood_2", "test_set_id"]
            split_display = {
                "test_set_ood_1": "OOD-1",
                "test_set_ood_2": "OOD-2",
                "test_set_id": "ID",
            }

            labels = []
            before_values = []
            after_values = []

            for split_name in split_order:
                split_before_df = before_df[before_df["split"] == split_name]
                split_after_df = after_df[after_df["split"] == split_name]
                if split_before_df.empty or split_after_df.empty:
                    continue

                best_before_row = split_before_df.loc[
                    split_before_df["accuracy_percent"].idxmax()
                ]
                best_after_row = split_after_df.loc[
                    split_after_df["accuracy_percent"].idxmax()
                ]
                before_model_name = str(best_before_row["model"])
                after_method_name = method_display.get(
                    str(best_after_row["method"]), str(best_after_row["method"])
                )
                labels.append(split_display.get(split_name, split_name))
                before_values.append(float(best_before_row["accuracy_percent"]))
                after_values.append(float(best_after_row["accuracy_percent"]))

            generated_before_after_plot = False
            if labels:
                labels.append("Mean\nAcross Splits")
                before_values.append(float(np.mean(before_values)))
                after_values.append(float(np.mean(after_values)))

                x = np.arange(len(labels))
                width = 0.36
                plt.figure(figsize=(13, 6.2))
                bars_before = plt.bar(
                    x - (width / 2.0),
                    before_values,
                    width=width,
                    color="#4C72B0",
                    alpha=0.94,
                    label="Best Single Model",
                )
                bars_after = plt.bar(
                    x + (width / 2.0),
                    after_values,
                    width=width,
                    color="#55A868",
                    alpha=0.94,
                    label="Best Ensemble Method",
                )

                for bars, values in [(bars_before, before_values), (bars_after, after_values)]:
                    for bar, value in zip(bars, values):
                        plt.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            value + 0.2,
                            f"{value:.1f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

                for index, (before_value, after_value) in enumerate(
                    zip(before_values, after_values)
                ):
                    delta = after_value - before_value
                    delta_color = "#2E8B57" if delta >= 0 else "#B22222"
                    plt.text(
                        x[index],
                        max(before_value, after_value) + 1.0,
                        f"Delta {delta:+.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color=delta_color,
                        fontweight="bold",
                    )

                plt.xticks(x, labels, rotation=0, ha="center")
                plt.ylabel("Accuracy (%)")
                plt.title("Best Single Model vs Best Ensemble (Per Split)")
                plt.legend(loc="upper left", frameon=False)
                y_max = max(before_values + after_values)
                plt.ylim(0.0, max(100.0, y_max + 4.0))
                plt.figtext(
                    0.5,
                    0.01,
                    "Split key: OOD-1/OOD-2 = out-of-distribution test sets, ID = in-distribution test set.",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
                plt.tight_layout(rect=[0, 0.05, 1, 0.98])
                generated_before_after_plot = True
            else:
                print("WARNING: no overlapping split data for before/after ensemble chart.")

            if generated_before_after_plot:
                plt.savefig(
                    os.path.join(FINAL_PLOTS_DIR, "before_after_ensemble_accuracy_bar.png"),
                    dpi=180,
                )
                plt.close()
        else:
            print("WARNING: missing data for before/after ensemble chart.")
    else:
        print("WARNING: accuracy summaries missing; skipping before/after ensemble chart.")

    if single_model_accuracy_df is not None:
        split_filter = ["test_set_ood_1", "test_set_ood_2", "test_set_id"]
        split_display = {
            "test_set_ood_1": "OOD-1",
            "test_set_ood_2": "OOD-2",
            "test_set_id": "ID",
        }
        single_test_df = single_model_accuracy_df[
            single_model_accuracy_df["split"].isin(split_filter)
        ].copy()
        if not single_test_df.empty:
            mean_accuracy_df = (
                single_test_df.groupby("model", as_index=False)["accuracy_percent"]
                .mean()
                .sort_values(["accuracy_percent", "model"], ascending=[False, True])
            )
            best_accuracy_model = str(mean_accuracy_df.iloc[0]["model"])
            best_model_df = single_test_df[
                single_test_df["model"] == best_accuracy_model
            ].copy()

            x_labels = []
            values = []
            for split_name in split_filter:
                split_rows = best_model_df[best_model_df["split"] == split_name]
                if split_rows.empty:
                    continue
                x_labels.append(split_display.get(split_name, split_name))
                values.append(float(split_rows.iloc[0]["accuracy_percent"]))

            if values:
                mean_value = float(np.mean(values))
                x = np.arange(len(values))
                plt.figure(figsize=(8, 5))
                bars = plt.bar(x, values, color="#4C72B0", alpha=0.94)
                for bar, value in zip(bars, values):
                    plt.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        value + 0.2,
                        f"{value:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )
                plt.axhline(
                    mean_value,
                    color="#C44E52",
                    linestyle="--",
                    linewidth=1.4,
                    label=f"Mean={mean_value:.1f}%",
                )
                plt.xticks(x, x_labels, rotation=0, ha="center")
                plt.ylabel("Accuracy (%)")
                plt.title(f"Single Best Model Accuracy ({best_accuracy_model})")
                plt.legend(loc="upper left", frameon=False)
                plt.ylim(0.0, max(100.0, max(values) + 3.0))
                plt.figtext(
                    0.5,
                    0.01,
                    "Split key: OOD-1/OOD-2 = out-of-distribution test sets, ID = in-distribution test set.",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
                plt.tight_layout(rect=[0, 0.06, 1, 1])
                plt.savefig(
                    os.path.join(FINAL_PLOTS_DIR, "single_best_model_accuracy_bar.png"),
                    dpi=180,
                )
                plt.close()
        else:
            print("WARNING: single-model summary has no test rows; skipping best-model chart.")
    else:
        print("WARNING: single-model summary missing; skipping best-model chart.")

    best_feature_method = None
    best_feature_method_score = None
    top_feature_methods_df = None
    if os.path.exists(subset_summary_path):
        with open(subset_summary_path, "r") as handle:
            subset_rows = json.load(handle)
        subset_df_for_pie = pd.DataFrame(subset_rows)
        if not subset_df_for_pie.empty:
            valid_pie_mask = subset_df_for_pie["status"].isin(
                ["success", "cached", "reused_subset_result"]
            )
            valid_pie_df = subset_df_for_pie[valid_pie_mask].copy()
            if not valid_pie_df.empty:
                valid_pie_df["overall_auc"] = valid_pie_df[
                    ["mean_auc_test_set_ood_1", "mean_auc_test_set_ood_2", "mean_auc_test_set_id"]
                ].mean(axis=1, skipna=True)
                valid_pie_df = valid_pie_df.sort_values(
                    ["overall_auc", "method"], ascending=[False, True]
                )
                top_feature_methods_df = valid_pie_df.copy()
                best_feature_method = str(valid_pie_df.iloc[0]["method"])
                best_feature_method_score = float(valid_pie_df.iloc[0]["overall_auc"])

    if best_feature_method is not None:
        ranking_path = os.path.join(
            FEATURE_IMPORTANCE_DIR,
            "aggregated",
            best_feature_method,
            "feature_ranking.csv",
        )
        if os.path.exists(ranking_path):
            ranking_df = pd.read_csv(ranking_path)
            ranking_df = ranking_df[
                ~ranking_df["feature"].isin(["age", "sex"])
            ].copy()
            ranking_df = ranking_df.sort_values("rank", ascending=True)
            if not ranking_df.empty:
                all_feature_df = ranking_df.sort_values("score", ascending=True)
                all_feature_labels = all_feature_df["feature"].astype(str).tolist()
                all_feature_display_labels = [
                    format_feature_label(feature_name) for feature_name in all_feature_labels
                ]
                all_feature_scores = all_feature_df["score"].astype(float).to_numpy()
                all_colors = [
                    "#4C72B0" if rank_value <= FEATURE_SUBSET_SIZE else "#AFC4E2"
                    for rank_value in all_feature_df["rank"].astype(int).tolist()
                ]

                plt.figure(figsize=(12, max(6.0, 0.27 * len(all_feature_labels))))
                bars = plt.barh(
                    all_feature_display_labels,
                    all_feature_scores,
                    color=all_colors,
                    alpha=0.95,
                )
                for bar, score_value in zip(bars, all_feature_scores):
                    if score_value < 0.01:
                        continue
                    plt.text(
                        score_value + 0.001,
                        bar.get_y() + bar.get_height() / 2.0,
                        f"{score_value:.3f}",
                        ha="left",
                        va="center",
                        fontsize=7,
                    )
                plt.xlabel("Aggregated Feature Score")
                plt.ylabel("Feature")
                plt.title(
                    f"All Features ({best_feature_method}, mean AUC={best_feature_method_score:.3f})"
                )
                plt.figtext(
                    0.5,
                    0.01,
                    "Feature names are mapped in results/feature_importance/feature_definitions.csv",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
                plt.tight_layout(rect=[0, 0.03, 1, 1])
                plt.savefig(
                    os.path.join(FINAL_PLOTS_DIR, "best_features_all_bar.png"),
                    dpi=180,
                )
                plt.close()

                top_feature_count = min(10, len(ranking_df))
                pie_df = ranking_df.sort_values("score", ascending=False).copy()
                pie_labels = [
                    format_feature_label(feature_name)
                    for feature_name in pie_df["feature"].head(top_feature_count).astype(str).tolist()
                ]
                pie_scores = pie_df["score"].head(top_feature_count).astype(float).to_numpy()
                other_score = float(
                    pie_df["score"].iloc[top_feature_count:].astype(float).sum()
                )
                if len(pie_df) > top_feature_count and other_score > 0:
                    pie_labels.append("other_features")
                    pie_scores = np.append(pie_scores, other_score)
                if np.nansum(pie_scores) <= 0:
                    pie_scores = np.ones(len(pie_labels), dtype=float)

                plt.figure(figsize=(10, 7))
                colors = plt.cm.tab20(np.linspace(0, 1, len(pie_labels)))
                wedges, texts, autotexts = plt.pie(
                    pie_scores,
                    labels=pie_labels,
                    autopct="%1.1f%%",
                    startangle=120,
                    colors=colors,
                    wedgeprops={"edgecolor": "white", "linewidth": 1.0},
                    textprops={"fontsize": 9},
                )
                for autotext in autotexts:
                    autotext.set_color("black")
                    autotext.set_fontsize(8)
                plt.title(
                    f"Top Features + Other ({best_feature_method}, all features included)"
                )
                plt.tight_layout()
                plt.savefig(
                    os.path.join(FINAL_PLOTS_DIR, "best_features_pie.png"),
                    dpi=180,
                )
                plt.close()

                def prepare_pie_parts(feature_df, score_column, label):
                    local_df = feature_df[
                        ~feature_df["feature"].isin(["age", "sex"])
                    ].copy()
                    if local_df.empty or score_column not in local_df.columns:
                        return None
                    local_df[score_column] = (
                        local_df[score_column]
                        .astype(float)
                        .replace([np.inf, -np.inf], np.nan)
                        .fillna(0.0)
                    )
                    local_df = local_df.sort_values(score_column, ascending=False)
                    if local_df.empty:
                        return None

                    local_top_count = min(6, len(local_df))
                    local_labels = [
                        format_feature_label(feature_name)
                        for feature_name in local_df["feature"].head(local_top_count).astype(str).tolist()
                    ]
                    local_values = (
                        local_df[score_column].head(local_top_count).to_numpy(dtype=float)
                    )
                    local_other = float(
                        local_df[score_column]
                        .iloc[local_top_count:]
                        .astype(float)
                        .sum()
                    )
                    if len(local_df) > local_top_count and local_other > 0:
                        local_labels.append("other_features")
                        local_values = np.append(local_values, local_other)

                    if np.nansum(local_values) <= 0:
                        local_values = np.ones(len(local_labels), dtype=float)

                    return {"label": label, "labels": local_labels, "values": local_values}

                top_models_for_pie = []
                if single_model_accuracy_df is not None:
                    split_filter = ["test_set_ood_1", "test_set_ood_2", "test_set_id"]
                    single_test_df = single_model_accuracy_df[
                        single_model_accuracy_df["split"].isin(split_filter)
                    ].copy()
                    if not single_test_df.empty:
                        mean_accuracy_df = (
                            single_test_df.groupby("model", as_index=False)["accuracy_percent"]
                            .mean()
                            .sort_values(["accuracy_percent", "model"], ascending=[False, True])
                        )
                        top_models_for_pie = mean_accuracy_df["model"].head(5).astype(str).tolist()

                pie_panels = []
                for model_name in top_models_for_pie:
                    model_feature_path = os.path.join(
                        FEATURE_IMPORTANCE_DIR,
                        "models",
                        model_name,
                        "feature_importance.csv",
                    )
                    if not os.path.exists(model_feature_path):
                        continue
                    model_feature_df = pd.read_csv(model_feature_path)
                    prepared_model_panel = prepare_pie_parts(
                        model_feature_df,
                        "importance",
                        f"{model_name} (single model)",
                    )
                    if prepared_model_panel is not None:
                        pie_panels.append(prepared_model_panel)

                prepared_aggregated_panel = prepare_pie_parts(
                    ranking_df,
                    "score",
                    f"Aggregated best ({best_feature_method})",
                )
                if prepared_aggregated_panel is not None:
                    pie_panels.append(prepared_aggregated_panel)

                if pie_panels:
                    n_panels = len(pie_panels)
                    n_cols = min(3, n_panels)
                    n_rows = int(np.ceil(n_panels / n_cols))
                    fig, axes = plt.subplots(
                        n_rows,
                        n_cols,
                        figsize=(5.0 * n_cols, 4.0 * n_rows),
                    )
                    if isinstance(axes, np.ndarray):
                        axes = axes.flatten()
                    else:
                        axes = [axes]

                    for panel_index, panel in enumerate(pie_panels):
                        axis = axes[panel_index]
                        panel_colors = plt.cm.tab20(
                            np.linspace(0, 1, len(panel["labels"]))
                        )
                        _, _, autotexts = axis.pie(
                            panel["values"],
                            labels=panel["labels"],
                            autopct="%1.1f%%",
                            startangle=120,
                            colors=panel_colors,
                            wedgeprops={"edgecolor": "white", "linewidth": 1.0},
                            textprops={"fontsize": 7},
                        )
                        for autotext in autotexts:
                            autotext.set_color("black")
                            autotext.set_fontsize(7)
                        axis.set_title(panel["label"], fontsize=9)

                    for panel_index in range(len(pie_panels), len(axes)):
                        axes[panel_index].axis("off")

                    fig.suptitle(
                        "Feature Importance: Top 5 Single Models + Aggregated Best",
                        fontsize=12,
                    )
                    fig.tight_layout(rect=[0, 0, 1, 0.95])
                    fig.savefig(
                        os.path.join(FINAL_PLOTS_DIR, "best_features_multi_pie.png"),
                        dpi=180,
                    )
                    plt.close(fig)

                if top_feature_methods_df is not None and not top_feature_methods_df.empty:
                    method_display = {
                        "hard_voting": "Hard Voting",
                        "soft_voting": "Soft Voting",
                        "performance_weighted_soft": "Weighted Soft",
                        "k_filter_voting": "Top-K Voting",
                        "stacking_logistic": "Stacking (LR)",
                        "robust_rank_aggregation": "RRA",
                        "mean_rank_aggregation": "Mean Rank",
                    }
                    method_panels = []
                    top_methods = top_feature_methods_df["method"].head(4).astype(str).tolist()
                    for method_name in top_methods:
                        method_ranking_path = os.path.join(
                            FEATURE_IMPORTANCE_DIR,
                            "aggregated",
                            method_name,
                            "feature_ranking.csv",
                        )
                        if not os.path.exists(method_ranking_path):
                            continue
                        method_ranking_df = pd.read_csv(method_ranking_path)
                        method_panel = prepare_pie_parts(
                            method_ranking_df,
                            "score",
                            method_display.get(method_name, method_name),
                        )
                        if method_panel is not None:
                            method_panels.append(method_panel)

                    if method_panels:
                        n_panels = len(method_panels)
                        n_cols = min(2, n_panels)
                        n_rows = int(np.ceil(n_panels / n_cols))
                        fig, axes = plt.subplots(
                            n_rows,
                            n_cols,
                            figsize=(5.2 * n_cols, 4.0 * n_rows),
                        )
                        if isinstance(axes, np.ndarray):
                            axes = axes.flatten()
                        else:
                            axes = [axes]

                        for panel_index, panel in enumerate(method_panels):
                            axis = axes[panel_index]
                            panel_colors = plt.cm.tab20(
                                np.linspace(0, 1, len(panel["labels"]))
                            )
                            _, _, autotexts = axis.pie(
                                panel["values"],
                                labels=panel["labels"],
                                autopct="%1.1f%%",
                                startangle=120,
                                colors=panel_colors,
                                wedgeprops={"edgecolor": "white", "linewidth": 1.0},
                                textprops={"fontsize": 7},
                            )
                            for autotext in autotexts:
                                autotext.set_color("black")
                                autotext.set_fontsize(7)
                            axis.set_title(panel["label"], fontsize=9)

                        for panel_index in range(len(method_panels), len(axes)):
                            axes[panel_index].axis("off")

                        fig.suptitle(
                            "Top Feature Aggregation Methods (by subset AUC)",
                            fontsize=12,
                        )
                        fig.tight_layout(rect=[0, 0, 1, 0.95])
                        fig.savefig(
                            os.path.join(FINAL_PLOTS_DIR, "best_features_methods_pie.png"),
                            dpi=180,
                        )
                        plt.close(fig)
            else:
                print("WARNING: ranking file has no features for pie chart.")
        else:
            print("WARNING: best method ranking file missing for pie chart.")
    else:
        print("WARNING: no successful feature method found for pie chart.")

    print("Final summary plots completed.")




###############################################################################
#                                  4. Main                                    #
###############################################################################
def main():
    ensure_inputs()
    ensure_validation_split()
    run_model_sweep()
    export_probabilities()
    run_ensemble_methods()
    run_probability_calibration()
    run_abstain_uncertainty_logic()
    run_feature_importance_analysis()
    run_final_summary_plots()


if __name__ == "__main__":
    main()
