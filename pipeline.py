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
ACCURACY_PATH = os.path.join(RESULTS_DIR, "accuracy_summary.json")
ENSEMBLES_DIR = os.path.join(RESULTS_DIR, "ensembles")
CALIBRATION_DIR = os.path.join(RESULTS_DIR, "calibration")
ABSTAIN_DIR = os.path.join(RESULTS_DIR, "abstain_uncertainty")

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
    "performance_weighted_soft",
    "soft_voting",
    "k_filter_voting",
    "hard_voting",
]
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


def run_ensemble_methods():
    if not os.path.isdir(RESULTS_DIR):
        print("ERROR: results/ directory not found; skipping ensembles.")
        return

    os.makedirs(ENSEMBLES_DIR, exist_ok=True)

    summary_rows = []
    weights_used = {}
    selected_models = {}

    model_dirs = [
        name
        for name in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, name)) and name != "ensembles"
    ]
    model_dirs.sort()

    for split_name, _ in DATASETS:
        base_df = None
        base_keys = None
        model_probabilities = {}

        for model_name in model_dirs:
            prob_path = os.path.join(RESULTS_DIR, model_name, f"probabilities_{split_name}.csv")
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
    for source_name in sorted(os.listdir(RESULTS_DIR)):
        source_path = os.path.join(RESULTS_DIR, source_name)
        if not os.path.isdir(source_path):
            continue
        if source_name in {"ensembles", "calibration"}:
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


if __name__ == "__main__":
    main()
