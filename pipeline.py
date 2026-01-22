# Source: Local pipeline entry point for the capstone

""" ***************************************************************************
# * File Description:                                                         *
# * Pipeline entry point for capstone stages                                  *
# *                                                                           *
# * The contents of this script are:                                          *
# * 1. Importing Libraries                                                    *
# * 2. Configurations                                                         *
# * 3. Helper Functions                                                       *
# * 4. Smoke Test (single model sweep)                                        *
# * 5. Main                                                                   *
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
import datetime
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

import pandas as pd


###############################################################################
#                             2. Configurations                               #
###############################################################################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_SWEEP_PATH = os.path.join(BASE_DIR, "model_sweep.py")
SYNTHETIC_SCRIPT_PATH = os.path.join(BASE_DIR, "scripts", "make_synthetic_data.py")

FEATURES_PATH = os.path.join(
    BASE_DIR, "data", "cleaned", "selected_features_f1_final_new_common_nate.txt"
)
TRAIN_PATH = os.path.join(BASE_DIR, "data", "data_ood_id", "train_data.csv")
TEST_OOD_1_PATH = os.path.join(BASE_DIR, "data", "data_ood_id", "test_set_ood_1.csv")
TEST_OOD_2_PATH = os.path.join(BASE_DIR, "data", "data_ood_id", "test_set_ood_2.csv")
TEST_ID_PATH = os.path.join(BASE_DIR, "data", "data_ood_id", "test_set_id.csv")

RUNS_DIR = os.path.join(BASE_DIR, "runs")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")


###############################################################################
#                             3. Helper Functions                             #
###############################################################################
def read_text(path):
    with open(path, "r") as f:
        return f.read()


def print_progress(step, total, message):
    print(f"[{step}/{total}] {message}")


def extract_classifiers(text):
    # Scan line by line for classifiers.update({ ... }) with a quoted key
    keys = []
    key_lines = {}
    pattern = re.compile(r'classifiers\.update\(\{\s*["\']([^"\']+)["\']\s*:')
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        match = pattern.search(line)
        if match:
            key = match.group(1)
            keys.append(key)
            key_lines[key] = line
    return keys, key_lines


def extract_output_patterns(text):
    json_paths = re.findall(r'["\'](results/[^"\']+\.json)["\']', text)
    joblib_paths = re.findall(r'["\'](models/[^"\']+\.joblib)["\']', text)
    return json_paths, joblib_paths


def validate_inputs():
    issues = []
    missing_files = []

    if not os.path.exists(FEATURES_PATH):
        missing_files.append(FEATURES_PATH)
    elif os.path.getsize(FEATURES_PATH) == 0:
        issues.append(f"Feature list empty: {FEATURES_PATH}")

    for path in [TRAIN_PATH, TEST_OOD_1_PATH, TEST_OOD_2_PATH, TEST_ID_PATH]:
        if not os.path.exists(path):
            missing_files.append(path)

    if missing_files:
        return False, issues, missing_files, None

    feature_names = [line.strip() for line in read_text(FEATURES_PATH).splitlines() if line.strip()]
    if not feature_names:
        issues.append("No valid feature names in feature list.")

    required_columns = [
        "filename",
        "patient_id",
        "occuluded_artery",
        "age",
        "sex",
        "left_circumflex",
        "right_coronary_artery",
        "left_anterior_descending",
    ] + feature_names

    def check_columns(path):
        try:
            df = pd.read_csv(path, nrows=1)
        except Exception:
            issues.append(f"Cannot read: {path}")
            return
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            issues.append(f"{os.path.basename(path)} missing columns: {', '.join(missing)}")

    check_columns(TRAIN_PATH)
    check_columns(TEST_OOD_1_PATH)
    check_columns(TEST_OOD_2_PATH)
    check_columns(TEST_ID_PATH)

    data_info = None
    if not issues:
        data_info = {
            "train_rows": len(pd.read_csv(TRAIN_PATH)),
            "test_ood_1_rows": len(pd.read_csv(TEST_OOD_1_PATH)),
            "test_ood_2_rows": len(pd.read_csv(TEST_OOD_2_PATH)),
            "test_id_rows": len(pd.read_csv(TEST_ID_PATH)),
        }

    return len(issues) == 0, issues, missing_files, data_info


def build_single_model_text(model_text, selected_key, selected_line):
    lines = model_text.splitlines()
    out_lines = []
    in_classifiers_block = False
    inserted = False

    for line in lines:
        if not in_classifiers_block and line.strip() == "classifiers = {}":
            in_classifiers_block = True
            out_lines.append(line)
            if not inserted:
                out_lines.append(selected_line)
                inserted = True
            continue

        if in_classifiers_block:
            if "5. Hyper-parameters" in line:
                in_classifiers_block = False
                out_lines.append(line)
                continue

            stripped = line.lstrip()
            if "classifiers.update(" in line and not stripped.startswith("#"):
                continue

            out_lines.append(line)
            continue

        out_lines.append(line)

    if not inserted:
        raise ValueError(f"Could not insert classifier line for: {selected_key}")

    return "\n".join(out_lines) + "\n"


def enable_sweep_progress(model_text):
    pattern = re.compile(r"\bverbose\s*=\s*1\b")
    if pattern.search(model_text):
        return pattern.sub("verbose=10", model_text, count=1)
    return model_text


def find_new_files(root, suffix, since_ts):
    matches = []
    if not os.path.isdir(root):
        return matches

    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith(suffix):
                continue
            path = os.path.join(dirpath, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if mtime >= since_ts:
                matches.append(path)
    matches.sort()
    return matches


def run_subprocess(cmd, cwd, log_path=None):
    if log_path:
        with open(log_path, "w") as log_file:
            result = subprocess.run(cmd, cwd=cwd, stdout=log_file, stderr=log_file)
    else:
        result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


###############################################################################
#                             4. Smoke Test (single model sweep)              #
###############################################################################
def run_smoketest():
    print("Running pipeline smoke test (single model sweep)")
    total_steps = 6
    step = 0

    def advance(message):
        nonlocal step
        step += 1
        print_progress(step, total_steps, message)

    if not os.path.exists(MODEL_SWEEP_PATH):
        print(f"ERROR: model_sweep.py not found at {MODEL_SWEEP_PATH}")
        sys.exit(1)

    if not os.path.exists(SYNTHETIC_SCRIPT_PATH):
        print(f"ERROR: scripts/make_synthetic_data.py not found at {SYNTHETIC_SCRIPT_PATH}")
        sys.exit(1)

    model_text = read_text(MODEL_SWEEP_PATH)
    _ = read_text(SYNTHETIC_SCRIPT_PATH)

    json_patterns, joblib_patterns = extract_output_patterns(model_text)
    if json_patterns or joblib_patterns:
        print("Expected output patterns (from model_sweep.py):")
        for path in sorted(set(json_patterns)):
            print(f"- {path}")
        for path in sorted(set(joblib_patterns)):
            print(f"- {path}")

    valid, issues, missing_files, data_info = validate_inputs()
    advance("Initial input validation complete.")
    if not valid and missing_files:
        print("Missing required files. Generating synthetic data...")
        code = run_subprocess([sys.executable, SYNTHETIC_SCRIPT_PATH], cwd=BASE_DIR)
        if code != 0:
            print("ERROR: synthetic data generation failed.")
            sys.exit(code)
        valid, issues, missing_files, data_info = validate_inputs()
        advance("Synthetic data generation complete.")
    else:
        advance("Data files present, no generation needed.")

    if missing_files:
        print("ERROR: missing required files after generation:")
        for path in missing_files:
            print(f"- {path}")
        sys.exit(1)

    if not valid:
        print("ERROR: input validation failed:")
        for issue in issues:
            print(f"- {issue}")
        sys.exit(1)

    classifier_keys, classifier_lines = extract_classifiers(model_text)
    if not classifier_keys:
        print("ERROR: no classifiers found in model_sweep.py")
        sys.exit(1)

    if "LDA" in classifier_keys:
        selected_key = "LDA"
    elif "Random Forest" in classifier_keys:
        selected_key = "Random Forest"
    else:
        selected_key = classifier_keys[0]
    if selected_key not in classifier_lines:
        print(f"ERROR: selected classifier not found: {selected_key}")
        sys.exit(1)

    print(f"Selected classifier: {selected_key}")
    advance("Classifier selection complete.")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, timestamp)
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "sweep.log")
    start_ts = time.time()
    temp_path = None
    result_code = None
    heartbeat_interval = 30
    try:
        patched_text = build_single_model_text(
            model_text, selected_key, classifier_lines[selected_key]
        )
        patched_text = enable_sweep_progress(patched_text)
        temp_fd, temp_path = tempfile.mkstemp(prefix="model_sweep_single_", suffix=".py")
        with os.fdopen(temp_fd, "w") as temp_file:
            temp_file.write(patched_text)
        advance("Temporary sweep script prepared.")

        env = os.environ.copy()
        warning_filter = r"ignore:.*sklearn\.utils\.parallel\.delayed.*:UserWarning"
        existing = env.get("PYTHONWARNINGS", "")
        if existing:
            env["PYTHONWARNINGS"] = f"{existing},{warning_filter}"
        else:
            env["PYTHONWARNINGS"] = warning_filter

        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                [sys.executable, "-u", temp_path],
                cwd=BASE_DIR,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None

            sweep_start_ts = time.time()
            stop_event = threading.Event()

            def heartbeat():
                while not stop_event.wait(heartbeat_interval):
                    elapsed = int(time.time() - sweep_start_ts)
                    print(f"Sweep running... {elapsed}s elapsed")

            heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
            heartbeat_thread.start()
            try:
                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()
            finally:
                stop_event.set()
                heartbeat_thread.join(timeout=heartbeat_interval)

            result_code = process.wait()
        advance("Model sweep completed.")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    if result_code is None or result_code != 0:
        print("ERROR: model sweep failed. See log for details:")
        print(f"- {log_path}")
        sys.exit(result_code if result_code is not None else 1)

    new_json = find_new_files(RESULTS_DIR, ".json", start_ts)
    new_joblib = find_new_files(MODELS_DIR, ".joblib", start_ts)

    if not new_json:
        print("ERROR: no results JSON found after the run.")
        print(f"- {log_path}")
        sys.exit(1)
    if not new_joblib:
        print("ERROR: no model artifacts found after the run.")
        print(f"- {log_path}")
        sys.exit(1)

    advance("Artifacts verified.")

    print("Run summary:")
    print(f"- Model: {selected_key}")
    print(f"- train_data.csv rows: {data_info['train_rows']}")
    print(f"- test_set_ood_1.csv rows: {data_info['test_ood_1_rows']}")
    print(f"- test_set_ood_2.csv rows: {data_info['test_ood_2_rows']}")
    print(f"- test_set_id.csv rows: {data_info['test_id_rows']}")
    if new_json:
        print("- Results JSON:")
        for path in new_json:
            print(f"  - {os.path.relpath(path, BASE_DIR)}")
    else:
        print("- Results JSON: none found")
    if new_joblib:
        print("- Model artifacts:")
        for path in new_joblib:
            print(f"  - {os.path.relpath(path, BASE_DIR)}")
    else:
        print("- Model artifacts: none found")
    print(f"- Log: {os.path.relpath(log_path, BASE_DIR)}")


###############################################################################
#                                  5. Main                                    #
###############################################################################
def main():
    run_smoketest()


if __name__ == "__main__":
    main()
