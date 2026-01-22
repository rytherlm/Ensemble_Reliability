""" ***************************************************************************
# * File Description:                                                         *
# * Workflow for model building                                               *
# *                                                                           *
# * The contents of this script are:                                          *
# * 1. Importing Libraries                                                    *
# * 2. Get data                                                               *
# * 3. Create train and test set                                              *
# * 4. Classifiers                                                            *
# * 5. Hyper-parameters                                                       *
# * 6. Feature Selection: Removing highly correlated features                 *
# * 7. Tuning a classifier to use with RFECV                                  *
# * 8. Custom pipeline object to use with RFECV                               *
# * 9. Feature Selection: Recursive Feature Selection with Cross Validation   *
# * 10. Performance Curve                                                     *
# * 11. Feature Selection: Recursive Feature Selection                        *
# * 12. Visualizing Selected Features Importance                              *
# * 13. Classifier Tuning and Evaluation                                      *
# * 14. Visualing Results                                                     *
# *                                                                           *
# * --------------------------------------------------------------------------*
# * AUTHORS(S): Frank Ceballos <frank.ceballos89@gmail.com>                   *
# * --------------------------------------------------------------------------*
# * DATE CREATED: June 26, 2019                                               *
# * --------------------------------------------------------------------------*
# * NOTES: None                                                               *
# * ************************************************************************"""


###############################################################################
#                          1. Importing Libraries                             #
###############################################################################
# For reading, visualizing, and preprocessing data
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
# from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score

# Classifiers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
import joblib
import os
import json
from sklearn.utils.class_weight import compute_sample_weight
import inspect
import copy
from sklearn.preprocessing import StandardScaler
import ast

SEED = 42
scoring = 'f1'
n_jobs = 8
RESULTS_BASE_DIR = os.path.join("results")
MODELS_BASE_DIR = RESULTS_BASE_DIR

def create_results_structure(sets=("train", "test_1", "test_2", "test_3")):
    metrics = ["sensitivity", "specificity", "PPV", "NPV", "AUC"]
    arteries = ["overall", "left_circumflex", "right_coronary_artery", "left_anterior_descending"]

    template = {artery: {metric: None for metric in metrics} for artery in arteries}
    results = {set_name: template.copy() for set_name in sets}
    results = {set_name: copy.deepcopy(template) for set_name in sets}

    return results

def sensitivity_specificity_ppv_npv(y_true, y_pred, threshold=0.5):
    y_hat = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    ppv = tp / (tp + fp) if (tp + fp) > 0 else None
    npv = tn / (tn + fn) if (tn + fn) > 0 else None
    return sensitivity, specificity, ppv, npv

def print_overall_results(results, y_true, y_pred, group="test", threshold=0.5):
    sens, spec, ppv, npv = sensitivity_specificity_ppv_npv(y_true, y_pred, threshold)
    if ppv is None:
        ppv = -10.0
    if npv is None:
        npv = -10.0
    if sens is None:
        sens = -10.0
    if spec is None:
        spec = -10.0
    auc = roc_auc_score(y_true, y_pred)

    results[group]["overall"]["sensitivity"] = float(sens)
    results[group]["overall"]["specificity"] = float(spec)
    results[group]["overall"]["PPV"] = float(ppv)
    results[group]["overall"]["NPV"] = float(npv)
    results[group]["overall"]["AUC"] = float(auc)

    print(f"Overall {group}: AUC={auc}, sensitivity={sens}, specificity={spec}, PPV={ppv}, NPV={npv}")


def print_results_by_artery(results, X, model, arteries, features, group="", threshold=0.5):
    print(f"{group} results")
    for artery in arteries:
        other_arteries = [a for a in arteries if a != artery]
        artery_1_mask = X[other_arteries[0]] == 0
        artery_2_mask = X[other_arteries[1]] == 0
        self_mask = X[artery] == 1
        artery_mask = ((artery_1_mask & artery_2_mask) | self_mask)
        X_new = X[artery_mask]
        y_pred = model.predict_proba(X_new[features])[:, 1]
        auc = roc_auc_score(X_new[artery], y_pred)
        sens, spec, ppv, npv = sensitivity_specificity_ppv_npv(
            X_new[artery], y_pred, threshold=threshold)
        if ppv is None:
            ppv = -10.0
        if npv is None:
            npv = -10.0
        if sens is None:
            sens = -10.0
        if spec is None:
            spec = -10.0
        results[group][artery]["AUC"] = float(auc)
        results[group][artery]["sensitivity"] = float(sens)
        results[group][artery]["specificity"] = float(spec)
        results[group][artery]["PPV"] = float(ppv)
        results[group][artery]["NPV"] = float(npv)
        print(f"{artery}=1: AUC={auc}, sensitivity={sens}, specificity={spec}, PPV={ppv}, NPV={npv}")

###############################################################################
#                                 2. Get data                                 #
###############################################################################

# prev_model = joblib.load('models/random_forest_final_model_f1_new_overlap_common.joblib')
# features = prev_model['features']
# features = prev_model['cols']
# scaler = prev_model['scaler']
# cols_to_scale = prev_model['cols']

with open("data/cleaned/selected_features_f1_final_new_common_nate.txt", "r") as f:
    features = f.read().splitlines()
features = features + ["age", "sex"]

donot_include_columns = ["patient_id"]
arteries = ["left_circumflex", "right_coronary_artery", "left_anterior_descending"]

# train data
train_data = pd.read_csv('data/data_ood_id/train_data.csv')
train_data = train_data.drop(columns=['filename'])

scaler = StandardScaler()
# cols_to_scale = [c for c in train_data.columns if c not in donot_include_columns and c not in arteries and c != 'occuluded_artery']
cols_to_scale = features
X_train_scaled = scaler.fit_transform(train_data[cols_to_scale])

# X_train_scaled = scaler.transform(train_data[cols_to_scale])
X_train_scaled = pd.DataFrame(X_train_scaled, columns=cols_to_scale, index=train_data.index)
X_train_scaled.columns = [
    "_".join(map(str, col)) if isinstance(col, tuple) else col
    for col in X_train_scaled.columns
]
X_train_scaled = pd.concat([X_train_scaled, train_data[list(donot_include_columns) + arteries]], axis=1)
y_train = train_data["occuluded_artery"]


###############################################################################
#                        3. Create CV splits                         #
###############################################################################

groups = X_train_scaled['patient_id']
X_model = X_train_scaled.drop(columns=['patient_id'])
y_model = y_train
cv = StratifiedGroupKFold(n_splits=10, random_state=SEED, shuffle=True)
cv_splits = list(cv.split(X_model, y_model, groups=groups))


###############################################################################
#                               4. Classifiers                                #
###############################################################################
# Create list of tuples with classifier label and classifier object
classifiers = {}
classifiers.update({"LDA": LinearDiscriminantAnalysis()})
classifiers.update({"QDA": QuadraticDiscriminantAnalysis()})
classifiers.update({"AdaBoost": AdaBoostClassifier(random_state=SEED)})
classifiers.update({"Bagging": BaggingClassifier(random_state=SEED)})
classifiers.update({"Extra Trees Ensemble": ExtraTreesClassifier(random_state=SEED)})
classifiers.update({"Gradient Boosting": GradientBoostingClassifier(random_state=SEED)})
classifiers.update({"Random Forest": RandomForestClassifier(random_state=SEED)})
########## classifiers.update({"Ridge": RidgeClassifier()})
########## classifiers.update({"SGD": SGDClassifier(random_state=SEED)})
classifiers.update({"BNB": BernoulliNB()})
classifiers.update({"GNB": GaussianNB()})
classifiers.update({"KNN": KNeighborsClassifier()})
classifiers.update({"MLP": MLPClassifier(random_state=SEED)})
######## classifiers.update({"LSVC": LinearSVC(random_state=SEED)})
classifiers.update({"NuSVC": NuSVC(random_state=SEED, probability=True)})
######## classifiers.update({"SVC": SVC(random_state=SEED, probability=True)})
classifiers.update({"DTC": DecisionTreeClassifier(random_state=SEED)})
classifiers.update({"ETC": ExtraTreeClassifier(random_state=SEED)})


###############################################################################
#                             5. Hyper-parameters                             #
###############################################################################
# Initiate parameter grid
parameters = {}

# Update dict with LDA
parameters.update({"LDA": {"solver": ["svd"], 
                                         }})

# Update dict with QDA
parameters.update({"QDA": {"reg_param":[0.01*ii for ii in range(0, 101)], 
                                         }})
# Update dict with AdaBoost
parameters.update({"AdaBoost": { 
                                "estimator": [DecisionTreeClassifier(max_depth = ii) for ii in range(1,6)],
                                "n_estimators": [200],
                                "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 1.0]
                                 }})

# Update dict with Bagging
parameters.update({"Bagging": { 
                                "estimator": [DecisionTreeClassifier(max_depth = ii) for ii in range(1,6)],
                                "n_estimators": [200],
                                "max_features": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                "n_jobs": [n_jobs]
                                }})

# Update dict with Gradient Boosting
parameters.update({"Gradient Boosting": { 
                                        "learning_rate":[0.15,0.1,0.05,0.01,0.005,0.001], 
                                        "n_estimators": [200],
                                        "max_depth": [2,3,4,5,6],
                                        "min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                        "min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                        "max_features": ["sqrt", "log2"],
                                        "subsample": [0.8, 0.9, 1]
                                         }})


# Update dict with Extra Trees
parameters.update({"Extra Trees Ensemble": { 
                                            "n_estimators": [200],
                                            "class_weight": [None, "balanced"],
                                            "max_features": ["sqrt", "log2"],
                                            "max_depth" : [3, 4, 5, 6, 7, 8],
                                            "min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                            "min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                            "criterion" :["gini", "entropy"]     ,
                                            "n_jobs": [n_jobs]
                                             }})


# Update dict with Random Forest Parameters
parameters.update({"Random Forest": { 
                                    "n_estimators": [200],
                                    "class_weight": [None, "balanced"],
                                    "max_features": ["sqrt", "log2"],
                                    "max_depth" : [3, 4, 5, 6, 7, 8],
                                    "min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                    "min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                    "criterion" :["gini", "entropy"]     ,
                                    "n_jobs": [n_jobs]
                                     }})

# Update dict with Ridge
parameters.update({"Ridge": { 
                            "alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]
                             }})

# Update dict with SGD Classifier
parameters.update({"SGD": { 
                            "alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0],
                            "penalty": ["l1", "l2"],
                            "n_jobs": [n_jobs]
                             }})


# Update dict with BernoulliNB Classifier
parameters.update({"BNB": { 
                            "alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]
                             }})

# Update dict with GaussianNB Classifier
parameters.update({"GNB": { 
                            "var_smoothing": [1e-9, 1e-8,1e-7, 1e-6, 1e-5]
                             }})

# Update dict with K Nearest Neighbors Classifier
parameters.update({"KNN": { 
                            # "n_neighbors": list(range(1,31)),
                            # "p": [1, 2, 3, 4, 5],
                            # "leaf_size": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                            "n_neighbors": [1],
                            "p": [1],
                            "leaf_size": [5],
                            "n_jobs": [n_jobs]
                             }})

# Update dict with MLPClassifier
parameters.update({"MLP": { 
                            "hidden_layer_sizes": [(5), (10), (5,5), (10,10), (5,5,5), (10,10,10)],
                            "activation": ["identity", "logistic", "tanh", "relu"],
                            "learning_rate": ["constant", "invscaling", "adaptive"],
                            "max_iter": [500, 1000, 2000],
                            "alpha": list(10.0 ** -np.arange(1, 10)),
                             }})

parameters.update({"LSVC": { 
                            "penalty": ["l2"],
                            "C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100]
                             }})

parameters.update({"NuSVC": { 
                            "nu": [0.25, 0.50, 0.75],
                            "kernel": ["linear", "rbf", "poly"],
                            "degree": [1,2,3,4,5,6],
                             }})

parameters.update({"SVC": { 
                            "kernel": ["linear", "rbf", "poly"],
                            "gamma": ["auto"],
                            "C": [0.1, 0.5, 1, 5, 10, 50, 100],
                            "degree": [1, 2, 3, 4, 5, 6]
                             }})


# Update dict with Decision Tree Classifier
parameters.update({"DTC": { 
                            "criterion" :["gini", "entropy"],
                            "splitter": ["best", "random"],
                            "class_weight": [None, "balanced"],
                            "max_features": ["sqrt", "log2"],
                            "max_depth" : [1,2,3, 4, 5, 6, 7, 8],
                            "min_samples_split": [0.005, 0.01, 0.05, 0.10],
                            "min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                             }})

# Update dict with Extra Tree Classifier
parameters.update({"ETC": { 
                            "criterion" :["gini", "entropy"],
                            "splitter": ["best", "random"],
                            "class_weight": [None, "balanced"],
                            "max_features": ["sqrt", "log2"],
                            "max_depth" : [1,2,3, 4, 5, 6, 7, 8],
                            "min_samples_split": [0.005, 0.01, 0.05, 0.10],
                            "min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                             }})


###############################################################################
#                     7. Tuning a classifier as final stage model             #
###############################################################################
# Define classifier to use as the base of the recursive feature elimination algorithm

# removed for speed, with real runs add lopo back for all models to be run
# for name, classifier in classifiers.items():

first_model = next(iter(classifiers.items()))
name, classifier = first_model
results = create_results_structure()
print(f"Tuning Classifier {name} — go grab a beer or something")

# Define parameter grid
param_grid = parameters[name]

# Initialize GridSearch object
gscv = GridSearchCV(
    classifier,
    param_grid,
    cv=cv_splits,
    n_jobs=n_jobs,
    verbose=1,
    scoring=scoring,
    error_score=0.0
)

# Compute sample weights to handle class imbalance
sample_weights = compute_sample_weight("balanced", y_model)

# Try fitting with sample_weight if supported
fit_signature = inspect.signature(classifier.fit)
if "sample_weight" in fit_signature.parameters:
    print(f" → {name} supports sample_weight. Using weighted fit.")
    gscv.fit(X_model[features], y_model, sample_weight=sample_weights)
else:
    print(f"{name} does NOT support sample_weight. Using unweighted fit.")
    gscv.fit(X_model[features], y_model)

# Get best parameters and score
best_params = gscv.best_params_
best_score = gscv.best_score_
best_classifier = gscv.best_estimator_
    
# Update classifier parameters
# tuned_params = {item: best_params[item] for item in best_params}
# classifier.set_params(**tuned_params)

#evaluating on train data
y_pred_train = best_classifier.predict_proba(X_train_scaled[features])[:, 1]
print_overall_results(results, y_train, y_pred_train, group="train", threshold=0.5)
print("")
print_results_by_artery(results, X_train_scaled, best_classifier, arteries, features, group="train", threshold=0.5)


#test set 1
test_set_1 = pd.read_csv('data/data_ood_id/test_set_ood_1.csv')
X_test_1 = test_set_1.drop(columns=["occuluded_artery"])
y_test_1 = test_set_1["occuluded_artery"]
X_test_1_scaled = scaler.transform(X_test_1[cols_to_scale])
X_test_1_scaled = pd.DataFrame(X_test_1_scaled, columns=cols_to_scale, index=X_test_1.index)
X_test_1_scaled.columns = [
    "_".join(map(str, col)) if isinstance(col, tuple) else col
    for col in X_test_1_scaled.columns
]
X_test_1_scaled = pd.concat([X_test_1_scaled, X_test_1[list(donot_include_columns) + arteries]], axis=1)
y_pred_test_1 = best_classifier.predict_proba(X_test_1_scaled[features])[:, 1]
print_overall_results(results, y_test_1, y_pred_test_1, group="test_1", threshold=0.5)
print("")
print_results_by_artery(results, X_test_1_scaled, best_classifier, arteries, features, group="test_1", threshold=0.5)

#test set 2
test_set_2 = pd.read_csv('data/data_ood_id/test_set_ood_2.csv')
X_test_2 = test_set_2.drop(columns=["occuluded_artery"])
y_test_2 = test_set_2["occuluded_artery"]
X_test_2_scaled = scaler.transform(X_test_2[cols_to_scale])
X_test_2_scaled = pd.DataFrame(X_test_2_scaled, columns=cols_to_scale, index=X_test_2.index)
X_test_2_scaled.columns = [
    "_".join(map(str, col)) if isinstance(col, tuple) else col
    for col in X_test_2_scaled.columns
]
X_test_2_scaled = pd.concat([X_test_2_scaled, X_test_2[list(donot_include_columns) + arteries]], axis=1)
y_pred_test_2 = best_classifier.predict_proba(X_test_2_scaled[features])[:, 1]
print_overall_results(results, y_test_2, y_pred_test_2, group="test_2", threshold=0.5)
print("")
print_results_by_artery(results, X_test_2_scaled, best_classifier, arteries, features, group="test_2", threshold=0.5)


#test set 3
test_set_3 = pd.read_csv('data/data_ood_id/test_set_id.csv')
X_test_3 = test_set_3.drop(columns=["occuluded_artery"])
y_test_3 = test_set_3["occuluded_artery"]
X_test_3_scaled = scaler.transform(X_test_3[cols_to_scale])
X_test_3_scaled = pd.DataFrame(X_test_3_scaled, columns=cols_to_scale, index=X_test_3.index)
X_test_3_scaled.columns = [
    "_".join(map(str, col)) if isinstance(col, tuple) else col
    for col in X_test_3_scaled.columns
]
X_test_3_scaled = pd.concat([X_test_3_scaled, X_test_3[list(donot_include_columns) + arteries]], axis=1)
y_pred_test_3 = best_classifier.predict_proba(X_test_3_scaled[features])[:, 1]
print_overall_results(results, y_test_3, y_pred_test_3, group="test_3", threshold=0.5)
print("")
print_results_by_artery(results, X_test_3_scaled, best_classifier, arteries, features, group="test_3", threshold=0.5)

results_dir = os.path.join(RESULTS_BASE_DIR, name)
models_dir = os.path.join(MODELS_BASE_DIR, name)
results["best_params"] = best_params
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
with open(os.path.join(results_dir, "final_model_f1_new_overlap.json"), 'w') as f:
    json.dump(results, f, indent=4, default=str)

joblib.dump({
    'model': best_classifier,
    'features': features,
    'scaler': scaler,
    'cols': cols_to_scale,
    'best_params': gscv.best_params_,
}, os.path.join(models_dir, "final_model_f1_new_overlap.joblib"))

print(f"Done for {name}......")
