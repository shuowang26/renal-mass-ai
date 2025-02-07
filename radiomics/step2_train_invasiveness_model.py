# Copyright 2024, Shuo Wang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# This script trains radiomics model for invasiveness prediction.

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score, roc_curve
import joblib

# load data split csv, get the gt labels
csv_path='csv/demo_split.csv'
df = pd.read_csv(csv_path)  # class 0: benign, 1: indolent, 2: invasive

# prepare data table
radiomics_feat = pd.read_csv("radiomics/radiomics_feat.csv", index_col=0)
radiomics_feat.insert(0, "class", df['class'])
radiomics_feat.insert(0, "pid", df['path'].str.split('/').str[-1].to_list())

rm_shape_features = ['A_original_shape2D_Elongation',
                     'A_original_shape2D_MajorAxisLength',
                     'A_original_shape2D_MaximumDiameter',
                     'A_original_shape2D_MeshSurface',
                     'A_original_shape2D_MinorAxisLength',
                     'A_original_shape2D_Perimeter',
                     'A_original_shape2D_PerimeterSurfaceRatio',
                     'A_original_shape2D_PixelSurface',
                     'A_original_shape2D_Sphericity',
                     'V_original_shape2D_Elongation',
                     'V_original_shape2D_MajorAxisLength',
                     'V_original_shape2D_MaximumDiameter',
                     'V_original_shape2D_MeshSurface',
                     'V_original_shape2D_MinorAxisLength',
                     'V_original_shape2D_Perimeter',
                     'V_original_shape2D_PerimeterSurfaceRatio',
                     'V_original_shape2D_PixelSurface',
                     'V_original_shape2D_Sphericity',
                     ]
radiomics_feat = radiomics_feat.drop(rm_shape_features, axis=1)  # drop repeated shape features

# get training set
train_feat = radiomics_feat[df['split'] == 'train']
X_train = train_feat.values[:, 2:].astype(np.float32)
scaler = StandardScaler().fit(X_train)
y_train = train_feat.values[:, 1].astype(int)
y_train = np.array(y_train == 2).astype(int)  # binary classification label for invasiveness

# get validation set
val_feat = radiomics_feat[df['split'] == 'val']
X_val = val_feat.values[:, 2:].astype(np.float32)
y_val = val_feat.values[:, 1].astype(int)
y_val = np.array(y_val == 2).astype(int)

print('Train: %s' % len(y_train))
print('Test: %s' % len(y_val))

# perform SMOTE to balance training set
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
print('Resampled train: %s' % len(y_train))

### feature selection with LASSO
# find best alpha for lasso
lasso = Lasso()
parameters = {'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]}
lasso_grid = GridSearchCV(lasso, parameters, cv=5)
lasso_grid.fit(scaler.transform(X_train), y_train)
optimal_alpha = lasso_grid.best_params_['alpha']
lasso_ = Lasso(alpha=optimal_alpha, max_iter=1000).fit(scaler.transform(X_train), y_train)

# show 20 largest lasso coeff
features_names = radiomics_feat.columns.values[2:]
lasso_df = pd.DataFrame({
    "name": features_names,
    "coeff": list(np.abs(lasso_.coef_) * 100),
})
lasso_df.sort_values("coeff", inplace=True, ascending=False)
print(lasso_df.iloc[:20, :])

### search best Top-K features for classification
for K in np.arange(1, 20):
    print(K)
    features_names = radiomics_feat.columns.values[2:]
    lasso_df = pd.DataFrame({"name": features_names, "coeff": list(np.abs(lasso_.coef_) * 100)})
    lasso_df.sort_values("coeff", inplace=True, ascending=False)

    selected_index = list(lasso_df.iloc[:K, :].index)
    selected_features = list(lasso_df.iloc[:K, 0])
    X_train_filtered = scaler.transform(X_train)[:, selected_index]

    X_test_filtered = scaler.transform(X_val)[:, selected_index]

    print(selected_features)

    fpr_list, tpr_list, auc_list, f1score_list, acc_list, sens_list, spec_list, proba_list, gt_list = [], [], [], [], [], [], [], [], []

    # three repeated runs
    for seed in [1, 2, 3]:
        cv_results = cross_validate(
            RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed),
            X_train_filtered, y_train,
            scoring=['accuracy', 'roc_auc'],
            return_estimator=True,
            cv=5)

        val_predict_prob = [estimator.predict_proba(X_test_filtered) for estimator in cv_results['estimator']]
        test_proba = np.stack(val_predict_prob, axis=2).mean(axis=2)
        proba_list.append(test_proba[:, 1])
        gt_list.append(y_val)

        auc = roc_auc_score(y_val, test_proba[:, 1])
        fpr, tpr, thresh_auc = roc_curve(y_val, test_proba[:, 1])
        fpr_list.append(fpr)
        tpr_list.append(tpr)

        # best threshold - youden
        youden = tpr - fpr
        index = np.argmax(youden)
        thresh_opt = round(thresh_auc[index], ndigits=4)

        # threshold-related metrics
        test_pred = deepcopy(test_proba[:, 1])
        test_pred = np.array(test_pred > thresh_opt, dtype=int)
        acc = accuracy_score(y_val, test_pred)
        confusion = confusion_matrix(y_val, test_pred)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        sens = TP / float(TP + FN)  # recall
        spec = TN / float(TN + FP)
        prec = TP / float(TP + FP)
        f1 = f1_score(y_val, test_pred)
        auc_list.append(auc)
        f1score_list.append(f1)
        acc_list.append(acc)
        sens_list.append(sens)
        spec_list.append(spec)

    auc_list = np.array(auc_list)
    f1score_list = np.array(f1score_list)
    acc_list = np.array(acc_list)
    sens_list = np.array(sens_list)
    spec_list = np.array(spec_list)

    print(f"======================")
    print(f"K: ", K)
    print(f"AUC: ", auc_list)
    print(f"auc:\t {auc_list.mean():.3f} ± {auc_list.std():.3f}")
    print(f"acc:\t {acc_list.mean():.3f} ± {acc_list.std():.3f}")
    print(f"sens:\t {sens_list.mean():.3f} ± {sens_list.std():.3f}")
    print(f"spec:\t {spec_list.mean():.3f} ± {spec_list.std():.3f}")
    print(f"f1:\t {f1score_list.mean():.3f} ± {f1score_list.std():.3f}")
    print(f"======================")

### build model with best K
K = 14
features_names = radiomics_feat.columns.values[2:]
lasso_df = pd.DataFrame({"name": features_names, "coeff": list(np.abs(lasso_.coef_) * 100)})
lasso_df.sort_values("coeff", inplace=True, ascending=False)
selected_index = list(lasso_df.iloc[:K, :].index)
selected_features = list(lasso_df.iloc[:K, 0])
X_train_filtered = scaler.transform(X_train)[:, selected_index]
X_test_filtered = scaler.transform(X_val)[:, selected_index]

cv_results = cross_validate(
    RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=1),
    X_train_filtered, y_train,
    scoring=['accuracy', 'roc_auc'],
    return_estimator=True,
    cv=5)

# save models and related features
joblib.dump(selected_features, 'radiomics/trained_models/model12_selected_features.pkl')
joblib.dump(scaler, 'radiomics/trained_models/model12_scaler.pkl')
joblib.dump(cv_results, 'radiomics/trained_models/model12_cv_results.pkl')
joblib.dump(selected_index, 'radiomics/trained_models/model12_selected_index.pkl')
