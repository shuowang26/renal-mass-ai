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

# This script loads radiomics model for invasiveness prediction.

import os
import pandas as pd
import numpy as np
import joblib
scaler = joblib.load('radiomics/trained_models/model12_scaler.pkl')
cv_results = joblib.load('radiomics/trained_models/model12_cv_results.pkl')
selected_features = joblib.load('radiomics/trained_models/model12_selected_features.pkl')
selected_index = joblib.load('radiomics/trained_models/model12_selected_index.pkl')

csv_path = 'radiomics/demo_split.csv'
df = pd.read_csv(csv_path)  # class 0, 1, 2

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

# change the name of test cohort
# cohort_name = 'train'
# cohort_name = 'val'
cohort_name = 'internal'
# cohort_name = 'external'
# cohort_name = 'prospective'

train_feat = radiomics_feat[df['split'] == 'train']
test_ind = df['split'] == cohort_name
test_feat = radiomics_feat[test_ind]
test_pid_list = radiomics_feat[test_ind].iloc[:, 0].to_list()
pid_list = df['path'].str.split('/').str[-1].to_list()

X_train = train_feat.values[:, 2:].astype(np.float32)
y_train = train_feat.values[:, 1].astype(int)
y_train = np.array(y_train == 2).astype(int)  # binary classification label for invasiveness

X_test = test_feat.values[:, 2:].astype(np.float32)
y_test = test_feat.values[:, 1].astype(int)
y_test = np.array(y_test == 2).astype(int)  # binary classification label for invasiveness

print('Train: %s' % len(y_train))
print('Test: %s'  % len(y_test))

X_train_filtered = scaler.transform(X_train)[:, selected_index]
X_test_filtered = scaler.transform(X_test)[:, selected_index]

test_predict_prob = [estimator.predict_proba(X_test_filtered) for estimator in cv_results['estimator']]
test_proba = np.stack(test_predict_prob, axis=2).mean(axis=2)

pred = test_proba[:, 1]

# construct dataframe and save prediction results
indices = test_ind[test_ind].index.tolist()
df_pred = pd.DataFrame({
    'pid': [pid_list[i] for i in indices],  # Correctly index pid_list
    'radiomics_invasive': pred,
    'GT': df['class'][test_ind]
})

df_pred.to_excel('radiomics/res-12.xlsx')
