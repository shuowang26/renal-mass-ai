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

# This script extracts radiomics features for the training, validation, test sets.

import os
import pickle
import numpy as np
import pandas as pd
from radiomics import featureextractor
from tqdm import tqdm
import SimpleITK as sitk

def rm_redundant(result):
    """ remove redundant/useless features """
    filtered_result = result.copy()
    for k, v in result.items():
        if k[:8] == 'diagnost':
            del filtered_result[k]
    return filtered_result



# load list of all feature names
with open("radiomics/all_feat_names.pickle", 'rb') as f:
    all_feats_name = pickle.load(f)

# load data split csv, prepare list of npy for feature extraction
csv_path='csv/demo_split.csv'
df = pd.read_csv(csv_path)  # class 0: benign, 1: indolent, 2: invasive
ind = df['path'].apply(
    lambda x: os.path.exists(x + '_AXN.npy') & os.path.exists(x + '_AXA.npy') & os.path.exists(x + '_AXV.npy') & os.path.exists(x + '_AXM.npy'))
npy_list, class_label_list = df['path'][ind].to_list(), df['class'][ind].to_list()

# extract radiomics features using PyRadiomics
extractor_params = "radiomics/radiomics_feat.yaml"
extractor = featureextractor.RadiomicsFeatureExtractor(extractor_params)
print('Feature Extractor:')
print('Extraction parameters:\n\t', extractor.settings)
print('Enabled filters:\n\t', extractor.enabledImagetypes)
print('Enabled features:\n\t', extractor.enabledFeatures)

all_feat_list = []
for k in tqdm(range(len(npy_list))):
    npy_name = npy_list[k]
    pid = npy_name.split('/')[-1]
    roi_path = '{:}_AXM.npy'.format(npy_name)
    roi_image = sitk.GetImageFromArray(np.load(roi_path).squeeze())
    p_feat_df = []
    for s in ['N', 'A', 'V']:
        img_path = '{:}_AX{}.npy'.format(npy_name, s)
        npy_image = np.load(img_path).squeeze()
        itk_image = sitk.GetImageFromArray(npy_image)
        try:
            p_s_feat = rm_redundant(extractor.execute(itk_image, roi_image))
        except:
            p_s_feat = {k: np.nan for k in all_feats_name}
        p_s_feat_df = pd.DataFrame(data=p_s_feat, index=[k]).add_prefix(f"{s}_")
        p_feat_df.append(p_s_feat_df)
    all_feat_list.append(p_feat_df)

# save the feature table
all_feat_df = pd.concat(all_feat_list)

# fill na with mean, can also be done with each subset
for f in all_feat_df.columns:
    all_feat_df[f].fillna(all_feat_df[f].mean(), inplace=True)

# save feature table
all_feat_df.to_csv("radiomics/radiomics_feat.csv")
