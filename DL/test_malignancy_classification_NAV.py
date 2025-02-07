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

import sys

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/media/bucket/Project/renal-mass-ai'])

import pandas as pd
import torch
from tqdm import tqdm
import os
import numpy as np
# from DL.networks import HybridNAV
from networks import HybridNAV
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import normal_intensity_CT

# build model
model = HybridNAV(input_channel=1)
device = torch.device("cuda:{:d}".format(0) if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load('DL/models/best_M01-HybridNAV_epoch_100_test.pt'))
model.eval()

# set test cohort here
cohort_name = 'internal'
# cohort_name = 'external'
# cohort_name = 'prospective'

df = pd.read_csv('csv/demo_split.csv', dtype={'pid': str})
df = df[df['split'] == cohort_name]

## inference
pid_list = df['pid'].to_list()
npy_folder = './demo/npy'

pred_score_list = []

for pid in tqdm(pid_list):
    try:
        npy_name = os.path.join(npy_folder, '{}_AXN.npy'.format(pid))
        imgN = np.load(npy_name).squeeze()[np.newaxis, ::]
        npy_name = os.path.join(npy_folder, '{}_AXA.npy'.format(pid))
        imgA = np.load(npy_name).squeeze()[np.newaxis, ::]
        npy_name = os.path.join(npy_folder, '{}_AXV.npy'.format(pid))
        imgV = np.load(npy_name).squeeze()[np.newaxis, ::]

        imgN = normal_intensity_CT(imgN)[np.newaxis, ::]
        imgA = normal_intensity_CT(imgA)[np.newaxis, ::]
        imgV = normal_intensity_CT(imgV)[np.newaxis, ::]

        N_tensor = torch.tensor(imgN, dtype=torch.float32).to(device)
        A_tensor = torch.tensor(imgA, dtype=torch.float32).to(device)
        V_tensor = torch.tensor(imgV, dtype=torch.float32).to(device)

        pred = model(N_tensor, A_tensor, V_tensor).detach().cpu().numpy()[0][1]

        pred_score_list.append(pred)
    except:
        pred_score_list.append(-1)

# construct dataframe and save prediction results
df_pred = pd.DataFrame({
    'pid': df['pid'],
    'DL_malignant': pred,
    'GT': df['class']
})

df_pred.to_excel('DL/res-01.xlsx')
