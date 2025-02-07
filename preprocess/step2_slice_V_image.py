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

import numpy as np
import pandas as pd
import nibabel as nib
import os
from tqdm import tqdm
from skimage.transform import resize
from utils import crop_3Dimage

# set the folder of *.nii.gz images (N.nii.gz, A.nii.gz, V.nii.gz)
nii_folder = './demo/img'
dest_folder = './demo/npy'
df = pd.read_csv('./csv/demo_split.csv', dtype={'pid': str})
pid_list = df['pid'].to_list()

# set phase
X = 'V'
# load venous phase images and tumor center, cropped to npy
error_list = []
for k in tqdm(range(0, len(pid_list))):
    try:
        pid = pid_list[k]
        TC_xyz_path = os.path.join(nii_folder, pid, 'TC_coordinates.npy')
        if not (os.path.exists(TC_xyz_path)):
            print('TC not found Error:{}!'.format(pid))
            continue

        tmp = np.load(TC_xyz_path, allow_pickle=True).item()
        xyz1 = tmp['XYZ1']

        X_path = os.path.join(nii_folder, pid, '{}.nii.gz'.format(X))
        if not (os.path.exists(X_path)):
            print('{} not found Error:{}!'.format(X, pid))
            continue

        X_nib = nib.load(X_path)

        # 140 x 140 x 160 mm
        # 224 x 224 x 32 px
        img_X = X_nib.get_fdata()

        raw_ds = X_nib.header['pixdim']
        affine = X_nib.affine
        raw_dm, raw_dn, raw_dz = raw_ds[1], raw_ds[2], raw_ds[3]
        crop_win = int(120 / raw_dm), int(120 / raw_dn), 0
        M_c, N_c, K_c, _ = np.linalg.inv(affine).dot(xyz1)
        M_c, N_c, K_c = int(M_c), int(N_c), int(K_c)
        crop_X = crop_3Dimage(img_X, (M_c, N_c, K_c), crop_win)
        crop_X = resize(crop_X, [224, 224, 1], order=1)
        crop_res = np.array([crop_X])
        np.save(arr=crop_res, file=os.path.join(dest_folder, '{}_AX{}.npy'.format(pid, X)))

    except:
        print('Error:{}!'.format(pid))

# ### visualize
# tmp = np.load('demo/npy/P00001_AXA.npy')
# fig = title_crop(tmp[0])
# fig.show()
