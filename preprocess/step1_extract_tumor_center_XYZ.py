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

# This script extracts coordinates of tumor center from segmentation mask.

import numpy as np
import nibabel as nib
import os
from scipy.ndimage.measurements import center_of_mass, label
from tqdm import tqdm

def get_tumor_center(seg, modality='P'):
    if modality in ['P']:  # mannual draw center with label > 0
        M, N, K = np.array(center_of_mass(seg > 0)).astype(int)
    elif modality == 'AI':  # AI segmentation with label 2
        label_mask, nf = label(seg == 2)  # check tumor
        area_nf = [np.sum(label_mask == k) for k in range(1, nf + 1)]
        ### attention, need to sort by size and pick the largest
        seg1 = (label_mask == np.flip(np.argsort(area_nf))[0] + 1)
        M, N, K = np.array(center_of_mass(seg1 > 0)).astype(int)
    return M, N, K


source_folder = './demo/img'
dest_folder = './demo/img'

# load patient list with images
pid_list = os.listdir(source_folder)

for k in tqdm(range(len(pid_list))):
    pid = pid_list[k]
    modality = ''
    try:
        dict_path = os.path.join(dest_folder, pid, 'TC_coordinates.npy')

        if os.path.exists(os.path.join(source_folder, pid, 'A-label-man.nii.gz')):
            seg_path = os.path.join(source_folder, pid, 'A-label-man.nii.gz')
            modality = 'P'  # mannually added
        elif os.path.exists(os.path.join(source_folder, pid, 'A-label-AI.nii.gz')):
            seg_path = os.path.join(source_folder, pid, 'A-label-AI.nii.gz')
            modality = 'AI'  # nnUNet
        else:
            print('{:s} no mask!\n'.format(pid))
            continue

        seg_nib = nib.load(seg_path)
        affine = seg_nib.affine

        seg = seg_nib.get_fdata()
        M, N, K = get_tumor_center(seg, modality=modality)
        MNK1 = np.array([M, N, K, 1])
        XYZ1 = affine.dot(MNK1)

        dict_res = {'MNK': MNK1, 'XYZ1': XYZ1, 'Phase': modality}

        # save XYZ coordinates to patient nii folder
        np.save(dict_path, dict_res)

    except:
        print('Error:{}!'.format(pid))
