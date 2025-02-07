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
import matplotlib.pyplot as plt

def normal_intensity_CT(img):
    a_min, a_max = -500, 500
    img = np.clip(img, a_min=a_min, a_max=a_max)
    img = (img - a_min)/(a_max-a_min)
    return img


def crop_3Dimage(image, center, size, affine_matrix=None):
    """ Crop a 3D image using a bounding box centred at (c0, c1, c2) with specified size (size0, size1, size2) """
    c0, c1, c2 = center
    size0, size1, size2 = size

    S0, S1, S2 = image.shape

    r0, r1, r2 = int(size0 / 2), int(size1 / 2), int(size2 / 2)
    start0, end0 = c0 - r0, c0 + r0
    start1, end1 = c1 - r1, c1 + r1
    start2, end2 = c2 - r2, c2 + r2 + 1

    start0_, end0_ = max(start0, 0), min(end0, S0)
    start1_, end1_ = max(start1, 0), min(end1, S1)
    start2_, end2_ = max(start2, 0), min(end2, S2)

    # Crop the image
    crop = image[start0_: end0_, start1_: end1_, start2_: end2_]
    crop = np.pad(crop,
                  ((start0_ - start0, end0 - end0_), (start1_ - start1, end1 - end1_), (start2_ - start2, end2 - end2_)),
                  'constant')

    if affine_matrix is None:
        return crop
    else:
        R, b = affine_matrix[0:3, 0:3], affine_matrix[0:3, -1]
        affine_matrix[0:3, -1] = R.dot(np.array([c0-r0, c1-r1, c2-r2])) + b
        return crop, affine_matrix

def title_crop(img):
    plt.style.use('dark_background')
    fig, axs = plt.subplots(nrows=5, ncols=int(np.ceil(img.shape[2]/5)), frameon=False,
                            gridspec_kw={'wspace':0.05, 'hspace':0.05})
    fig.set_size_inches([12, 12])
    [ax.set_axis_off() for ax in axs.ravel()]
    axs = axs.ravel()

    for k in range(0, img.shape[2]):
        axs[k].imshow(np.flipud(img[:, :, k].transpose()), cmap='gray', vmin=-100, vmax=400)
    return fig