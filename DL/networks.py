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

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet121, EfficientNetBN, ViT


class HybridNAV(nn.Module):
    def __init__(self, input_channel=1, f_dim=512, dropout_prob=0.5):
        super(HybridNAV, self).__init__()

        self.modelN = torchvision.models.resnet18(pretrained=True)
        self.modelN.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.modelN.fc = nn.Identity()  # only keep backbone

        self.modelA = torchvision.models.resnet18(pretrained=True)
        self.modelA.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.modelA.fc = nn.Identity()  # only keep backbone

        self.modelV = torchvision.models.resnet18(pretrained=True)
        self.modelV.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.modelV.fc = nn.Identity()  # only keep backbone

        self.fc_all = torch.nn.Sequential(
            nn.Linear(f_dim * 3, f_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(f_dim, f_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(f_dim, f_dim),
            nn.Linear(f_dim, 2)
        )

    def forward(self, xN, xA, xV):
        fN = self.modelN(xN)
        fA = self.modelA(xA)
        fV = self.modelV(xV)
        concatenated_features = torch.cat([fN, fA, fV], dim=1)
        y = self.fc_all(concatenated_features)
        y = F.softmax(y, dim=1)
        return y


class HybridXX(nn.Module):
    def __init__(self, input_channel=1, f_dim=512, dropout_prob=0.5):
        super(HybridXX, self).__init__()

        self.model1 = torchvision.models.resnet18(pretrained=True)
        self.model1.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model1.fc = nn.Identity()  # only keep backbone

        self.model2 = torchvision.models.resnet18(pretrained=True)
        self.model2.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model2.fc = nn.Identity()  # only keep backbone

        self.fc_all = torch.nn.Sequential(
            nn.Linear(f_dim * 2, f_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(f_dim, f_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(f_dim, f_dim),
            nn.Linear(f_dim, 2)
        )

    def forward(self, x1, x2):
        f1 = self.model1(x1)
        f2 = self.model2(x2)
        concatenated_features = torch.cat([f1, f2], dim=1)
        y = self.fc_all(concatenated_features)
        y = F.softmax(y, dim=1)
        return y


class Res18(nn.Module):
    def __init__(self, input_channel=16, out_channel=2):
        super(Res18, self).__init__()

        model = torchvision.models.resnet18(pretrained=True)

        model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, out_channel)

        self.Res2D = model

    def forward(self, x):
        y = F.softmax(self.Res2D(x))
        return y


class Res50(nn.Module):
    def __init__(self, input_channel=16, out_channel=2):
        super(Res50, self).__init__()

        model = torchvision.models.resnet50(pretrained=True)

        model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, out_channel)

        self.Res2D = model

    def forward(self, x):
        y = F.softmax(self.Res2D(x))
        return y


class Res101(nn.Module):
    def __init__(self, input_channel=1, out_channel=2):
        super(Res101, self).__init__()

        model = torchvision.models.resnet101(pretrained=True)

        model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, out_channel)

        self.Res2D = model

    def forward(self, x):
        y = F.softmax(self.Res2D(x))
        return y


class ViTClassifier(nn.Module):
    def __init__(self, input_channel=3, out_channel=2, img_size=224):
        super(ViTClassifier, self).__init__()

        # Load ViT from MONAI
        self.model = ViT(
            in_channels=input_channel,
            img_size=(img_size, img_size),  # Ensure this is a 2D tuple
            patch_size=(16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            pos_embed="conv",
            classification=True,
            num_classes=out_channel,
            spatial_dims=2
        )

    def forward(self, x):
        y = F.softmax(self.model(x)[0], dim=1)
        return y


class DenseNet(nn.Module):
    def __init__(self, input_channel=16, out_channel=2):
        super(DenseNet, self).__init__()

        # Load DenseNet121 from MONAI
        self.model = DenseNet121(spatial_dims=2, in_channels=input_channel, out_channels=out_channel, pretrained=True)

    def forward(self, x):
        y = F.softmax(self.model(x), dim=1)
        return y


class EfficientNet(nn.Module):
    def __init__(self, input_channel=16, out_channel=2, model_name='efficientnet-b0'):
        super(EfficientNet, self).__init__()

        # Load EfficientNet from MONAI
        self.model = EfficientNetBN(model_name="efficientnet-b0", spatial_dims=2, in_channels=input_channel,
                                    num_classes=out_channel, pretrained=True)

    def forward(self, x):
        y = F.softmax(self.model(x), dim=1)
        return y
