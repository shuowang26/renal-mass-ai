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

import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import copy
import argparse
import numpy as np
from sklearn import metrics
# from DL.networks import HybridNAV
# from DL.dataset import HybridSlice
from networks import HybridNAV
from dataset import HybridSlice
from torch.utils.data.sampler import WeightedRandomSampler


def PredLoss(pred, gt):
    gt = F.one_hot(gt, num_classes=2).to(torch.float32)
    Loss_BCE = F.binary_cross_entropy(pred, gt, reduction='mean')
    return Loss_BCE


def train(epoch):
    model.train()
    train_loss = 0

    all_gt, all_pred = [], []

    for batch_idx, (imgN, imgA, imgV, gt) in enumerate(train_dataloader):

        if imgN.shape[0] == 1:
            continue

        # train model
        optimizer.zero_grad()
        imgN = imgN.to(device, dtype=torch.float32)
        imgA = imgA.to(device, dtype=torch.float32)
        imgV = imgV.to(device, dtype=torch.float32)
        gt = gt.to(device, dtype=torch.int64)

        pred = model(imgN, imgA, imgV)

        loss = PredLoss(pred, gt)

        all_gt.append(gt.squeeze().detach().cpu().numpy())
        all_pred.append(pred.squeeze().detach().cpu().numpy())

        loss.backward()
        optimizer.step()

        # loss update and log
        train_loss += loss.item()
        if batch_idx % int(len(train_dataloader) / 5) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx, len(train_dataloader),
                                                                           100. * batch_idx / len(train_dataloader),
                                                                           loss.item()))

    train_loss /= batch_idx + 1

    pred_array, gt_array = np.concatenate(all_pred), np.concatenate(all_gt)
    pred_class = np.argmax(pred_array, axis=-1)

    acc_s0 = metrics.accuracy_score(y_true=gt_array, y_pred=pred_class)
    auc_s0 = metrics.roc_auc_score(gt_array == 0, pred_array[:, 0])

    print('====> Epoch: {} '.format(epoch))
    print('    > training loss: {:.4f}'.format(train_loss))
    print('    > training acc: {:.4f}'.format(acc_s0))
    print('    > training AUC: {:.4f}'.format(auc_s0))
    print('    > pred 0: {:.4f}'.format(np.sum(pred_class == 0) / len(pred_class)))
    print('    > true 0: {:.4f}'.format(np.sum(gt_array == 0) / len(gt_array)))

    return train_loss, acc_s0, auc_s0


def test(epoch):
    model.eval()
    test_loss = 0

    all_gt, all_pred = [], []
    with torch.no_grad():
        for batch_idx, (imgN, imgA, imgV, gt) in enumerate(val_dataloader):

            if imgN.shape[0] == 1:
                continue

            imgN = imgN.to(device, dtype=torch.float32)
            imgA = imgA.to(device, dtype=torch.float32)
            imgV = imgV.to(device, dtype=torch.float32)

            gt = gt.to(device, dtype=torch.int64)

            pred = model(imgN, imgA, imgV)

            all_gt.append(gt.squeeze().detach().cpu().numpy())
            all_pred.append(pred.squeeze().detach().cpu().numpy())

            test_loss += PredLoss(pred, gt).item()

    test_loss /= batch_idx + 1

    pred_array, gt_array = np.concatenate(all_pred), np.concatenate(all_gt)
    pred_class = np.argmax(pred_array, axis=-1)

    acc_s0 = metrics.accuracy_score(y_true=gt_array, y_pred=pred_class)
    auc_s0 = metrics.roc_auc_score(gt_array == 0, pred_array[:, 0])

    print('====> Epoch: {} '.format(epoch))
    print('    > test loss: {:.4f}'.format(test_loss))
    print('    > test acc: {:.4f}'.format(acc_s0))
    print('    > test auc: {:.4f}'.format(auc_s0))
    print('    > pred 0: {:.4f}'.format(np.sum(pred_class == 0) / len(pred_class)))
    print('    > true 0: {:.4f}'.format(np.sum(gt_array == 0) / len(gt_array)))

    return test_loss, acc_s0, auc_s0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='malignancy_NAV')
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--lr', default=1e-5)
    parser.add_argument('--epochs', default=200)
    parser.add_argument('--GPU', default=0)
    parser.add_argument('--tag', default='NA')

    args = parser.parse_args()

    # hyperparameter
    lr = float(args.lr)
    batch_size = int(args.batch_size)
    N = int(args.epochs)
    gpu_id = int(args.GPU)
    tag = str(args.tag)

    model_type = 'M01-HybridNAV'

    # build model
    model = HybridNAV(input_channel=1)
    device = torch.device("cuda:{:d}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Ndict = {k: v for k, v in torch.load('DL/models/M01_N_pretrained.pt').items() if
    #          'fc' not in k}
    # model.modelN.load_state_dict(Ndict, strict=False)
    # Adict = {k: v for k, v in torch.load('DL/models/M01_A_pretrained.pt').items() if
    #          'fc' not in k}
    # model.modelA.load_state_dict(Adict, strict=False)
    # Vdict = {k: v for k, v in torch.load('DL/models/M01_V_pretrained.pt').items() if
    #          'fc' not in k}
    # model.modelV.load_state_dict(Vdict, strict=False)

    best_model, best_auc = [], 0

    
    import os  
    print("Current working directory:", os.getcwd())
    # load dataset
    train_dataset = HybridSlice(csv_path=f'{os.getcwd()}/csv/demo_split.csv',
                                fold='train', binary='01', augmentation=True, normal=True)
    tmp = train_dataset.get_classes_for_all_imgs()
    weights = 1 / torch.Tensor([np.sum(np.array(tmp) == 0), np.sum(np.array(tmp) == 1)])
    samples_weights = weights[tmp]
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(tmp))
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False, num_workers=4, sampler=sampler)

    val_dataset = HybridSlice(csv_path=f'{os.getcwd()}/csv/demo_split.csv',
                              fold='val', binary='01', augmentation=False, normal=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4)

    # loss function and optimizer
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0002)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model_name = '{}_epoch_{:d}_{}.pt'.format(model_type, N, tag)

    logdir = os.path.join('DL/log', model_name[0:-3])
    writer = SummaryWriter(logdir)
    writer.add_hparams({'type': model_type, 'epochs': N, 'tag': tag}, {})

    # train
    for epoch in tqdm(range(0, N)):
        train_loss, train_acc_s0, train_auc_s0 = train(epoch)
        val_loss, val_acc_s0, val_auc_s0 = test(epoch)

        # update best loss
        last_model = copy.deepcopy(model)
        torch.save(last_model.state_dict(),
                   os.path.join('DL/models', 'last_' + model_name))

        if val_auc_s0 > best_auc:
            best_model = last_model
            best_auc = val_auc_s0

            torch.save(last_model.state_dict(),
                       os.path.join('DL/models', 'best_' + model_name))

        writer.add_scalars('Loss', {'train': train_loss,
                                    'val': val_loss}, epoch)

        writer.add_scalars('acc', {'train': train_acc_s0,
                                   'val': val_acc_s0}, epoch)

        writer.add_scalars('auc', {'train': train_auc_s0,
                                   'val': val_auc_s0}, epoch)

    writer.close()
