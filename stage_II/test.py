# -*- coding:utf-8 -*-

import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from qdnet.conf.config import load_yaml
from qdnet.optimizer.optimizer import GradualWarmupSchedulerV2
from qdnet.dataset.dataset import get_df, QDDataset
from qdnet.dataaug.dataaug import get_transforms
from qdnet.models.effnet import Effnet
from qdnet.models.resnest import Resnest
from qdnet.models.se_resnext import SeResnext
from qdnet.loss.loss import Loss
from qdnet.conf.constant import Constant


parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--config_path',default= "conf/resnest101.yaml", help='config file path')
# parser.add_argument('--n_splits', help='n_splits', type=int)
args = parser.parse_args()
config = load_yaml(args.config_path, args)


def test_epoch(model, loader, mel_idx, get_output=False):

    model.eval()
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):
            
            data, target = data.to(device), target.to(device)
            probs = model(data)

            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    # note: since the model only train for classifying positive (1) and negative (0),
    # samples with uncertain label (2) are discarded when evaluating the classification performance
    acc = (PROBS.argmax(1) == TARGETS)[TARGETS != 2].mean()
    auc = roc_auc_score((TARGETS[TARGETS != 2] == mel_idx).astype(float), PROBS[TARGETS != 2, mel_idx])
    return acc, auc, PROBS.argmax(1)

def main():

    df_test, mel_idx = get_df( config["data_dir"], config["auc_index"] , stage = 'test') 

    _, transforms_val = get_transforms(int(config["image_size"]))   

    LOGITS = []
    PROBS = []
    dfs = []
    # for fold in range(args.n_splits):

        # df_valid = df[df['fold'] == fold]

    dataset_test = QDDataset(df_test, 'valid', transform=transforms_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=int(config["batch_size"]), num_workers=int(config["num_workers"]),drop_last=False)

    if config["eval"] == 'best':
        model_file = os.path.join(config["model_dir"], f'best_fold.pth')
    if config["eval"] == 'final':
        model_file = os.path.join(config["model_dir"], f'final_fold.pth')

    model = ModelClass(
        enet_type = config["enet_type"],
        out_dim = int(config["out_dim"]),
        drop_nums = int(config["drop_nums"]),
        metric_strategy = config["metric_strategy"]
    )
    model = model.to(device)

    try:  # single GPU model_file
        model.load_state_dict(torch.load(model_file), strict=True)
    except:  # multi GPU model_file
        state_dict = torch.load(model_file)
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)

    if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
        model = torch.nn.DataParallel(model)

    model.eval()

    # this_LOGITS, this_PROBS = val_epoch(model, valid_loader, mel_idx, get_output=True)
    # LOGITS.append(this_LOGITS)
    # PROBS.append(this_PROBS)
    # dfs.append(df_valid)

    acc, auc, pred_target = test_epoch(model, test_loader, mel_idx)

    # dfs = pd.concat(dfs).reset_index(drop=True)
    # dfs['pred'] = np.concatenate(PROBS).squeeze()[:, mel_idx]
    #
    # auc_all_raw = roc_auc_score(dfs['target'] == mel_idx, dfs['pred'])
    #
    # dfs2 = dfs.copy()
    # for i in range(args.n_splits):
    #     dfs2.loc[dfs2['fold'] == i, 'pred'] = dfs2.loc[dfs2['fold'] == i, 'pred'].rank(pct=True)
    # auc_all_rank = roc_auc_score(dfs2['target'] == mel_idx, dfs2['pred'])
    #
    # content = f'Eval {config["eval"]}:\nauc_all_raw : {auc_all_raw:.5f}\nauc_all_rank : {auc_all_rank:.5f}\n'

    content = f'Eval {config["eval"]}:\nacc: {acc:.5f}\nauc : {auc:.5f}\n'
    print(content)
    with open(os.path.join(config["log_dir"], f'log.txt'), 'a') as appender:  
        appender.write(content + '\n')

    # np.save(os.path.join(config["oof_dir"], f'{config["eval"]}_oof.npy'), dfs['pred'].values)
    np.save(os.path.join(config["oof_dir"], f'{config["eval"]}_pred_target.npy'), pred_target)

if __name__ == '__main__':

    os.makedirs(config["oof_dir"], exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = config["CUDA_VISIBLE_DEVICES"]

    if config["enet_type"] in Constant.RESNEST_LIST:   
        ModelClass = Resnest
    elif config["enet_type"] in Constant.SERESNEXT_LIST:
        ModelClass = SeResnext
    elif config["enet_type"] in Constant.GEFFNET_LIST:
        ModelClass = Effnet
    else:
        raise NotImplementedError()

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss()

    main()
