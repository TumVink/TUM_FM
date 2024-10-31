from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                                                                         csv_path='{}/splits_{}.csv'.format(
                                                                             args.split_dir, i))

        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        # write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc,
                             'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc': all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))


# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default='/mnt/nfs03-R6/CAMELYON16/slides_cls',
                    help='data directory')
parser.add_argument('--feats_dir', type=str, default='feats_UNI',
                    help='feats directory related to model type')
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default='',
                    help='manually specify the set of splits to use, '
                         + 'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce','focal'
                                                                 ], default='ce',
                    help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil','clam_abmil'], default='clam_sb',
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
#parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--weighted_sample', default=True, help='enable weighted sampling or not')
parser.add_argument('--model_size', type=str, choices=['small', 'big', 'large'], default='small',
                    help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping', 'luad_vs_lusc','msi_vs_mss'],
                    default='task_1_tumor_vs_normal')
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                    help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                    help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False,
                    help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--alpha_focal', type=float, default=0.25,
                    help='alpha value for focal loss (default: 0.25)')
parser.add_argument('--gamma_focal', type=float, default=2.0,
                    help='gamma value for focal loss (default: 2.0), only used if loss is focal')
parser.add_argument('--B', type=int, default=8, help='number of positive/negative patches to sample for clam')
parser.add_argument('--csv_path', type=str, default='tumor_vs_normal.csv',
                    help='specify the csv file for split')
parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
parser.add_argument('--stop_epoch', type=int, default=20, help='early stopping stop_epoch')
parser.add_argument('--para_group', type=str, help='parameters group from grid research')
parser.add_argument('--stopping_criterion', type=str, default='loss',choices= ['loss','error','auc'] ,help='early stopping criterion')


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
    settings.update({'bag_weight': args.bag_weight,
                     'inst_loss': args.inst_loss,
                     'B': args.B})

print('\nLoad Dataset')

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path=os.path.join(args.data_root_dir, args.csv_path),
                                  data_dir=os.path.join(args.data_root_dir, args.feats_dir),
                                  shuffle=False,
                                  seed=args.seed,
                                  print_info=True,
                                  label_dict={'normal': 0, 'tumor': 1},
                                  patient_strat=False,
                                  ignore=[])
elif args.task == 'msi_vs_mss':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path=os.path.join(args.data_root_dir, args.csv_path),
                                  data_dir=os.path.join(args.data_root_dir, args.feats_dir),
                                  shuffle=False,
                                  seed=args.seed,
                                  print_info=True,
                                  label_dict={'MSS':0, 'MSI':1},
                                  patient_strat=False,
                                  ignore=[])
elif args.task == 'luad_vs_lusc':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path=os.path.join(args.data_root_dir, args.csv_path),
                                  data_dir=os.path.join(args.data_root_dir, args.feats_dir),
                                  shuffle=False,
                                  seed=args.seed,
                                  print_info=True,
                                  label_dict={'LUAD': 0, 'LUSC': 1},
                                  patient_strat=False,
                                  ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes = 3
    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/tumor_subtyping_dummy_clean.csv',
                                  data_dir=os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                                  shuffle=False,
                                  seed=args.seed,
                                  print_info=True,
                                  label_dict={'subtype_1': 0, 'subtype_2': 1, 'subtype_3': 2},
                                  patient_strat=False,
                                  ignore=[])

    if args.model_type in ['clam_sb', 'clam_mb']:
        assert args.subtyping

else:
    raise NotImplementedError

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + args.para_group)
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.csv_path.endswith('new.csv'):
    print('mode: new, treat MSI-L as MSS ')
    args.split_dir = os.path.join(args.data_root_dir, 'splits_new')
else:
    args.split_dir = os.path.join(args.data_root_dir, 'splits')

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))

if __name__ == "__main__":
    results = main(args)
    # dataset = Generic_MIL_Dataset(csv_path = os.path.join(args.data_root_dir, 'tumor_vs_normal_cam16.csv'),
    #                         data_dir= os.path.join(args.data_root_dir, args.feats_dir),
    #                         shuffle = False,
    #                         seed = args.seed,
    #                         print_info = True,
    #                         label_dict = {'normal':0, 'tumor':1},
    #                         patient_strat=False,
    #                         ignore=[])
    # train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
    #                                                                  csv_path='{}/splits_{}.csv'.format(args.split_dir,
    #                                                                                                     0))
    # print(train_dataset.slide_cls_ids[0].shape[0])
    # print(train_dataset.slide_cls_ids[1].shape[0])
    # print(val_dataset.slide_cls_ids[0].shape[0])
    # print(val_dataset.slide_cls_ids[1].shape[0])
    # print(test_dataset.slide_cls_ids[0].shape[0])
    # print(test_dataset.slide_cls_ids[1].shape[0])

    # iterate the training_dataset
    # for i, (data, label) in enumerate(train_dataset):
    #     print(data.shape)
    #     print(label)
    #     break
    #
    #
    # print("finished!")
    # print("end script")

    # running command
    # python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 1 --weighted_sample --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type clam_sb --feats_dir
    # python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 1 --weighted_sample --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type clam_mb --feats_dir feats_Dino_tum --data_root_dir /mnt/nfs03-R6/SLN_cli/slides_cls/ --embed_dim 1536 --exp_code TUM_mb

