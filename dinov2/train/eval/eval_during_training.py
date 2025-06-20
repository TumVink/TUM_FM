from dinov2.downstream.eval_patch_features.extract_patch_features import get_dino_finetuned_downloaded, \
    extract_patch_features_from_dataloader, get_dino_large_finetued_downloaded_dist, \
    extract_patch_features_from_dataloader_dist
from dinov2.downstream.eval_patch_features.metrics import save_metrics_as_json
from dinov2.downstream.dataset.mhist_dataset import DatasetMHIST
from dinov2.downstream.dataset.crc_dataset import CRC_Dataset
from dinov2.downstream.dataset.ccrcc_dataset import CCRCC_patch
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from dinov2.downstream.eval_patch_features.linear_probe import eval_linear_probe
import torch.distributed as dist
import os


def broadcast_result_dict(result_dict, src=0):
    object_list = [result_dict if dist.get_rank() == src else None]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]

def gather_dist_results(world_size,rank,features,device):
    gather_list_element = {}
    gather_list = [gather_list_element for _ in range(world_size)]
    torch.distributed.all_gather_object(gather_list, features)
    # extract features and labels from gather_list only in rank 0
    test_feats_all = torch.tensor(())
    test_labels_all = torch.tensor(())
    if rank == 0:
        for i in range(len(gather_list)):
            test_feats_all = torch.cat((test_feats_all,gather_list[i]['embeddings'].to(torch.float32)))
            test_labels_all = torch.cat((test_labels_all,gather_list[i]['labels'].to(torch.long)))
    test_feats_all = test_feats_all.to(device)
    test_labels_all = test_labels_all.to(device)
    return test_feats_all, test_labels_all

def inf_during_training(variant='vit_giant2',ckp_path=None, local_id=0,iter='training_0'):
    patch_tasks = ['mhist','crc_norm', 'crc_unnorm','ccrcc']
    lin_acc_ls = []
    lin_bacc_ls = []

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    for task in patch_tasks:
        if task == 'mhist':
            data_root = '/mnt/nfs03-R6/mhist/'
            train_dataset = DatasetMHIST(data_root, 'train', transform=transform)
            val_dataset = DatasetMHIST(data_root, 'val', transform=transform)
            test_dataset = DatasetMHIST(data_root, 'test', transform=transform)
        elif task == 'crc_norm':
            data_root = '/mnt/nfs03-R6/CRC/'
            train_dataset = CRC_Dataset(data_root, 'train', transform=transform, norm=True)
            test_dataset = CRC_Dataset(data_root, 'test', transform=transform, norm=True)
            val_dataset = None
        elif task == 'crc_unnorm':
            data_root = '/mnt/nfs03-R6/CRC/'
            train_dataset = CRC_Dataset(data_root, 'train', transform=transform, norm=False)
            test_dataset = CRC_Dataset(data_root, 'test', transform=transform, norm=False)
            val_dataset = None
        elif task == 'ccrcc':
            data_root = '/mnt/nfs03-R6/CCRCC_patch_cls/tissue_classification/'
            train_dataset = CCRCC_patch(data_root, 'train', transform=transform)
            test_dataset = CCRCC_patch(data_root, 'test', transform=transform)
            val_dataset = None
        else:
            raise ValueError(f'Unknown task: {task}')
        # data_root = '/mnt/nfs03-R6/mhist/'
        # train_dataset = DatasetMHIST(data_root, 'train', transform=transform)
        # val_dataset = DatasetMHIST(data_root, 'val', transform=transform)
        # test_dataset = DatasetMHIST(data_root, 'test', transform=transform)
        batch_size = 32
        num_workers = 4

        model = get_dino_large_finetued_downloaded_dist(DINO_PATH_FINETUNED_DOWNLOADED=ckp_path, rank=local_id,variant=variant)
        model.eval()

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                       sampler=DistributedSampler(train_dataset), num_workers=num_workers)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                      sampler=DistributedSampler(test_dataset), num_workers=num_workers)

        train_feats = extract_patch_features_from_dataloader_dist(model, train_dataloader)
        test_feats = extract_patch_features_from_dataloader_dist(model, test_dataloader)
        if val_dataset is not None:
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                         sampler=DistributedSampler(val_dataset), num_workers=num_workers)
            val_feats = extract_patch_features_from_dataloader_dist(model, val_dataloader)

        torch.distributed.barrier()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        world_size = torch.distributed.get_world_size()
        # idea0: using dist.all_gather_object
        train_feats_all, train_labels_all = gather_dist_results(world_size, local_id, train_feats, device)
        test_feats_all, test_labels_all = gather_dist_results(world_size, local_id, test_feats, device)
        if val_feats is not None:
            val_feats_all, val_labels_all = gather_dist_results(world_size, local_id, val_feats, device)
            val_labels_all = val_labels_all.to(torch.long)
        else:
            val_feats_all = None
            val_labels_all = None
        #only rank 0 will run the linear probe
        if local_id == 0:
            train_labels_all = train_labels_all.to(torch.long)
            test_labels_all = test_labels_all.to(torch.long)
            #print the datatype of labels
            print(f'train_labels_all dtype: {train_labels_all.dtype}')
            print(f'test_labels_all dtype: {train_feats_all.dtype}')
            linprobe_eval_metrics, linprobe_dump = eval_linear_probe(
                train_feats=train_feats_all,
                train_labels=train_labels_all,
                valid_feats=val_feats_all,
                valid_labels=val_labels_all,
                test_feats=test_feats_all,
                test_labels=test_labels_all,
                max_iter=1000,
                verbose=True,
                combine_trainval=False
            )
            #print(f'linprobe_eval_metrics: {linprobe_eval_metrics}')
            lin_acc_ls.append(linprobe_eval_metrics['lin_acc'])
            lin_bacc_ls.append(linprobe_eval_metrics['lin_bacc'])
            linear_save_dir = os.path.join('/home/ge54xof/dino-tum/dinov2/downstream/results/jsons',
                                           task + '_' + iter + '_linear.json')
            save_metrics_as_json(linprobe_eval_metrics, linear_save_dir)
    dist.barrier()
    #calculate the mean of lin_acc_ls and lin_bacc_ls
    if local_id == 0:
        lin_acc = sum(lin_acc_ls) / len(lin_acc_ls)
        lin_bacc = sum(lin_bacc_ls) / len(lin_bacc_ls)
        print(f'lin_acc: {lin_acc}')
        print(f'lin_bacc: {lin_bacc}')
        lin_dict = {'lin_acc': lin_acc, 'lin_bacc': lin_bacc}
    else:
        lin_dict = {'lin_acc': 0, 'lin_bacc': 0}
    result = broadcast_result_dict(lin_dict, src=0)

    # if local_id == 1:
    #     print(f'linprobe_eval_metrics: {linprobe_eval_metrics}')
    return result

def test():
    pretrained = torch.load('/home/ge54xof/dino-tum/dinov2/train/eval/training_0/teacher_checkpoint.pth', map_location=torch.device('cpu'))
    print(pretrained['teacher'].keys())

if __name__ == '__main__':
    test()