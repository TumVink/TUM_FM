# Inference script to run any trained model on any test data in any evaluation mode
# pass the model, test data, and evaluation mode as arguments to the script
import argparse
import torch
from eval_patch_features.extract_patch_features import get_dino_finetuned_downloaded, extract_patch_features_from_dataloader
from eval_patch_features.extract_patch_features import get_moco_finetuned_downloaded,get_moco_finetuned_downloaded_dist
from eval_patch_features.extract_patch_features import extract_patch_features_from_dataloader_dist,get_dino_finetuned_downloaded_dist
from dataset.patchcam_dataset import PatchCamelyon
from dataset.mhist_dataset import DatasetMHIST
from dataset.crc_dataset import CRC_Dataset
from eval_patch_features.linear_probe import eval_linear_probe
from eval_patch_features.fewshot import eval_knn
from eval_patch_features.metrics import get_eval_metrics, print_metrics,save_metrics_as_json
import sys
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
import os
import builtins
#os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_SOCKET_IFNAME"] = "ibp170s0f0"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

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



def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1234"

    #print('finish')
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(rank)


def main(rank,world_size,args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dist and rank !=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    # Load the model

    if args.model == 'Dino_manual_74340':
        batch_size = 512
        num_workers = 0
        model_dir = '/home/ge54xof/dino-tum/eval/manual_74340/teacher_checkpoint.pth'
        if args.dist:
            ddp_setup(rank, world_size)
            model = get_dino_finetuned_downloaded_dist(DINO_PATH_FINETUNED_DOWNLOADED=model_dir,rank=rank)
            model.eval()
            batch_size = 16
            num_workers = 4
        else:
            model = get_dino_finetuned_downloaded(DINO_PATH_FINETUNED_DOWNLOADED=model_dir)
            model.to(device)
            model.eval()

        #pass
    # define dataset
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    # # Load the test data
    if args.test_data == 'Patch_cam':
        data_root = '/raid/pcamv1/'
        train_dataset = PatchCamelyon(data_root, 'train', transform=transform)
        test_dataset = PatchCamelyon(data_root, 'test', transform=transform)
        val_dataset = PatchCamelyon(data_root, 'val', transform=transform)
    elif args.test_data == 'MHIST':
        data_root = '/mnt/nfs03-R6/mhist/'
        train_dataset = DatasetMHIST(data_root, 'train', transform=transform)
        val_dataset = DatasetMHIST(data_root, 'val', transform=transform)
        test_dataset = DatasetMHIST(data_root, 'test', transform=transform)
    elif args.test_data == 'CRC':
        print('CRC')
        data_root = '/mnt/nfs03-R6/CRC/'
        train_dataset = CRC_Dataset(data_root, 'train', transform=transform)
        test_dataset = CRC_Dataset(data_root, 'test', transform=transform)
        val_dataset = None
        #num_workers = 0

    save_feat_dir = '/home/ge54xof/dino-tum/dinov2/downstream/feats'
    save_labels_dir = '/home/ge54xof/dino-tum/dinov2/downstream/labels'
    train_feat_dir = os.path.join(save_feat_dir, args.test_data+'_'+args.model+'_train.pth')
    train_labels_dir = os.path.join(save_labels_dir, args.test_data + '_' + args.model + '_train.pth')

    val_feat_dir = os.path.join(save_feat_dir, args.test_data + '_' + args.model + '_val.pth')
    val_labels_dir = os.path.join(save_labels_dir, args.test_data + '_' + args.model + '_val.pth')
    test_feat_dir = os.path.join(save_feat_dir, args.test_data + '_' + args.model + '_test.pth')
    test_labels_dir = os.path.join(save_labels_dir, args.test_data + '_' + args.model + '_test.pth')

    if os.path.exists(test_feat_dir):
        train_feats = torch.load(train_feat_dir).to(device).to(torch.float32)
        train_labels = torch.load(train_labels_dir).to(device).to(torch.long)
        print(train_feats.shape)
        print(train_labels.shape)
        if val_dataset is not None:
            val_feats = torch.load(val_feat_dir).to(device).to(torch.float32)
            val_labels = torch.load(val_labels_dir).to(device).to(torch.long)
            print(val_feats.shape)
            print(val_labels.shape)
        test_feats = torch.load(test_feat_dir).to(device).to(torch.float32)
        test_labels = torch.load(test_labels_dir).to(device).to(torch.long)
    elif args.dist:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,pin_memory=True,
                                                       sampler=DistributedSampler(train_dataset),num_workers=num_workers)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True,
                                                      sampler = DistributedSampler(test_dataset),num_workers=num_workers)

        if val_dataset is not None:
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,pin_memory=True,
                                                         sampler=DistributedSampler(val_dataset),num_workers=num_workers)

        test_feats = extract_patch_features_from_dataloader_dist(model, test_dataloader)
        train_feats = extract_patch_features_from_dataloader_dist(model, train_dataloader)
        if val_dataset is not None:
            val_feats = extract_patch_features_from_dataloader_dist(model, test_dataloader)

        torch.distributed.barrier()

        #idea0: using dist.all_gather_object
        train_feats_all, train_labels_all = gather_dist_results(world_size, rank, train_feats,device)
        test_feats_all,test_labels_all = gather_dist_results(world_size,rank,test_feats,device)
        if val_dataset is not None:
            val_feats_all, val_labels_all = gather_dist_results(world_size,rank,val_feats,device)
            if rank == 0:
                torch.save(val_feats_all, val_feat_dir)
                torch.save(val_labels_all, val_labels_dir)
                val_feats = val_feats_all
                val_labels = val_labels_all

        # Save all features and labels only on rank 0
        if rank == 0:
            print('gathered')
            print(test_feats_all.shape)
            print(test_labels_all.shape)
            torch.save(train_feats_all, train_feat_dir)
            torch.save(train_labels_all, train_labels_dir)
            torch.save(test_feats_all, test_feat_dir)
            torch.save(test_labels_all, test_labels_dir)

            # load  again
            train_feats = torch.load(train_feat_dir).to(device).to(torch.float32)
            train_labels = torch.load(train_labels_dir).to(device).to(torch.long)
            print(train_feats.shape)
            print(train_labels.shape)
            if val_dataset is not None:
                val_feats = torch.load(val_feat_dir).to(device).to(torch.float32)
                val_labels = torch.load(val_labels_dir).to(device).to(torch.long)
                print(val_feats.shape)
                print(val_labels.shape)
            test_feats = torch.load(test_feat_dir).to(device).to(torch.float32)
            test_labels = torch.load(test_labels_dir).to(device).to(torch.long)

        else:
            return None

    # Save all features and labels
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
        if val_dataset is not None:
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
        # Extract features from the data
        #train_features = extract_patch_features_from_dataloader(model, train_dataloader)
        test_features = extract_patch_features_from_dataloader(model, test_dataloader)
        if val_dataset is not None:
            val_features = extract_patch_features_from_dataloader(model, val_dataloader)
        # convert these to torch
        #train_feats = torch.Tensor(train_features['embeddings'])
        #train_labels = torch.Tensor(train_features['labels']).type(torch.long)
        #print('Size of train_feats:',train_feats.size())
        test_feats = torch.Tensor(test_features['embeddings'])
        print('Size of test:', test_feats.size())
        test_labels = torch.Tensor(test_features['labels']).type(torch.long)
        if val_dataset is not None:
            val_feats = torch.Tensor(val_features['embeddings'])
            print('Size of val:', val_feats.size())
            val_labels = torch.Tensor(val_features['labels']).type(torch.long)

        # Save the features
        #torch.save(train_feats,  train_feat_dir)
        #torch.save(train_labels, train_labels_dir)
        if val_dataset is not None:
            torch.save(val_feats, val_feat_dir)
            torch.save(val_labels, val_labels_dir)
        torch.save(test_feats, test_feat_dir)
        torch.save(test_labels, test_labels_dir)

    # print('Bit size of train_feats:', sys.getsizeof(train_feats))
    # print('Bit size of test_feats:', sys.getsizeof(test_feats))
    #
    # #### Linear probe ####
    #if args.eval_mode == 'linear':
    if val_dataset is None:
        val_feats = None
        val_labels = None
    linprobe_eval_metrics, linprobe_dump = eval_linear_probe(
        train_feats=train_feats,
        train_labels=train_labels,
        valid_feats=val_feats ,
        valid_labels=val_labels,
        test_feats=test_feats,
        test_labels=test_labels,
        max_iter=1000,
        verbose=True,
    )
    print_metrics(linprobe_eval_metrics)
    linear_save_dir = os.path.join('/home/ge54xof/dino-tum/dinov2/downstream/results', args.test_data + '_' + args.model + '_linear.json')
    save_metrics_as_json(linprobe_eval_metrics, linear_save_dir)
    #
    # #### Few-shot ####
    #elif args.eval_mode == 'KNN':
    # set env variable OPENBLAS_NUM_THREADS to 64 or lower
    #os.environ["OPENBLAS_NUM_THREADS"] = "4"
    knn_eval_metrics, knn_dump, proto_eval_metrics, proto_dump = eval_knn(
        train_feats=train_feats,
        train_labels=train_labels,
        test_feats=test_feats,
        test_labels=test_labels,
        center_feats=True,
        normalize_feats=True,
        n_neighbors=10
    )
    knn_save_dir = os.path.join('/home/ge54xof/dino-tum/dinov2/downstream/results',
                                   args.test_data + '_' + args.model + '_KNN.json')
    print_metrics(knn_eval_metrics)
    save_metrics_as_json(knn_eval_metrics, knn_save_dir)
    print_metrics(proto_eval_metrics)
    proto_save_dir = os.path.join('/home/ge54xof/dino-tum/dinov2/downstream/results',
                                   args.test_data + '_' + args.model + '_proto.json')
    save_metrics_as_json(proto_eval_metrics, proto_save_dir)
    #destroy_process_group()

if __name__ == "__main__":
   import sys
   parser = argparse.ArgumentParser(description='Inference script to run the trained model on test data')
   parser.add_argument('--model', default='Dino_giant', type=str, help='Path to the model file')
   parser.add_argument('--test_data', default='MHIST', type=str, help='Path to the test data file')
   parser.add_argument('--eval_mode', default='KNN', type=str, help='Evaluation mode: test or val')
   parser.add_argument('--dist', default=False, help='Use multi GPU to speed up the inference')
   args = parser.parse_args()
   if args.dist:
       world_size = torch.cuda.device_count()
       print(world_size)
       mp.spawn(main, args=(world_size, args), nprocs=world_size)
   else:
       linear_save_dir = os.path.join('/home/ge54xof/dino-tum/dinov2/downstream/results',
                                      args.test_data + '_' + args.model + '_linear.json')
       if os.path.exists(linear_save_dir):
           print('Results already exist')
       else:
           main(0,1,args)











