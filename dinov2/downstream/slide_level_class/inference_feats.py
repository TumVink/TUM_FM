# Inference script to run any trained model on any test data in any evaluation mode
# pass the model, test data, and evaluation mode as arguments to the script


#export PYTHONPATH=$PYTHONPATH:~/dino-tum in python
import os
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":~/dino-tum"

import argparse
import torch
from dinov2.downstream.eval_patch_features.extract_patch_features import get_dino_finetuned_downloaded, extract_patch_features_from_dataloader
from dinov2.downstream.eval_patch_features.extract_patch_features import get_moco_finetuned_downloaded,get_moco_finetuned_downloaded_dist
from dinov2.downstream.eval_patch_features.extract_patch_features import get_UNI_downloaded_dist
from dinov2.downstream.eval_patch_features.extract_patch_features import extract_patch_features_from_dataloader_dist,get_dino_finetuned_downloaded_dist,extract_patch_features_from_slide_dist
from dinov2.downstream.dataset.patchcam_dataset import PatchCamelyon
from dinov2.downstream.dataset.mhist_dataset import DatasetMHIST
from dinov2.downstream.dataset.crc_dataset import CRC_Dataset
from dinov2.downstream.dataset.slides_dataset import Single_slide_dataset, CAM16_single_slide, Multi_Slides
from dinov2.downstream.eval_patch_features.linear_probe import eval_linear_probe
from dinov2.downstream.eval_patch_features.fewshot import eval_knn
from dinov2.downstream.eval_patch_features.metrics import get_eval_metrics, print_metrics,save_metrics_as_json
import sys
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
import builtins
import pandas as pd
import os

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
            test_labels_all = torch.cat((test_labels_all,gather_list[i]['idx'].to(torch.long)))
    test_feats_all = test_feats_all
    test_labels_all = test_labels_all
    return test_feats_all, test_labels_all

def order_feats_idx(feats,labels):
    # order the features and labels based on the idx
    # feats: torch tensor of shape (N,D) in type float32
    # labels: torch tensor of shape (N,) in type long
    # returns: ordered_feats, ordered_labels

    idx = torch.argsort(labels)
    ordered_feats = feats[idx]
    ordered_labels = labels[idx]
    return ordered_feats, ordered_labels

def save_feats_each_slide(feats,slide_id_ls,num_patches_ls,save_dir):
    # save the features of each slide in a separate file
    # feats: torch tensor of shape (N,D) in type float32
    # slide_id_ls: list of slide ids
    # num_patches_ls: list of number of patches in each slide
    # save_dir: directory to save the features
    for i in range(len(slide_id_ls)):
        slide_path = slide_id_ls[i]
        slide_id = os.path.basename(slide_path).split('.')[0]+'.pt'
        num_patches = num_patches_ls[i]
        slide_feats = feats[:num_patches]
        feats = feats[num_patches:]
        slide_feat_dir = os.path.join(save_dir,slide_id)
        torch.save(slide_feats,slide_feat_dir)





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
    if args.model == 'Dino_giant':
        batch_size = 512
        num_workers = 0
        model_dir = '/home/ge54xof/dino-tum/eval/manual_35400/teacher_checkpoint.pth'
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
    elif args.model == 'Dino_manual_74340':
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

    elif args.model == 'MOCO_4':
        if args.dist:
            ddp_setup(rank, world_size)
            model_dir = '/home/ge54xof/Foundation-Model-for-Pathology/trained_moco/checkpoint_Epo0_Bat4.pth.tar'
            model = get_moco_finetuned_downloaded_dist(moco_path=model_dir,rank=rank)
            model.eval()
        else:
            model_dir = '/home/ge54xof/Foundation-Model-for-Pathology/trained_moco/checkpoint_Epo0_Bat4.pth.tar'
            model = get_moco_finetuned_downloaded(moco_path=model_dir)
            model.eval()

        batch_size = 32
        num_workers = 4

    elif args.model == 'MOCO_5':
        if args.dist:
            ddp_setup(rank, world_size)
            model_dir = '/home/ge54xof/Foundation-Model-for-Pathology/trained_moco/checkpoint_Epo0_Bat5.pth.tar'
            model = get_moco_finetuned_downloaded_dist(moco_path=model_dir,rank=rank)
            model.eval()
        else:
            model_dir = '/home/ge54xof/Foundation-Model-for-Pathology/trained_moco/checkpoint_Epo0_Bat5.pth.tar'
            model = get_moco_finetuned_downloaded(moco_path=model_dir)
            model.eval()

        batch_size = 32
        num_workers = 4

    elif args.model == 'Dino_helmhotz':
        batch_size = 512
        num_workers = 0
        model_dir = '/home/ge54xof/dino-tum/weights/teacher_checkpoint.pth'
        if args.dist:
            ddp_setup(rank, world_size)
            model = get_dino_finetuned_downloaded_dist(DINO_PATH_FINETUNED_DOWNLOADED=model_dir,rank=rank)
            batch_size = 16
            num_workers = 4
            model.eval()
        else:
            model = get_dino_finetuned_downloaded(DINO_PATH_FINETUNED_DOWNLOADED=model_dir)
            model.to(device)
            model.eval()
    elif args.model == 'UNI':
        batch_size = 512
        num_workers = 0
        #model_dir = '/home/ge54xof/dino-tum/weights/teacher_checkpoint.pth'

        #login()
        if args.dist:
            ddp_setup(rank, world_size)
            model = get_UNI_downloaded_dist(UNI_dir=None,rank=rank)
            batch_size = 16
            num_workers = 4
            model.eval()
        else:
            model = get_UNI_downloaded_dist(DINO_PATH_FINETUNED_DOWNLOADED=model_dir)
            model.to(device)
            model.eval()
    else:
        return None
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
    elif args.test_data == 'CAM17':
        print('CAMELYON17')
        data_root = '/mnt/nfs03-R6/CAMELYON17/'
        #train_file = data_root+'split/train_slides.csv'
        train_pd = pd.read_csv(data_root+'split/train_slides.csv')
        train_len = len(train_pd['patient'].values.tolist())
        batch_size = 512
        num_workers = 0
        for i in range(train_len):
            slide_id = train_pd['patient'].values[i]
            num_patches = train_pd['num_patches'].values[i]
            train_dataset = Single_slide_dataset(data_root, slide_id,num_patches,split='train',transform=transform)
            save_feat_dir = data_root+'split/feats_'+args.model
            train_feat_dir = os.path.join(save_feat_dir,slide_id.replace('.tif','_train.pth'))
    elif args.test_data == 'CAM16_multi_slides':
        print('CAM16_multi_slides')
        data_root = '/mnt/nfs03-R6/CAMELYON16/clam_all_patches/'
        # train_file = data_root+'split/train_slides.csv'
        slides_df = data_root + 'process_list_autogen.csv'

        train_dataset = Multi_Slides(data_root, slides_df, split='train', transform=transform)
        save_feat_dir = '/mnt/nfs03-R6/CAMELYON16/' + 'feats/feats_' + args.model
        train_feat_dir = os.path.join(save_feat_dir, 'batch_'+str(i)+'.pth')
        batch_size = 32
        num_workers = 0
        if args.dist:
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                                           pin_memory=True,
                                                           sampler=DistributedSampler(train_dataset),
                                                           num_workers=num_workers)
            train_feats = extract_patch_features_from_slide_dist(model, train_dataloader)
            torch.distributed.barrier()
            train_feats_all,train_labels_all = gather_dist_results(world_size, rank, train_feats, device)
            order_feats, order_labels = order_feats_idx(train_feats_all, train_labels_all)
            if rank == 0:
                mask = slides_df['status'] == "tbs"
                data_pd = slides_df[mask]
                num_ls = data_pd['process'].to_numpy()
                slide_id_ls = data_pd['slide_id'].values.tolist()
                save_feats_each_slide(order_feats,num_ls, slide_id_ls, save_feat_dir)

    elif args.test_data == 'SLN':
        print('SLN_multi_slides')
        data_root = '/mnt/nfs03-R6/SLN_cli/clam_all_patches/'
        # train_file = data_root+'split/train_slides.csv'
        slides_df = data_root + 'process_list_autogen.csv'

        train_dataset = Multi_Slides(data_root, slides_df, split='train', transform=transform)
        save_feat_dir = '/mnt/nfs03-R6/SLN_cli/' + 'feats/feats_' + args.model

        batch_size = 16
        num_workers = 4
        if args.dist:
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                                           pin_memory=True,
                                                           sampler=DistributedSampler(train_dataset),
                                                           num_workers=num_workers)
            train_feats = extract_patch_features_from_slide_dist(model, train_dataloader)
            torch.distributed.barrier()
            train_feats_all,train_labels_all = gather_dist_results(world_size, rank, train_feats, device)
            order_feats, order_labels = order_feats_idx(train_feats_all, train_labels_all)
            if rank == 0:
                data_pd = pd.read_csv(slides_df)
                #data_pd = data_pd.iloc[:10, :]
                mask = data_pd['status'] == "processed"
                data_pd = data_pd[mask]
                num_ls = data_pd['process'].to_numpy()
                slide_id_ls = data_pd['slide_id'].values.tolist()
                save_feats_each_slide(order_feats,slide_id_ls, num_ls, save_feat_dir)



    elif args.test_data == 'CAM16':
        print('CAMELYON16')
        data_root = '/mnt/nfs03-R6/CAMELYON16/clam_all_patches/'
        # train_file = data_root+'split/train_slides.csv'
        slide_df = pd.read_csv(data_root + 'process_list_autogen.csv')
        train_len = len(slide_df['slide_id'].values.tolist())
        batch_size = 16
        num_workers = 0
        for i in range(train_len):
            slide_path = slide_df['slide_id'].values[i]
            slide_id = os.path.basename(slide_path).split('.')[0]+'.h5'
            num_patches = slide_df['process'].values[i]
            train_dataset = CAM16_single_slide(data_root, slide_id, num_patches, split='train', transform=transform)
            save_feat_dir = '/mnt/nfs03-R6/CAMELYON16/' + 'feats/feats_' + args.model
            train_feat_dir = os.path.join(save_feat_dir, slide_id.replace('.h5', '.pth'))

            if os.path.exists(train_feat_dir):
                print('already exists')
                pass
            elif args.dist:
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,pin_memory=True,
                                                               sampler=DistributedSampler(train_dataset),num_workers=num_workers)
                # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True,
                #                                               sampler = DistributedSampler(test_dataset),num_workers=num_workers)
                #
                # if val_dataset is not None:
                #     val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,pin_memory=True,
                #                                                  sampler=DistributedSampler(val_dataset),num_workers=num_workers)

                #test_feats = extract_patch_features_from_slide_dist(model, test_dataloader)
                train_feats = extract_patch_features_from_slide_dist(model, train_dataloader)
                # if val_dataset is not None:
                #     val_feats = extract_patch_features_from_slide_dist(model, test_dataloader)
                #print(test_features['embeddings'].shape)
                #print(test_features['labels'].shape)
                # test_feats_all = torch.zeros((len(test_dataset),test_features['embeddings'].shape[1]),
                #                                 dtype = test_features['embeddings'].dtype, device = test_features['embeddings'].device)
                # test_labels_all = torch.zeros(len(test_dataset),dtype = torch.long,
                #                               device = test_features['embeddings'].device)

                # train_features = extract_patch_features_from_dataloader_dist(model, train_dataloader)
                # train_features_all = torch.zeros((len(train_dataset),train_features['embeddings'].shape[1]),
                #                                 dtype = train_features['embeddings'].dtype, device = train_features['embeddings'].device)
                # train_labels_all = torch.zeros(len(train_dataset),dtype = torch.long,
                #                               device = train_features['embeddings'].device)

                torch.distributed.barrier()
                # if val_dataset is not None:
                #     val_features = extract_patch_features_from_dataloader_dist(model, val_dataloader)
                #     val_features_all = torch.zeros((len(val_dataset),test_features['embeddings'].shape[1]),
                #                                     dtype = val_features['embeddings'].dtype, device = val_features['embeddings'].device)
                #
                #     val_labels_all = torch.zeros(len(val_dataset),dtype = torch.long,
                #                                   device = val_features['embeddings'].device)
                #     torch.distributed.all_gather_into_tensor(val_features_all, val_features['embeddings'])
                #     torch.distributed.all_gather_into_tensor(val_labels_all,
                #                                              torch.Tensor(val_features['labels']).type(torch.long))
                # torch.distributed.all_gather_into_tensor(train_features_all, train_features['embeddings'])
                # torch.distributed.all_gather_into_tensor(train_labels_all, torch.Tensor(train_features['labels']).type(torch.long))
                # if rank ==0:
                #     torch.save(train_features_all, train_feat_dir)
                #     torch.save(train_labels_all, train_labels_dir)

                #idea0: using dist.all_gather_object
                train_feats_all = gather_dist_results(world_size, rank, train_feats,device)
                # test_feats_all,test_labels_all = gather_dist_results(world_size,rank,test_feats,device)
                # if val_dataset is not None:
                #     val_feats_all, val_labels_all = gather_dist_results(world_size,rank,val_feats,device)
                #     if rank == 0:
                #         torch.save(val_feats_all, val_feat_dir)
                #         torch.save(val_labels_all, val_labels_dir)
                #         val_feats = val_feats_all
                #         val_labels = val_labels_all
                if rank == 0:
                    print('gathered')
                    print(train_feats_all.shape)
                    #print(test_labels_all.shape)
                    torch.save(train_feats_all, train_feat_dir)
                #torch.save(train_labels_all, train_labels_dir)
                # torch.save(test_feats_all, test_feat_dir)
                # torch.save(test_labels_all, test_labels_dir)




if __name__ == "__main__":
   import sys
   parser = argparse.ArgumentParser(description='Inference script to run the trained model on test data')
   parser.add_argument('--model', default='UNI', type=str, help='Path to the model file')
   parser.add_argument('--test_data', default='CAM16', type=str, help='Path to the test data file')
   parser.add_argument('--eval_mode', default='KNN', type=str, help='Evaluation mode: test or val')
   parser.add_argument('--dist', default=True, help='Use multi GPU to speed up the inference')
   args = parser.parse_args()
   if args.dist:
       if args.model == 'UNI':
           from huggingface_hub import login
           login()
       world_size = torch.cuda.device_count()
       print(world_size)
       mp.spawn(main, args=(world_size, args), nprocs=world_size)

   else:
       main(0,1,args)












