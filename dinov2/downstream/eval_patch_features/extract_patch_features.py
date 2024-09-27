import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
# from ..models import build_model_from_cfg
# from ..utils.config import setup

#from ..get_encoder import get_encoder

torch.multiprocessing.set_sharing_strategy("file_system")

@torch.no_grad()
def extract_patch_features_from_dataloader(model, dataloader):
    """Uses model to extract features+labels from images iterated over the dataloader.

    Args:
        model (torch.nn): torch.nn CNN/VIT architecture with pretrained weights that extracts d-dim features.
        dataloader (torch.utils.data.DataLoader): torch.utils.data.DataLoader object of N images.

    Returns:
        dict: Dictionary object that contains (1) [N x D]-dim np.array of feature embeddings, and (2) [N x 1]-dim np.array of labels

    """
    all_embeddings, all_labels = [], []
    batch_size = dataloader.batch_size
    device = next(model.parameters())[0].device

    for batch_idx, (batch, target) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        remaining = batch.shape[0]
        if remaining != batch_size:
            _ = torch.zeros((batch_size - remaining,) + batch.shape[1:]).type(
                batch.type()
            )
            batch = torch.vstack([batch, _])

        batch = batch.to(device)
        with torch.inference_mode():
            embeddings = model(batch).detach().cpu()[:remaining, :].cpu()
            labels = target.numpy()[:remaining]
            assert not torch.isnan(embeddings).any()

        all_embeddings.append(embeddings)
        all_labels.append(labels)

    asset_dict = {
        "embeddings": np.vstack(all_embeddings).astype(np.float32),
        "labels": np.concatenate(all_labels),
    }

    return asset_dict

@torch.no_grad()
def extract_patch_features_from_dataloader_dist(model, dataloader):
    """Uses model to extract features+labels from images iterated over the dataloader.

    Args:
        model (torch.nn): torch.nn CNN/VIT architecture with pretrained weights that extracts d-dim features.
        dataloader (torch.utils.data.DataLoader): torch.utils.data.DataLoader object of N images.

    Returns:
        return embeddings, labels in tensors on GPU for dist
    """
    all_embeddings, all_labels = [], []
    batch_size = dataloader.batch_size
    device = next(model.parameters())[0].device

    for batch_idx, (batch, target) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        remaining = batch.shape[0]
        if remaining != batch_size:
            _ = torch.zeros((batch_size - remaining,) + batch.shape[1:]).type(
                batch.type()
            )
            batch = torch.vstack([batch, _])

        batch = batch.to(device)
        with torch.inference_mode():
            embeddings = model(batch)[:remaining, :]
            labels = target[:remaining]
            assert not torch.isnan(embeddings).any()

        all_embeddings.append(embeddings)
        all_labels.append(labels)

    asset_dict = {
        "embeddings": torch.vstack(all_embeddings).double().detach().cpu(),
        "labels": torch.cat(all_labels).long().detach().cpu(),
    }

    return asset_dict

@torch.no_grad()
def extract_patch_features_from_slide_dist(model, dataloader):
    """Uses model to extract features+labels from images iterated over the dataloader.

    Args:
        model (torch.nn): torch.nn CNN/VIT architecture with pretrained weights that extracts d-dim features.
        dataloader (torch.utils.data.DataLoader): torch.utils.data.DataLoader object of N images.

    Returns:
        return embeddings, labels in tensors on GPU for dist
    """
    all_embeddings, all_labels = [], []
    batch_size = dataloader.batch_size
    device = next(model.parameters())[0].device

    for batch_idx, (batch, target) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        remaining = batch.shape[0]
        if remaining != batch_size:
            _ = torch.zeros((batch_size - remaining,) + batch.shape[1:]).type(
                batch.type()
            )
            batch = torch.vstack([batch, _])

        batch = batch.to(device)
        with torch.inference_mode():
            embeddings = model(batch)[:remaining, :]
            labels = target[:remaining]
            assert not torch.isnan(embeddings).any()

        all_embeddings.append(embeddings)
        all_labels.append(labels)
        # del embeddings
        # del labels

    asset_dict = {
        "embeddings": torch.vstack(all_embeddings).double().detach().cpu(),
        "idx": torch.cat(all_labels).long().detach().cpu(),
    }

    return asset_dict






def get_dino_finetuned_downloaded(DINO_PATH_FINETUNED_DOWNLOADED=None):
    # load the original DINOv2 model with the correct architecture and parameters. The positional embedding is too large.
    # load vits or vitg
    #model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    #model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    # load finetuned weights
    pretrained = torch.load(DINO_PATH_FINETUNED_DOWNLOADED, map_location=torch.device('cpu'))
    #print(model)
    # make correct state dict for loading
    new_state_dict = {}
    for key, value in pretrained['teacher'].items():
        if 'dino_head' in key:
            print('not used')
        else:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value
    #print(new_state_dict.keys())

    # use training method
    # input_tensor = model.pos_embed
    # tensor_corr_shape = interpolate_pos_encoding(input_tensor, 16, 16)
    # pos_embed = nn.Parameter(torch.zeros(1, 257))
    # pos_embed.data = tensor_corr_shape
    # model.pos_embed = pos_embed
    # # load state dict
    # model.load_state_dict(pretrained, strict=True)
    #print(model.pos_embed.shape)

    # load classical method
    pos_embed = nn.Parameter(torch.zeros(1, 257, 1536))
    model.pos_embed = pos_embed
    # load state dict
    model.load_state_dict(new_state_dict, strict=True)
    return model

def get_dino_finetuned_downloaded_dist(DINO_PATH_FINETUNED_DOWNLOADED=None,rank=0):
    # load the original DINOv2 model with the correct architecture and parameters. The positional embedding is too large.
    # load vits or vitg
    #model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    #model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
    model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

    loc = 'cuda:{}'.format(rank)
    model.to(loc)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # load finetuned weights
    pretrained = torch.load(DINO_PATH_FINETUNED_DOWNLOADED, map_location=loc)
    #print(pretrained)
    # make correct state dict for loading
    new_state_dict = {}
    for key, value in pretrained['teacher'].items():
        if 'dino_head' in key:
            print('not used')
        else:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value
    #print(new_state_dict.keys())

    # use training method
    # input_tensor = model.pos_embed
    # tensor_corr_shape = interpolate_pos_encoding(input_tensor, 16, 16)
    # pos_embed = nn.Parameter(torch.zeros(1, 257))
    # pos_embed.data = tensor_corr_shape
    # model.pos_embed = pos_embed
    # # load state dict
    # model.load_state_dict(pretrained, strict=True)

    # load classical method
    pos_embed = nn.Parameter(torch.zeros(1, 257, 1536))
    model.pos_embed = pos_embed
    # load state dict
    model.load_state_dict(new_state_dict, strict=False)
    return model

def get_UNI_downloaded_dist(UNI_dir=None,rank=0):
    # load the original DINOv2 model with the correct architecture and parameters. The positional embedding is too large.
    # load vits or vitg
    #model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    #model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
    import timm
    model = timm.create_model("hf_hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)

    loc = 'cuda:{}'.format(rank)
    model.to(loc)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])


    #print(new_state_dict.keys())

    # use training method
    # input_tensor = model.pos_embed
    # tensor_corr_shape = interpolate_pos_encoding(input_tensor, 16, 16)
    # pos_embed = nn.Parameter(torch.zeros(1, 257))
    # pos_embed.data = tensor_corr_shape
    # model.pos_embed = pos_embed
    # # load state dict
    # model.load_state_dict(pretrained, strict=True)

    # load classical method
    # pos_embed = nn.Parameter(torch.zeros(1, 257, 1536))
    # model.pos_embed = pos_embed
    # # load state dict
    # model.load_state_dict(new_state_dict, strict=False)
    return model



#import vits_moco as vits
import os
#import moco.builder_inference
from functools import partial
def get_moco_finetuned_downloaded(moco_path=None):

    arch = 'vit_huge'
    model = moco.builder_inference.MoCo_ViT(
        partial(vits.__dict__[arch], stop_grad_conv1=True))

    if os.path.isfile(moco_path):
        print("=> loading checkpoint '{}'".format(moco_path))
        checkpoint = torch.load(moco_path, map_location="cpu")
        linear_keyword = 'head'

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        # del model.pre_logits
        # del model.head
        #print(state_dict.keys())
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                # remove prefix
                state_dict[k[len("module."):]] = state_dict[k]
                print(k[len("module."):])
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        #print(msg)
        #assert set(msg.missing_keys) == {"%s.0.weight" % linear_keyword, "%s.0.bias" % linear_keyword}

        print("=> loaded pre-trained model '{}'".format(moco_path))

    else:
        print("=> no checkpoint found at '{}'".format(moco_path))
    #print(model)
    return model

def get_moco_finetuned_downloaded_dist(moco_path=None,rank=0):

    arch = 'vit_huge'
    model = moco.builder_inference.MoCo_ViT(
        partial(vits.__dict__[arch], stop_grad_conv1=True))
    loc = 'cuda:{}'.format(rank)
    model.to(loc)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    if os.path.isfile(moco_path):
        print("=> loading checkpoint '{}'".format(moco_path))
        checkpoint = torch.load(moco_path, map_location=loc)
        linear_keyword = 'head'

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        # del model.pre_logits
        # del model.head
        #print(state_dict.keys())
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                # remove prefix
                state_dict[k[len("module."):]] = state_dict[k]
                print(k[len("module."):])
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        #print(msg)
        #assert set(msg.missing_keys) == {"%s.0.weight" % linear_keyword, "%s.0.bias" % linear_keyword}

        print("=> loaded pre-trained model '{}'".format(moco_path))

    else:
        print("=> no checkpoint found at '{}'".format(moco_path))
    #print(model)
    return model




#model=get_dino_finetuned_downloaded()
