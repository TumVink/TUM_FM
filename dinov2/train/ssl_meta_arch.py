# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# this file was changed

from functools import partial
import logging
import os

import torch
from torch import nn

from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss, KDELoss
from dinov2.models import build_model_from_cfg
from dinov2.layers import DINOHead
from dinov2.utils.utils import has_batchnorms
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model

from dinov2.models.vision_transformer import BlockChunk

try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")

logger = logging.getLogger("dinov2")

import math
import torch.nn.functional as nnf


# this is an adapted version of the interpolate_pos_encoding in the vision_transformer.py to change the shape of the positional encoding
def interpolate_pos_encoding(x, w, h):
    N = x.shape[1] - 1
    dim = x.shape[-1]
    w0 = w / int(math.sqrt(N))
    h0 = h / int(math.sqrt(N))

    # Interpolate the position embeddings without changing the first row (class token)
    patch_pos_embed = nnf.interpolate(
        x[:, 1:].reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0, h0),
        mode="bicubic",
    )

    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    # Concatenate the class token with the interpolated position embeddings
    return torch.cat((x[:, :1], patch_pos_embed), dim=1)


# this function returns a vit_s model with smaller positional embedding (shape 224x224). T
# The reduced size happens through cutting the middle part out
def get_downloaded_dino_vit_s():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    input_tensor = model.pos_embed
    class_token = input_tensor[:, 0:1, :]
    rest = input_tensor[:, 1:, :]

    reshaped_tensor = rest.view(1, 37, 37, 384)

    middle = 18
    middle_start = middle - 8
    middle_end = middle + 8
    middle_part = reshaped_tensor[:, middle_start:middle_end, middle_start:middle_end, :]
    flattened_tensor = middle_part.reshape(1, 256, 384)

    tensor_corr_shape = torch.cat((class_token, flattened_tensor), dim=1)

    pos_embed = nn.Parameter(torch.zeros(1, 257))
    pos_embed.data = tensor_corr_shape

    model.pos_embed = pos_embed

    return model


# this function returns either a vit_s or vit_g, depending on what is commented out. the model is loaded with from torch.hub wih the weights and the positional encoding is reshaped.
def get_downloaded_dino_interpolated(cfg):
    if cfg.student.arch == 'vit_giant2':
        # this is a workaround to load the giant model with registers
        print('loading giant with registers')
        return get_downloaded_dino_reg_interpolated()
    if 'large' in cfg.student.arch:
        print('-------- vit large loaded -------------')
        model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    elif 'giant' in cfg.student.arch:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    else:
        print('------------ vit small loaded -----------')
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    # model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    input_tensor = model.pos_embed
    tensor_corr_shape = interpolate_pos_encoding(input_tensor, 16, 16)
    pos_embed = nn.Parameter(torch.zeros(1, 257))
    pos_embed.data = tensor_corr_shape
    model.pos_embed = pos_embed

    return model


# this function is exactly the same as the one above, but with registers
def get_downloaded_dino_reg_interpolated():
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
    input_tensor = model.pos_embed
    tensor_corr_shape = interpolate_pos_encoding(input_tensor, 16, 16)
    pos_embed = nn.Parameter(torch.zeros(1, 257))
    pos_embed.data = tensor_corr_shape
    model.pos_embed = pos_embed

    return model


# This function allows to continue finetuning in case of the training breaking or also just if more experiments should be conducted.
# Both teacher and student are set here directly. It is easy to switch between vit_s and vit_g.
def get_dino_finetuned_downloaded(cfg, embed_dim):
    if 'large' in cfg.student.arch:
        model_student = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        model_teacher = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    elif 'giant' in cfg.student.arch:
        model_student = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        model_teacher = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    else:
        model_student = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        model_teacher = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    logger.info(f"finish loading")
    # load finetuned weights, here the path added to the config_files is used
    path_student = os.path.join('/home/ge24juj/dino-tum/weights',
                                'student_checkpoint.pth')  # os.path.join(cfg.head.head_path, 'student_checkpoint.pth')
    path_teacher = os.path.join('/home/ge24juj/dino-tum/weights',
                                'teacher_checkpoint.pth')  # os.path.join(cfg.head.head_path, 'teacher_checkpoint.pth')
    pretrained_student = torch.load(path_student, map_location=torch.device('cpu'))
    pretrained_teacher = torch.load(path_teacher, map_location=torch.device('cpu'))
    # logger.info(f"finish loading")
    # create correct state dict for loading (the name teacher has to be removed)
    teacher_state_dict = {}
    for key, value in pretrained_teacher['teacher'].items():
        if 'dino_head' in key:
            print('dino_head not used from backbone')
        else:
            new_key = key.replace('backbone.', '')
            teacher_state_dict[new_key] = value
    # create correct state dict for loading (the name student has to be removed)
    student_state_dict = {}
    for key, value in pretrained_student['student'].items():
        if 'dino_head' in key:
            print('dino_head not used from backbone')
        else:
            new_key = key.replace('backbone.', '')
            student_state_dict[new_key] = value
    # change shape of pos_embed
    pos_embed1 = nn.Parameter(torch.zeros(1, 257, embed_dim))
    pos_embed2 = nn.Parameter(torch.zeros(1, 257, embed_dim))
    model_student.pos_embed = pos_embed1
    model_teacher.pos_embed = pos_embed2
    # load state dict of with strict=true to make sure everything works corectly
    model_student.load_state_dict(student_state_dict, strict=True)
    model_teacher.load_state_dict(teacher_state_dict, strict=True)
    return model_student, model_teacher


class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()
        teacher_model_dict = dict()

        # This is commented out, because it was easier to create the model using the torch.hub, as this already returns the pretrained version with the correct architecture.
        # student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        if 'large' in cfg.student.arch:
            embed_dim = 1024
        elif 'giant' in cfg.student.arch:
            embed_dim = 1536
        else:
            embed_dim = 384
        #embed_dim = 1536  # use for vit_g
        # embed_dim = 384 # use for vit_s
        # embed_dim = 1024  # use for vit_l

        # use for cut loading downloaded weights
        '''
        student_backbone = get_downloaded_dino_vit_s()
        teacher_backbone = get_downloaded_dino_vit_s()
        '''


        if cfg.MODEL.Pretrained == 'Meta':
            # use for interpolated loading downloaded weights
            student_backbone = get_downloaded_dino_interpolated(cfg)
            teacher_backbone = get_downloaded_dino_interpolated(cfg)

            # use for interpolated loading downloaded weights with register

            # student_backbone = get_downloaded_dino_reg_interpolated()
            # teacher_backbone = get_downloaded_dino_reg_interpolated()

        # use for continuation with finetuned weights, in this case the dino head weights also have to be set
        elif cfg.MODEL.Pretrained == 'Helmholtz':
            student_backbone, teacher_backbone = get_dino_finetuned_downloaded(cfg, embed_dim)
        else:
            raise NotImplementedError

        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        # this is no longer required to load the weights, the functions above are used for this purpose
        # if cfg.student.pretrained_weights:
        #     chkpt = torch.load(cfg.student.pretrained_weights)
        #     logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")
        #     student_model_dict["backbone"] = student_backbone

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_kde = cfg.dino.kde_loss_weight > 0
        #assert self.do_kde != self.do_koleo, "KDE and KoLeo can't be enabled together"
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head

        logger.info("OPTIONS -- DINO")
        if self.do_dino:
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
            logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
            logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                self.koleo_loss = KoLeoLoss()
            elif self.do_kde:
                self.kde_loss = KDELoss()
                logger.info("OPTIONS -- DINO -- applying KDE regularization")


        else:
            logger.info("OPTIONS -- DINO -- not using DINO")

        # comment out to not use dino_head
        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head() #randomly init dino head if not specified
            teacher_model_dict["dino_head"] = dino_head()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")
        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert max(cfg.ibot.mask_ratio_min_max) > 0, "please provide a positive mask ratio tuple for ibot"
            assert cfg.ibot.mask_sample_probability > 0, "please provide a positive mask probability for ibot"
            self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else cfg.dino.head_n_prototypes
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
                logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
                logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
                logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
                ibot_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()
            else:
                logger.info("OPTIONS -- IBOT -- head shared with DINO")

        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # load dino_head weights if available
        if cfg.head.head_path:
            #print('loading dino_head')
            path_student = os.path.join(cfg.head.head_path, 'student_dino_head_checkpoint.pth')
            path_teacher = os.path.join(cfg.head.head_path, 'teacher_dino_head_checkpoint.pth')
            chkpt_teacher = torch.load(path_teacher)
            chkpt_student = torch.load(path_student)
            print('loading dino_head')
            student_model_dict["dino_head"].load_state_dict(chkpt_student['student_dino_head'], strict=True)
            teacher_model_dict["dino_head"].load_state_dict(chkpt_teacher['teacher_dino_head'], strict=True)

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        # for student activate backprop
        for p in self.student.parameters():
            p.requires_grad = True
        # disable backpropagation for student.backbone, this was tested to only train the dino_head without the backbone
        '''
        for p in self.student.backbone.parameters():
            p.requires_grad = False
        '''

        # for debugging, verify what is getting tracked
        # for name, param in self.student.named_parameters():
        #    if 'dino_head' in name:
        #        print(f'Parameter: {name}, Requires Grad: {param.requires_grad}')

        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, images, teacher_temp):
        n_global_crops = 2
        assert n_global_crops == 2
        n_local_crops = self.cfg.crops.local_crops_number

        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        local_crops = images["collated_local_crops"].cuda(non_blocking=True)

        masks = images["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)
        n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].cuda(non_blocking=True)

        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        do_dino = self.do_dino
        do_ibot = self.do_ibot

        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops

        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            x, n_global_crops_teacher = global_crops, n_global_crops
            teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
            # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
            teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
            ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
            _dim = ibot_teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]

            if do_ibot and not self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[n_cls_tokens: n_cls_tokens + n_masked_patches],
                )
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                # tokens_after_head = buffer_tensor_teacher # tested training without dino_head
                teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                masked_teacher_patch_tokens_after_head = tokens_after_head[
                                                         n_cls_tokens: n_cls_tokens + n_masked_patches
                                                         ]
            elif do_ibot and self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                # teacher_cls_tokens_after_head = teacher_cls_tokens # tested training without dino_head
                masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[
                                                         :n_masked_patches
                                                         ]
            else:
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                # teacher_cls_tokens_after_head = teacher_cls_tokens # tested training without dino_head
                masked_teacher_ibot_softmaxed_centered = None

            if self.cfg.train.centering == "centering":
                teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                if do_ibot:
                    masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
                    )
                    masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
                    self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])

            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])

                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        masked_teacher_patch_tokens_after_head,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor,
                    )

            else:
                raise NotImplementedError

            return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered

        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = get_teacher_output()
        reshard_fsdp_model(self.teacher)

        loss_dict = {}

        loss_accumulator = 0  # for backprop
        student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.backbone(
            [global_crops, local_crops], masks=[masks, None], is_training=True
        )

        inputs_for_student_head_list = []

        # 1a: local crops cls tokens
        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        # 1c: global crops patch tokens
        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            )
            if not self.ibot_separate_head:
                inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)[
                                                                :n_masked_patches
                                                                ]

        # 2: run
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))
        # outputs_list = _attn_bias.split(cat_inputs) # tested training without dino_head

        # 3a: local crops cls tokens
        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3b: global crops cls tokens
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3c: global crops patch tokens
        if do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]

        if n_local_crops > 0:
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            # store for display
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

        # process global crops
        loss_scales = 2  # this is here since we process global crops together

        if do_dino:
            # compute loss
            dino_global_crops_loss = (
                    self.dino_loss(
                        student_output_list=[student_global_cls_tokens_after_head],
                        teacher_out_softmaxed_centered_list=[
                            teacher_dino_softmaxed_centered_list.flatten(0, 1)
                        ],  # these were chunked and stacked in reverse so A is matched to B
                    )
                    * loss_scales
                    / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

            student_cls_tokens = student_global_cls_tokens

            if self.do_koleo:
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = (
                        koleo_loss / loss_scales
                )  # this is to display the same losses as before but we can remove eventually
                #print(loss_dict["koleo_loss"])
            elif self.do_kde:
                kde_loss = self.cfg.dino.kde_loss_weight * sum(
                    self.kde_loss(p) for p in student_cls_tokens.chunk(2)
                )
                loss_accumulator += kde_loss
                loss_dict["kde_loss"] = kde_loss / loss_scales


        if do_ibot:
            # compute loss
            ibot_patch_loss = (
                    self.ibot_patch_loss.forward_masked(
                        student_global_masked_patch_tokens_after_head,
                        masked_teacher_ibot_softmaxed_centered,
                        student_masks_flat=masks,
                        n_masked_patches=n_masked_patches,
                        masks_weight=masks_weight,
                    )
                    * loss_scales
                    * ibot_loss_scale
            )

            # store for display
            loss_dict["ibot_loss"] = ibot_patch_loss / 2

            # accumulate loss
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss
            #print(loss_dict["ibot_loss"])

        self.backprop_loss(loss_accumulator)

        self.fsdp_synchronize_streams()

        return loss_dict

    # use this to update everything as originally done
    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()

            for attr in {"_unshard_stream", "_post_backward_stream", "_pre_unshard_stream", "_all_reduce_stream", "_default_stream"}:
                stream = getattr(self.teacher.backbone, attr)
                setattr(self.student.dino_head, attr, stream)
                setattr(self.teacher.dino_head, attr, stream)
                setattr(self.student.backbone, attr, stream)

            self.need_to_synchronize_fsdp_streams = False

    # this was used for training the din_head without touhing the backbone
    '''
    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            self.student.backbone._streams = self.teacher.backbone._streams
            self.need_to_synchronize_fsdp_streams = False
    '''

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                # use to only update dino_head
                '''
                if k != 'backbone':
                    for ms, mt in zip(get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])):
                        student_param_list += ms.params
                        teacher_param_list += mt.params
                '''
                for ms, mt in zip(get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])):
                    student_param_list += ms.params
                    teacher_param_list += mt.params
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self):
        super().train()
        self.teacher.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        print('DISTRIBUTED FSDP -- preparing model for distributed training')
        if has_batchnorms(self.student):
            raise NotImplementedError
        # below will synchronize all student subnetworks across gpus:
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
            #print('running 1')
            teacher_model_cfg = self.cfg.compute_precision.teacher[k]
            self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(self.teacher[k])
            #print('running 2')
