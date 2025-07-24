# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# this file was changed

import sys
import os

# Add the root directory of the project to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import argparse
import logging
import math
import os
from functools import partial
import wandb
#os.environ["WANDB_MODE"]="offline" #use this so set wandb to offline mode
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from fvcore.common.checkpoint import PeriodicCheckpointer
import torch

from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler
import torch.distributed as dist

from dinov2.train.ssl_meta_arch import SSLMetaArch
os.environ["NCCL_DEBUG"] = "INFO" #INFO
#os.environ["NCCL_SOCKET_IFNAME"] = "ibp170s0f0"
#os.environ["GLOO_SOCKET_IFNAME"] = "ibp170s0f0"


torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
torch.backends.cudnn.benchmark = True
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="/home/ge24juj/dino-tum/dinov2/configs/ssl_default_config.yaml", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg,start_iter):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH+start_iter,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH+start_iter,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH+start_iter,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH+start_iter,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def do_test(cfg, model, iteration):
    dist.barrier()
    with FSDP.state_dict_type(model.teacher, StateDictType.FULL_STATE_DICT):
        new_state_dict = model.teacher.state_dict()
        # new_state_dict_student = model.student.state_dict()
        # state_dict_teacher_dino_head = model.teacher.dino_head.state_dict()
        # state_dict_student_dino_head = model.student.dino_head.state_dict()

    iterstring = str(iteration)
    eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
    if distributed.is_main_process():
        #print(eval_dir)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)
        # save student checkpoint
        # student_ckp_path = os.path.join(eval_dir, "student_checkpoint.pth")
        # torch.save({"student": new_state_dict_student}, student_ckp_path)
        #
        # # Save state_dict_teacher_dino_head for the teacher model
        # teacher_dino_head_ckp_path = os.path.join(eval_dir, "teacher_dino_head_checkpoint.pth")
        # torch.save({"teacher_dino_head": state_dict_teacher_dino_head}, teacher_dino_head_ckp_path)
        #
        # # Save state_dict_student_dino_head for the student model
        # student_dino_head_ckp_path = os.path.join(eval_dir, "student_dino_head_checkpoint.pth")
        # torch.save({"student_dino_head": state_dict_student_dino_head}, student_dino_head_ckp_path)

    #test the teacher model with downstream tasks
    print('test the teacher model')
    from dinov2.train.eval.eval_during_training import inf_during_training
    patch_metric = inf_during_training(variant=cfg.student.arch,ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth"),local_id=dist.get_rank(),iter=iteration)

    return patch_metric


def do_train(cfg, model, resume=False,args=None): # change resume to true?
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    # setup optimizer
    pretrained_iter = cfg.train.pretrained_iter #41000

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg,pretrained_iter)

    # checkpointer
    #with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        #checkpointer.resume_or_load()
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        checkpointer = FSDPCheckpointer(model, '/home/ge24juj/dino-tum/dinov2/ckp/', optimizer=optimizer, save_to_disk=True)

        start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1 #if use pretrained, resume=False
    print('start_iter: '+str(start_iter))
    #start_iter = 41000
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH + pretrained_iter
    print('max_iter: '+str(max_iter))

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=10 * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # setup data preprocessing

    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    data_transform = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # setup data loader

    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        transform=data_transform,
        target_transform=lambda _: (),
    )
    # for one GPU use this, for several switch to sampler_type = SamplerType.SHARDED_INFINITE
    #sampler_type = SamplerType.INFINITE
    #sampler_type = SamplerType.SHARDED_INFINITE
    #sampler_type = cfg.train.sampler_type
    #sampler_type = SamplerType.EPOCH
    if cfg.train.sampler_type == 0:
        sampler_type = SamplerType.DISTRIBUTED
    elif cfg.train.sampler_type == 1:
        sampler_type = SamplerType.EPOCH
    elif cfg.train.sampler_type == 2:
        sampler_type = SamplerType.INFINITE
    elif cfg.train.sampler_type == 3:
        sampler_type = SamplerType.SHARDED_INFINITE
    elif cfg.train.sampler_type == 4:
        sampler_type = SamplerType.SHARDED_INFINITE_NEW
    elif cfg.train.sampler_type == 5:
        sampler_type = SamplerType.TUM_DistributedSampler

    iteration = start_iter + pretrained_iter  #if use pretrained, then + pretrained_iter. If resume, then + 0
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=start_iter * cfg.train.batch_size_per_gpu,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )
    if not args.debug:
        if distributed.is_main_process():
            run = wandb.init(
            # Set the project where this run will be logged
            project="dino_training",
            name='vit_large_100k')


    # training loop
    # change this value manually, if continuing with earlier weights to keep the progress of the schedulers

    #iteration = 0

    logger.info("Starting training from iteration {}".format(iteration))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    from torch.profiler import profile, record_function, ProfilerActivity
    patch_metric = {'lin_acc': 0.0, 'lin_bacc': 0.0}
    if iteration == 0:
        patch_metric = do_test(cfg, model, f"training_{iteration}")

    if 'TUMShardDataset' in cfg.train.dataset_path:
        dataset.set_epoch(0)
    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        max_iter -pretrained_iter, #if use pretrained, then - pretrained_iter. If resume, then - 0
        start_iter,
    ):
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     with record_function("Training Step"):
                #print(iteration)
                #print(cfg.evaluation.eval_period_iterations)
                model.train()
                current_batch_size = data["collated_global_crops"].shape[0] / 2
                #logger.info("Batch size: {}".format(current_batch_size))
                if iteration > max_iter:
                    return

                # apply schedules

                lr = lr_schedule[iteration]
                wd = wd_schedule[iteration]
                mom = momentum_schedule[iteration]
                teacher_temp = teacher_temp_schedule[iteration]
                last_layer_lr = last_layer_lr_schedule[iteration]
                apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

                # compute losses

                optimizer.zero_grad(set_to_none=True)
                loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

                # clip gradients

                if fp16_scaler is not None:
                    if cfg.optim.clip_grad:
                        fp16_scaler.unscale_(optimizer)
                        for v in model.student.values():
                            v.clip_grad_norm_(cfg.optim.clip_grad)
                    fp16_scaler.step(optimizer)
                    fp16_scaler.update()
                else:
                    if cfg.optim.clip_grad:
                        for v in model.student.values():
                            v.clip_grad_norm_(cfg.optim.clip_grad)
                    optimizer.step()

                # perform teacher EMA update
                model.update_teacher(mom)

                # logging
                if distributed.get_global_size() > 1:
                    for v in loss_dict.values():
                        torch.distributed.all_reduce(v)
                loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}
                #print(str(distributed.get_global_size()))

                if math.isnan(sum(loss_dict_reduced.values())):
                    logger.info("NaN detected")
                    raise AssertionError
                losses_reduced = sum(loss for loss in loss_dict_reduced.values()) # removed the keleo regularizer, because it is often inf
                #losses_reduced = sum(loss for key, loss in loss_dict_reduced.items() if key != 'koleo_loss')

                kde_loss = loss_dict_reduced['kde_loss'] if 'kde_loss' in loss_dict_reduced.keys() else 'N/A'
                koleo_loss = loss_dict_reduced['koleo_loss'] if 'koleo_loss' in loss_dict_reduced.keys() else 'N/A'
                # print(koleo_loss)
                # print(loss_dict_reduced['dino_global_crops_loss'])
                # wandb logging
                if not args.debug:
                    if distributed.is_main_process():
                        wandb.log({"lr": lr, "loss": losses_reduced, "wd": wd, "mom": mom, "last_layer_lr": last_layer_lr, "current_batch_size": current_batch_size
                        , "koleo_loss": koleo_loss,"kde_loss":kde_loss, "dino_local_crops_loss": loss_dict_reduced['dino_local_crops_loss']
                        , "dino_global_crops_loss": loss_dict_reduced['dino_global_crops_loss'], "ibot_loss": loss_dict_reduced['ibot_loss'],"patch_acc":patch_metric['lin_acc'], "patch_bacc": patch_metric['lin_bacc']})
                metric_logger.update(lr=lr)
                metric_logger.update(wd=wd)
                metric_logger.update(mom=mom)
                metric_logger.update(last_layer_lr=last_layer_lr)
                metric_logger.update(current_batch_size=current_batch_size)
                metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

                # checkpointing and testing
                if cfg.evaluation.eval_period_iterations > 0 and (iteration+1) % cfg.evaluation.eval_period_iterations == 0:
                    patch_metric = do_test(cfg, model, f"training_{iteration}")
                    #do_test(cfg, model, f"training_{iteration}")
                    metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)
                    torch.cuda.synchronize()

                periodic_checkpointer.step(iteration)

                iteration = iteration + 1

        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # print(kde_loss)
    # print(loss_dict_reduced['dino_global_crops_loss'])
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    print('prepare for distributed training')
    model.prepare_for_distributed_training()

    #logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir='/home/ge24juj/dino-tum/dinov2/ckp/')
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    do_train(cfg, model, resume=not args.no_resume,args=args)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
