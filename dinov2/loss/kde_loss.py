# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# this file was changed, the eps in the pairwisedistance

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

# import torch.distributed as dist


logger = logging.getLogger("dinov2")


class KDELoss(nn.Module): #para: 5, weight: 0.05
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self,t=5):
        super().__init__()
        self.t = t
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    # def pairwise_NNs_inner(self, x):
    #     """
    #     Pairwise nearest neighbors for L2-normalized vectors.
    #     Uses Torch rather than Faiss to remain on GPU.
    #     """
    #     # parwise dot products (= inverse distance)
    #     dots = torch.mm(x, x.t())
    #     n = x.shape[0]
    #     dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
    #     # max inner prod -> min distance
    #     _, I = torch.max(dots, dim=1)  # noqa: E741
    #     return I

    def forward(self, student_output, t=5):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        # student_output = F.normalize(student_output, p=2, dim=1)
        #before computation, convert student_output from tensor_typr fp16 to fp32
        #student_output_fp32 = student_output.float()
        #print(student_output.shape)
        #normalization
        student_output = F.normalize(student_output, p=2, dim=1)
        student_output_fp32 = student_output.float()
        loss_32 = torch.pdist(student_output_fp32).pow(2).mul(-t).exp().mean().log()
        loss = loss_32.half()
        #print(torch.pdist(student_output))
        return loss
