import torch
#define random input tensors with shape [3,8]
input = torch.rand(2, 10).cuda()
print(input)
pdist = torch.nn.PairwiseDistance(2, eps=1e-8)
#pdist1 = pdist(input,input)
pdist2 = torch.pdist(input)

#print(pdist1)
print(pdist2)
