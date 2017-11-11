import sys
sys.path.append("..")

import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from model_defs import network

def select_to_match_dimensions(a,b):
        if a.size()[2] > b.size()[2]:
            a = torch.index_select(a, 2,
                                  network.np_to_variable(np.arange(0,
                                        b.size()[2]).astype(np.int32),
                                         is_cuda=True,dtype=torch.LongTensor))
        if a.size()[3] > b.size()[3]:
            a = torch.index_select(a, 3,
                                  network.np_to_variable(np.arange(0,
                                    b.size()[3]).astype(np.int32),
                                          is_cuda=True,dtype=torch.LongTensor))

        return a


#
#
# Random input data for tests
#
#

groups = 4

img_data = np.random.rand(3, groups, 20, 27).astype('float32')
target_data = np.random.rand(6, groups, 6, 6).astype('float32')

img_features = Variable(torch.from_numpy(img_data)).cuda()
target_features = Variable(torch.from_numpy(target_data)).cuda()


#
#
# Old version
#
#
#

ccs = []
diffs = []

padding = (max(0,int(target_features.size()[2]/2)),
           max(0,int(target_features.size()[3]/2)))
for b_ind in range(img_features.size()[0]):
    target_inds = network.np_to_variable(np.asarray([b_ind*2, b_ind*2+1]),
                                        is_cuda=True, dtype=torch.LongTensor)
    sample_targets1 = torch.index_select(target_features,0,target_inds[0])
    sample_targets2 = torch.index_select(target_features,0,target_inds[1])
    img_ind = network.np_to_variable(np.asarray([b_ind]),
                                        is_cuda=True, dtype=torch.LongTensor)
    sample_img = torch.index_select(img_features,0,img_ind)

    sample_targets1 = sample_targets1.view(-1,1,sample_targets1.size()[2],
                                           sample_targets1.size()[3])
    sample_targets2 = sample_targets2.view(-1,1,sample_targets2.size()[2],
                                           sample_targets2.size()[3])


    #get diff
    tf1_pooled = F.max_pool2d(sample_targets1,(sample_targets1.size()[2],
                                               sample_targets1.size()[3]))
    tf2_pooled = F.max_pool2d(sample_targets2,(sample_targets2.size()[2],
                                               sample_targets2.size()[3]))

    diff1 = sample_img - tf1_pooled.permute(1,0,2,3).expand_as(sample_img)
    diff2 = sample_img - tf2_pooled.permute(1,0,2,3).expand_as(sample_img)
    diffs.append(torch.cat([diff1,diff2],1))

    #do cross corr      
    cc1 = F.conv2d(sample_img,sample_targets1,padding=padding,groups=groups)
    cc2 = F.conv2d(sample_img,sample_targets2,padding=padding,groups=groups)
    cc = torch.cat([cc1,cc2],1)
    cc = select_to_match_dimensions(cc,sample_img)
    ccs.append(cc)
    
cc_old = torch.cat(ccs,0)
diffs_old = torch.cat(diffs,0)

#
#
# New version
#
#

batchsize, channels, H, W = img_features.size()
_, _, h, w = target_features.size()

padding = (max(0,int(h/2)),
           max(0,int(w/2)))

targets1 = target_features[::2].contiguous()
targets2 = target_features[1::2].contiguous()

tf1_pooled = F.max_pool2d(targets1, (h, w))
tf2_pooled = F.max_pool2d(targets2, (h, w))

diff1 = img_features - tf1_pooled.expand_as(img_features)
diff2 = img_features - tf2_pooled.expand_as(img_features)

diffs = torch.cat([diff1, diff2], 1)

cc1 = F.conv2d(
    img_features.view(1, -1, H, W),
    targets1.view(-1, 1, h, w),
    padding=padding,
    groups=batchsize * groups,
)
cc1 = cc1[:, :, :H, :W].contiguous().view(batchsize, channels, H, W)

cc2 = F.conv2d(
    img_features.view(1, -1, H, W),
    targets2.view(-1, 1, h, w),
    padding=padding,
    groups=batchsize * groups,
)
cc2 = cc2[:, :, :H, :W].contiguous().view(batchsize, channels, H, W)
cc = torch.cat([cc1, cc2], dim=1)


#
#
# Comparison
#
#

print("CC tensors equal:", np.allclose(cc.data.cpu().numpy(), cc_old.data.cpu().numpy()))
print("Diffs tensors equal:", np.allclose(diffs.data.cpu().numpy(), diffs_old.data.cpu().numpy()))


