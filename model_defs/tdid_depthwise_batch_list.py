import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.timer import Timer
#from rpn_msr.tdid_proposal_layer import proposal_layer as proposal_layer_py
#from rpn_msr.tdid_proposal_layer_batch import proposal_layer as proposal_layer_py
from rpn_msr.tdid_proposal_layer_batch_TODO import proposal_layer as proposal_layer_py
from rpn_msr.anchor_target_layer_batch import anchor_target_layer as anchor_target_layer_py

import network
from network import Conv2d, FC
from vgg16 import VGG16




class TDID(nn.Module):
    _feat_stride = [16, ]
    #anchor_scales = [8, 16, 32]
    anchor_scales = [2, 4, 8]
    groups=512

    def __init__(self):
        super(TDID, self).__init__()

        #first 5 conv layers of VGG? only resizing is 4 max pools
        self.features = VGG16(bn=False)

        self.conv1 = Conv2d(2*self.groups,512, 3, relu=False, same_padding=True)
        #self.conv2 = Conv2d(512,512, 3, relu=False, same_padding=True)
        self.score_conv = Conv2d(512, len(self.anchor_scales) * 3 * 2, 1, relu=False, same_padding=False)
        self.bbox_conv = Conv2d(512, len(self.anchor_scales) * 3 * 4, 1, relu=False, same_padding=False)

        # loss
        self.cross_entropy = None
        self.loss_box = None

    @property
    def loss(self):
        #return self.roi_cross_entropy
        return self.roi_cross_entropy + self.cross_entropy + self.loss_box * 10
        #return self.cross_entropy + self.loss_box * 10

    def forward(self, target_data, im_data, gt_boxes=None):

        im_info = im_data.shape[1:]   
      
        #get image features 
        im_data = network.np_to_variable(im_data, is_cuda=True)
        im_data = im_data.permute(0, 3, 1, 2)
        img_features = self.features(im_data)
       

 
        target_data = network.np_to_variable(target_data, is_cuda=True)
        target_data = target_data.permute(0, 3, 1, 2)
        target_features = self.features(target_data)
        #reshape target from 2xCxHxW to 2Cx1xHxW for cross corr
        target_features = target_features.view(-1,1,target_features.size()[2], 
                                               target_features.size()[3])
    
        #get cross correlation of each target's features with image features
        #(same padding)
        padding = (max(0,int(target_features.size()[2]/2)), 
                         max(0,int(target_features.size()[3]/2)))
        cc = F.conv2d(img_features,target_features,
                      padding=padding, groups=self.groups)

        #cc = self.select_to_match_dimensions(cc,img_features)


        
        rpn_conv1 = self.conv1(cc)
        #rpn_conv2 = self.conv2(img_features)

 
        # rpn score
        rpn_cls_score = self.score_conv(rpn_conv1)
        rpn_cls_score_reshape = self.reshape_layer(rpn_cls_score, 2)
        rpn_cls_prob = F.softmax(rpn_cls_score_reshape)
        rpn_cls_prob_reshape = self.reshape_layer(rpn_cls_prob, len(self.anchor_scales)*3*2)

        # rpn boxes
        #rpn_bbox_pred = self.bbox_conv(rpn_conv1)
        rpn_bbox_pred = self.bbox_conv(rpn_conv1)

        # proposal layer
        #cfg_key = 'TRAIN' if self.training else 'TEST'
        cfg_key = 'TRAIN'
        rois,scores, anchor_inds, labels = self.proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred,im_info,
                                            cfg_key, self._feat_stride, self.anchor_scales, gt_boxes)

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None
            rpn_data = self.anchor_target_layer(rpn_cls_score,gt_boxes, 
                                                im_info, self._feat_stride, self.anchor_scales)
            self.cross_entropy, self.loss_box = self.build_loss(rpn_cls_score_reshape, rpn_bbox_pred, rpn_data)
            self.roi_cross_entropy = self.build_roi_loss(rpn_cls_score, rpn_cls_prob_reshape, scores,anchor_inds, labels)

        #return target_features, features, rois, scores
        bbox_pred = []
        for il in range(len(rois)):
            bbox_pred.append(network.np_to_variable(np.zeros((rois[il].size()[0],8))))
        return scores, bbox_pred, rois






    def build_loss(self, rpn_cls_score_reshape, rpn_bbox_pred, rpn_data):
        # classification loss
        #rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1)#.contiguous().view(-1, 2)
#        rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous().view(-1, 4)

        batch_size = rpn_cls_score_reshape.size()[0]


        all_rpn_cls_score = None
        all_rpn_label = None
        all_rpn_bbox_pred = None
        all_rpn_bbox_targets = None
        fg_cnt = 0

        for batch_ind in range(batch_size):

            b_ind_var = network.np_to_variable(np.zeros(1)+batch_ind, 
                                              is_cuda=True,dtype=torch.LongTensor)
            rpn_label = rpn_data[0][batch_ind].view(-1)

            b_rpn_cls_score = torch.index_select(rpn_cls_score,0,b_ind_var).contiguous().view(-1, 2)

            rpn_keep = Variable(rpn_label.data.ne(-1).nonzero().squeeze()).cuda()
            b_rpn_cls_score = torch.index_select(b_rpn_cls_score, 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label, 0, rpn_keep)

            fg_cnt += torch.sum(rpn_label.data.ne(0))

            #weight = torch.FloatTensor([.1,5])
            #weight = weight.cuda()
            #rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label, weight=weight)
            #rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label, size_average=False)
            #rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label, size_average=False, weight=weight)
            #rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)

            # box loss
            #rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
            rpn_bbox_targets = rpn_data[1][batch_ind]
            rpn_bbox_inside_weights = rpn_data[2][batch_ind]
            rpn_bbox_outside_weights = rpn_data[3][batch_ind]
            rpn_bbox_targets = torch.mul(rpn_bbox_targets, rpn_bbox_inside_weights)
            b_rpn_bbox_pred = torch.index_select(rpn_bbox_pred,0,b_ind_var)
            b_rpn_bbox_pred = torch.mul(b_rpn_bbox_pred, rpn_bbox_inside_weights)


            if all_rpn_cls_score is None:
                all_rpn_cls_score = b_rpn_cls_score
                all_rpn_label = rpn_label
                all_rpn_bbox_pred = b_rpn_bbox_pred
                all_rpn_bbox_targets = rpn_bbox_targets
            else:
                all_rpn_cls_score = torch.cat([all_rpn_cls_score,b_rpn_cls_score],0)
                all_rpn_label = torch.cat([all_rpn_label,rpn_label],0)
                all_rpn_bbox_pred = torch.cat([all_rpn_bbox_pred, b_rpn_bbox_pred], 0)
                all_rpn_bbox_targets = torch.cat([all_rpn_bbox_targets,rpn_bbox_targets], 0)


        rpn_cross_entropy = F.cross_entropy(all_rpn_cls_score, all_rpn_label, size_average=False)

        rpn_loss_box = F.smooth_l1_loss(all_rpn_bbox_pred, all_rpn_bbox_targets, size_average=False) / (fg_cnt + 1e-4)

        return rpn_cross_entropy, rpn_loss_box


    def build_roi_loss(self, rpn_cls_score_reshape, rpn_cls_prob_reshape, scores, anchor_inds, labels):

        batch_size = rpn_cls_score_reshape.size()[0]
        rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1)#.contiguous().view(-1, 2)
        bg_scores = torch.index_select(rpn_cls_score,3,network.np_to_variable(np.arange(0,9),is_cuda=True, dtype=torch.LongTensor))
        fg_scores = torch.index_select(rpn_cls_score,3,network.np_to_variable(np.arange(9,18),is_cuda=True, dtype=torch.LongTensor))
        bg_scores = bg_scores.contiguous().view(batch_size,-1,1)
        fg_scores = fg_scores.contiguous().view(batch_size,-1,1)


        all_rpn_cls_scores = None
        all_labels = None

        for batch_ind in range(batch_size):

            b_ind_var = network.np_to_variable(np.zeros(1)+batch_ind, 
                                              is_cuda=True,dtype=torch.LongTensor)
            bg = torch.index_select(bg_scores,0,b_ind_var)
            fg = torch.index_select(fg_scores,0,b_ind_var)
            #rpn_cls_score = torch.cat([bg_scores[batch_ind,:,:],fg_scores[batch_ind,:,:]],1)
            rpn_cls_score = torch.cat([bg.squeeze(0),fg.squeeze(0)],1)
            rpn_cls_score = torch.index_select(rpn_cls_score, 0, anchor_inds[batch_ind])
            if all_rpn_cls_scores is None:
                all_rpn_cls_scores = rpn_cls_score
                all_labels = labels[batch_ind]
            else:
                all_rpn_cls_scores = torch.cat([all_rpn_cls_scores,rpn_cls_score],0)
                all_labels = torch.cat([all_labels,labels[batch_ind]],0)

        #roi_cross_entropy = F.cross_entropy(rpn_cls_score, labels, size_average=False)
        roi_cross_entropy = F.cross_entropy(all_rpn_cls_scores, all_labels, size_average=False)

        return roi_cross_entropy


    @staticmethod
    def reshape_layer(x, d):
        input_shape = x.size()
        # x = x.permute(0, 3, 1, 2)
        # b c w h
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        # x = x.permute(0, 2, 3, 1)
        return x

    @staticmethod
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


    @staticmethod
    def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchor_scales, gt_boxes=None):
        
        #convert to  numpy
        rpn_cls_prob_reshape = rpn_cls_prob_reshape.data.cpu().numpy()
        rpn_bbox_pred = rpn_bbox_pred.data.cpu().numpy()

        #rois, scores, anchor_inds, labels = proposal_layer_py(rpn_cls_prob_reshape, rpn_bbox_pred,
        prop_info = proposal_layer_py(rpn_cls_prob_reshape, rpn_bbox_pred,
                                                      im_info, cfg_key, 
                                                      _feat_stride=_feat_stride,
                                                      anchor_scales=anchor_scales,
                                                      gt_boxes=gt_boxes)

        rois = []
        scores = []
        anchor_inds = []
        labels = []
        for info in prop_info:

            rois.append(network.np_to_variable(info[0], is_cuda=True))
            anchor_inds.append(network.np_to_variable(info[2], is_cuda=True,
                                                     dtype=torch.LongTensor))
            labels.append(network.np_to_variable(info[3], is_cuda=True,
                                                 dtype=torch.LongTensor))

            #just get fg scores, make bg scores 0 
            b_scores = np.zeros((info[0].shape[0], 2))
            b_scores[:,1] = info[1][:,0]
            scores.append(network.np_to_variable(b_scores, is_cuda=True))

        return rois, scores, anchor_inds, labels




    @staticmethod
    def anchor_target_layer(rpn_cls_score, gt_boxes,im_info, _feat_stride, anchor_scales):
        """
        rpn_cls_score: for pytorch (1, Ax2, H, W) bg/fg scores of previous conv layer
        gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
        gt_ishard: (G, 1), 1 or 0 indicates difficult or not
        dontcare_areas: (D, 4), some areas may contains small objs but no labelling. D may be 0
        im_info: a list of [image_height, image_width, scale_ratios]
        _feat_stride: the downsampling ratio of feature map to the original input image
        anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
        ----------
        Returns
        ----------
        rpn_labels : (1, 1, HxA, W), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
        rpn_bbox_targets: (1, 4xA, H, W), distances of the anchors to the gt_boxes(may contains some transform)
                        that are the regression objectives
        rpn_bbox_inside_weights: (1, 4xA, H, W) weights of each boxes, mainly accepts hyper param in cfg
        rpn_bbox_outside_weights: (1, 4xA, H, W) used to balance the fg/bg,
        beacuse the numbers of bgs and fgs mays significiantly different
        """
        rpn_cls_score = rpn_cls_score.data.cpu().numpy()
        #rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
        #    anchor_target_layer_py(rpn_cls_score, gt_boxes, im_info, _feat_stride, anchor_scales)
        anchor_info = \
            anchor_target_layer_py(rpn_cls_score, gt_boxes, im_info, _feat_stride, anchor_scales)

        batch_size = rpn_cls_score.shape[0]

        for batch_ind in range(batch_size):

            anchor_info[0][batch_ind] = network.np_to_variable(anchor_info[0][batch_ind],
                                                               is_cuda=True, dtype=torch.LongTensor)
            anchor_info[1][batch_ind] = network.np_to_variable(anchor_info[1][batch_ind], is_cuda=True)
            anchor_info[2][batch_ind] = network.np_to_variable(anchor_info[2][batch_ind], is_cuda=True)
            anchor_info[3][batch_ind] = network.np_to_variable(anchor_info[3][batch_ind], is_cuda=True)
            #rpn_labels = network.np_to_variable(rpn_labels, is_cuda=True, dtype=torch.LongTensor)
            #rpn_bbox_targets = network.np_to_variable(rpn_bbox_targets, is_cuda=True)
            #rpn_bbox_inside_weights = network.np_to_variable(rpn_bbox_inside_weights, is_cuda=True)
            #rpn_bbox_outside_weights = network.np_to_variable(rpn_bbox_outside_weights, is_cuda=True)

        #return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
        return anchor_info 

    def load_from_npz(self, params):
        # params = np.load(npz_file)
        self.features.load_from_npz(params)

        pairs = {'conv1.conv': 'rpn_conv/3x3', 'score_conv.conv': 'rpn_cls_score', 'bbox_conv.conv': 'rpn_bbox_pred'}
        own_dict = self.state_dict()
        for k, v in pairs.items():
            key = '{}.weight'.format(k)
            param = torch.from_numpy(params['{}/weights:0'.format(v)]).permute(3, 2, 0, 1)
            own_dict[key].copy_(param)

            key = '{}.bias'.format(k)
            param = torch.from_numpy(params['{}/biases:0'.format(v)])
            own_dict[key].copy_(param)



    def get_features(self, im_data):
        im_data = network.np_to_variable(im_data, is_cuda=True)
        im_data = im_data.permute(0, 3, 1, 2)
        im_in =im_data# self.input_conv(im_data)
        features = self.features(im_in)

        return features


    
