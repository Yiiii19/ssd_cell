# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True, focal_loss = False):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        self.focal_loss = focal_loss

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        # loc_data: [batch_size, 8732, 4]
        # conf_data: [batch_size, 8732, num_class]
        # priors: [8732, 4]  default box
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)  # num = batch_size
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        # num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        best_prior = torch.Tensor(num)

        for idx in range(num):
            # targets是列表, 列表的长度为batch_size, 列表中每个元素为一个 tensor,
            # 其 shape 为 [num_objs, 5], 其中 num_objs 为当前图片中物体的数量,
            # 第二维前4个元素为边框坐标, 最后一个元素为类别编号(1~20)
            truths = targets[idx][:, :-1].data  # get locations [num_objs, 4]
            labels = targets[idx][:, -1].data   # get label [num_objs]
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx, best_prior)
            # conf_t: each prior box corresponds to each label e.g. 1, 2, 3 ...

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # # wrap targets
        # loc_t = Variable(loc_t, requires_grad=False)
        # conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0    # include True or False [num, num_priors]
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]

        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)  # [num, num_priors, 4] [True False]
        # since each box has 4 offsets, expand 4 times

        loc_p = loc_data[pos_idx].view(-1, 4)   # only matched/bigger overlap prior box activated
        loc_t = loc_t[pos_idx].view(-1, 4)      # same size with loc_p for calculating loss_l
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        # loss_c[pos] = 0
        loss_c[pos.view(-1, 1)] = 0 # matched, loss set 0
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)      # [num, num_priors]
        num_pos = pos.long().sum(1, keepdim=True)   # count positive number for each photo
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        # count negative number; num_prior — num_pos seems more reasonable(but will not reach it)

        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)     # [num, num_priors, num_classes]
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)     # [num, num_priors, num_classes]
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        # conf_p: [num, num_priors, num_classes]
        # targets_weighted: [num, num_priors]

        if self.focal_loss:
            loss_c = FocalLoss(gamma=2, alpha=0.25)(conf_data.view(-1, self.num_classes)
                                                    , conf_t)
        else:
            loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c, best_prior.sum()



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=0.25, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        # if input.dim()>2:
        #     input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        #     input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        #     input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)     # only choose the probability same with
        # target(true label). e.g. logpt:[0.2, 0.3, 0.5], true label is index=0(background) ->
        # 0.2 will be chosen -> logpt has the sam dimension with target [num * num_priors]

        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        # if self.alpha is not None:
        #     if self.alpha.type()!=input.data.type():
        #         self.alpha = self.alpha.type_as(input.data)
        #     at = self.alpha.gather(0, target.data.view(-1))
        #     print(at.size(), at)
        #     logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt * self.alpha
        if self.size_average: return loss.mean()
        else: return loss.sum()

