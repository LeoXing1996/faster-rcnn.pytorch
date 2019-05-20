import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN

from model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.relation.relation_block import relation_layer
from model.rpn.bbox_transform import clip_boxes, bbox_transform_inv

import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta


class _fasterRCNN_rel_cls(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN_rel_cls, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        self.bz = cfg.TRAIN.IMS_PER_BATCH
        if self.training:
            self.img_bz = cfg.TRAIN.BATCH_SIZE
        else:
            self.img_bz = cfg.TEST.RPN_POST_NMS_TOP_N
        # loss
        # self.RCNN_loss_cls = 0
        # self.RCNN_loss_bbox = 0
        # fix for mGPUs
        self.RCNN_loss_cls = torch.tensor([0])
        self.RCNN_loss_bbox = torch.tensor([0])

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0)

        self.skip_layers = ['RCNN_cls_score.weight', 'RCNN_cls_score.bias',
                            'RCNN_bbox_pred.weight', 'RCNN_bbox_pred.bias']

        self.cls_feat_layer = nn.Linear(2 * self.n_classes, 1).cuda()
        self.area_feat_layer = nn.Linear(2 * self.n_classes, 1).cuda()
        self.rel_weight_layer = nn.Linear(3, 1).cuda()

    def forward(self, im_data, im_info, gt_boxes, num_boxes, times=2):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        #####################################
        # relation work
        #####################################

        # 0. init the `graph`
        id_info, dist_info = self.init_graph(rois, im_info)
        RCNN_loss_cls_mean = 0
        RCNN_loss_bbox_mean = 0

        for _ in range(times):

            # 1. get pooled feat and cls, reg result
            avg_pooled_feat = self._head_to_tail(pooled_feat)
            cls_score = self.RCNN_cls_score(avg_pooled_feat)
            bbox_pred = self.RCNN_bbox_pred(avg_pooled_feat)

            # 2. get cls info
            cls_prob = torch.softmax(cls_score, dim=1)
            cls_info = cls_prob.view(self.bz, -1, self.n_classes)

            # 3. get area info
            box_tmp = self.get_bbox_temp(bbox_pred, rois, im_info)
            area_info = torch.cat([self.get_bbox_area(box_tmp[:, :, i * 4: (i + 1) * 4], im_info).unsqueeze(dim=-1)
                                   for i in range(self.n_classes)], dim=-1)

            # 4. fresh pooled feat
            pooled_feat = pooled_feat.view(self.bz, -1, 1024, 7, 7)
            pooled_feat = self.fresh_pooled_feat(cls_info, area_info, dist_info, id_info, pooled_feat)
            pooled_feat = pooled_feat.view(-1, 1024, 7, 7)

            # 5. calculate the loss
            if self.training and not self.class_agnostic:
                # select the corresponding columns according to roi labels
                bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
                bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                                rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1,
                                                                                                 4))
                bbox_pred = bbox_pred_select.squeeze(1)
            if self.training:
                # classification loss
                RCNN_loss_cls_tmp = F.cross_entropy(cls_score, rois_label)
                # bounding box regression L1 loss
                RCNN_loss_bbox_tmp = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
            else:
                RCNN_loss_cls_tmp = 0
                RCNN_loss_bbox_tmp = 0
            RCNN_loss_cls_mean = RCNN_loss_cls_tmp + RCNN_loss_cls_mean
            RCNN_loss_bbox_mean = RCNN_loss_bbox_tmp + RCNN_loss_cls_mean

        # # feed pooled features to top model
        # pooled_feat = self._head_to_tail(pooled_feat)
        #
        # # compute bbox offset
        # bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        # if self.training and not self.class_agnostic:
        #     # select the corresponding columns according to roi labels
        #     bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
        #     bbox_pred_select = torch.gather(bbox_pred_view, 1,
        #                                     rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
        #     bbox_pred = bbox_pred_select.squeeze(1)
        #
        # # compute object classification probability
        # cls_score = self.RCNN_cls_score(pooled_feat)
        # cls_prob = F.softmax(cls_score, 1)
        #
        # RCNN_loss_cls = 0
        # RCNN_loss_bbox = 0
        #
        # if self.training:
        #     # classification loss
        #     RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
        #
        #     # bounding box regression L1 loss
        #     RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
        RCNN_loss_cls = RCNN_loss_cls_mean / times
        RCNN_loss_bbox = RCNN_loss_bbox_mean / times
        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def load_ckpt(self, ckpt):
        state_dict = self.state_dict()
        pretrianed_model = {k: v for k, v in ckpt.items() if k in state_dict.keys() and k not in self.skip_layers}
        state_dict.update(pretrianed_model)
        self.load_state_dict(state_dict)

    def get_overlaps(self, proposals):
        N = proposals.shape[1]
        # bz = proposals.shape[0]
        proposals_1 = proposals.view(self.bz, 1, N, 4).expand(self.bz, N, N, 4)
        proposals_2 = proposals.view(self.bz, N, 1, 4).expand(self.bz, N, N, 4)
        w = torch.min(proposals_1[:, :, :, 2], proposals_2[:, :, :, 2]) - torch.max(proposals_1[:, :, :, 0],
                                                                                    proposals_2[:, :, :, 0]) + 1
        h = torch.min(proposals_1[:, :, :, 3], proposals_2[:, :, :, 3]) - torch.max(proposals_1[:, :, :, 1],
                                                                                    proposals_2[:, :, :, 1]) + 1
        w[w < 0] = 0
        h[h < 0] = 0

        area = (proposals[:, :, 3] - proposals[:, :, 1] + 1) * (proposals[:, :, 2] - proposals[:, :, 0] + 1)
        area_1 = area.view(self.bz, N, 1)
        area_2 = area.view(self.bz, 1, N)

        union = area_1 + area_2 - w * h
        inter = w * h
        overlaps = inter / union
        return overlaps

    def get_dist(self, proposal, im_info):
        N = proposal.shape[1]
        h = proposal[:, :, 3] - proposal[:, :, 1] + 1
        w = proposal[:, :, 2] - proposal[:, :, 0] + 1
        ctr_x = proposal[:, :, 0] + 0.5 * w
        ctr_y = proposal[:, :, 1] + 0.5 * h

        ctr_x_1 = ctr_x.view(self.bz, N, 1)
        ctr_x_2 = ctr_x.view(self.bz, 1, N)
        ctr_y_1 = ctr_y.view(self.bz, N, 1)
        ctr_y_2 = ctr_y.view(self.bz, 1, N)

        dist = torch.sqrt(torch.pow(ctr_x_1 - ctr_x_2, 2) + torch.pow(ctr_y_1 - ctr_y_2, 2))
        dist = dist / torch.sqrt(torch.pow(im_info[:, 0], 2) + torch.pow(im_info[:, 1], 2)).view(-1, 1, 1)
        return dist

    def get_bbox_area(self, proposals, im_info):
        img_area = im_info[:, 0] * im_info[:, 1]
        area = (proposals[:, :, 3] - proposals[:, :, 1] + 1) * (proposals[:, :, 2] - proposals[:, :, 0] + 1)
        return area / img_area.view(self.bz, 1)

    def get_target_rois(self, proposals, im_info, nea_nums=3, far_nums=3):

        overlaps = self.get_overlaps(proposals[:, :, 1:])
        dist = self.get_dist(proposals[:, :, 1:], im_info)

        # get furthest $NUM$ bboxes which overlap is NOT zero
        # and get neareat $NUM$ bboxes which overlap IS zero
        dist_nea = torch.where(overlaps != 0, dist, torch.zeros_like(dist))
        dist_far = torch.where(overlaps == 0, dist, torch.zeros_like(dist).fill_(dist.max()))

        bbox_nea_dis, bbox_nea_id = torch.topk(dist_nea, nea_nums, largest=True)
        bbox_far_dis, bbox_far_id = torch.topk(dist_far, far_nums, largest=False)

        return bbox_nea_id, bbox_far_id, bbox_nea_dis, bbox_far_dis

    def init_graph(self, proposals, im_info):
        bbox_nea_id, bbox_far_id, bbox_nea_dis, bbox_far_dis = self.get_target_rois(proposals, im_info)
        bbox_id = torch.cat((bbox_nea_id, bbox_far_id), dim=-1)
        dist_info = torch.cat((bbox_nea_dis, bbox_far_dis), dim=-1)  # --> keep as static
        return bbox_id, dist_info

    def get_bbox_temp(self, bbox_pred, proposals, im_info):
        bbox = bbox_pred.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() + torch.FloatTensor(
            cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        bbox = bbox.view(self.bz, -1, 4 * self.n_classes)
        bbox = bbox_transform_inv(proposals[:, :, 1:], bbox, self.bz)
        bbox = clip_boxes(bbox, im_info.data, self.bz)
        return bbox

    def get_rel_weight(self, cls_inf, area_inf, dist_inf):
        # cls_feat: [6, 2*cls_num]
        # area_feat: [6, 2*cls_num]
        # dist_feat: [6, 1]
        cls_feat = self.cls_feat_layer(cls_inf)
        area_feat = self.area_feat_layer(area_inf)
        rel_weight = self.rel_weight_layer(torch.cat((cls_feat, area_feat, dist_inf), dim=1))
        return rel_weight

    def get_rel_feat(self, cls_info, area_info, dist_info, id_info, pooled_feat):
        # cls_info ---> [cfg.TRAIN.BATCH_SIZE, len(cls_num)]
        # area_info --> [cfg.TRAIN.BATCH_SIZE, len(cls_num)]
        # dist_info --> [cfg.TRAIN.BATCH_SIZE, 6]
        # id_info ----> [cfg.TRAIN.BATCH_SIZE, 6]
        # pooled_feat-> [cfg.TRAIN.BATCH_SIZE, 1024, 7, 7] --> has been spited
        # self.img_bz --> cfg.TRAIN.BATCH_SIZE
        # rel_feats = torch.zeros_like(pooled_feat)
        rel_feats = []
        for i in range(self.img_bz):
            cls_info_sin = torch.cat((cls_info[id_info[i]], cls_info[i].expand(6, -1)), dim=-1)
            area_info_sin = torch.cat((area_info[id_info[i]], area_info[i].expand(6, -1)), dim=-1)
            dist_info_sin = dist_info[i].view(6, 1)

            rel_weight = self.get_rel_weight(cls_info_sin, area_info_sin, dist_info_sin)
            rel_feat = pooled_feat[id_info[i]] * torch.softmax(rel_weight / 100, 0).view(6, 1, 1, 1)
            rel_feat = rel_feat.sum(dim=0, keepdim=True)
            # rel_feats[i] = rel_feat
            rel_feats.append(rel_feat)
        return torch.cat(rel_feats, dim=0)

    def fresh_pooled_feat(self, cls_info, area_info, dist_info, id_info, pooled_feat):
        # cls_info ---> [cfg.TRAIN.IMS_PER_BATCH, cfg.TRAIN.BATCH_SIZE, len(cls_num)]
        # area_info --> [cfg.TRAIN.IMS_PER_BATCH, cfg.TRAIN.BATCH_SIZE, len(cls_num)]
        # dist_info --> [cfg.TRAIN.IMS_PER_BATCH, cfg.TRAIN.BATCH_SIZE, 6]
        # id_info ----> [cfg.TRAIN.IMS_PER_BATCH, cfg.TRAIN.BATCH_SIZE, 6]
        # pooled_feat-> [cfg.TRAIN.IMS_PER_BATCH, cfg.TRAIN.BATCH_SIZE, 1024, 7, 7] --> has been spited
        # self.bz -- > cfg.TRAIN.IMS_PER_BATCH
        rel_feats_batch = []
        for b in range(self.bz):
            cls_info_bz = cls_info[b]
            area_info_bz = area_info[b]
            dist_info_bz = dist_info[b]
            id_info_bz = id_info[b]
            rel_feats = self.get_rel_feat(cls_info_bz, area_info_bz, dist_info_bz, id_info_bz, pooled_feat[b])
            rel_feats_batch.append(rel_feats.unsqueeze(0))
            # pooled_feat[b] = rel_feats + pooled_feat[b]
        rel_feats_batch = torch.cat(rel_feats_batch, dim=0)

        return pooled_feat + rel_feats_batch
