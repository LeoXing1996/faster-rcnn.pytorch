import torch
import torch.nn as nn

from model.utils.config import cfg


class relation_layer(nn.Module):
    def __init__(self, class_nums, input_channels, nea_nums=3, far_nums=3):
        super(relation_layer, self).__init__()
        self.class_nums = class_nums
        self.nea_nums = nea_nums
        self.far_nums = far_nums
        self.input_channels = input_channels
        self.class_conv = nn.Sequential(*[nn.Conv2d(input_channels, self.class_nums, 3, 1, 1),
                                          nn.Conv2d(self.class_nums, input_channels, 3, 1, 1)])
        self.bz = cfg.TRAIN.BATCH_SIZE

    @staticmethod
    def get_dist(proposals):
        N = proposals.shape[0]
        h = proposals[:, 3] - proposals[:, 1] + 1
        w = proposals[:, 2] - proposals[:, 0] + 1
        ctr_x = proposals[:, 0] + 0.5 * w
        ctr_y = proposals[:, 1] + 0.5 * h

        ctr_x_1 = ctr_x.view(N, 1)
        ctr_x_2 = ctr_x.view(1, N)
        ctr_y_1 = ctr_y.view(N, 1)
        ctr_y_2 = ctr_y.view(1, N)

        dist = torch.sqrt(torch.pow(ctr_x_1 - ctr_x_2, 2) + torch.pow(ctr_y_1 - ctr_y_2, 2))
        return dist

    @staticmethod
    def get_overlaps(proposals):
        N = proposals.shape[0]
        proposals_1 = proposals.view(1, N, 4).expand(N, N, 4)
        proposals_2 = proposals.view(N, 1, 4).expand(N, N, 4)
        w = torch.min(proposals_1[:, :, 2], proposals_2[:, :, 2]) - torch.max(proposals_1[:, :, 0],
                                                                              proposals_2[:, :, 0]) + 1
        h = torch.min(proposals_1[:, :, 3], proposals_2[:, :, 3]) - torch.max(proposals_1[:, :, 1],
                                                                              proposals_2[:, :, 1]) + 1
        w[w < 0] = 0
        h[h < 0] = 0

        area = (proposals[:, 3] - proposals[:, 1] + 1) * (proposals[:, 2] - proposals[:, 0] + 1)
        area_1 = area.view(N, 1)
        area_2 = area.view(1, N)

        union = area_1 + area_2 - w * h
        inter = w * h
        overlaps = inter / union
        return overlaps

    def get_target_rois(self, proposals):
        # input proposals should be shape as [300, 4]
        overlaps = self.get_overlaps(proposals)
        dist = self.get_dist(proposals)

        # get furthest $NUM$ bboxes which overlap is NOT zero
        # and get nearest $NUM$ bboxes which overlap IS zero
        dist_nea = torch.where(overlaps != 0, dist, torch.zeros_like(dist))
        dist_far = torch.where(overlaps == 0, dist, torch.zeros_like(dist).fill_(dist.max()))

        bbox_nea_dis, bbox_nea_id = torch.topk(dist_nea, self.nea_nums, largest=True)
        bbox_far_dis, bbox_far_id = torch.topk(dist_far, self.far_nums, largest=False)

        return bbox_nea_id, bbox_far_id, bbox_nea_dis, bbox_far_dis

    def relation_single_batch(self, proposals, pooled_feat, times=2):
        # proposals [bz, 4]
        # pooled_feat [bz, channels, fm_size, fm_size]
        input_channels = pooled_feat.shape[1]
        fm_size = pooled_feat.shape[-1]
        relation_nums = self.nea_nums + self.far_nums

        bbox_nea_id, bbox_far_id, bbox_nea_dis, bbox_far_dis = self.get_target_rois(proposals)
        tar_bbox_id = torch.cat((bbox_nea_id, bbox_far_id), dim=1)

        #  here we think dist present the relation between bbox and norm distance through divide 100
        tar_bbox_weight_dist = torch.cat((bbox_nea_dis, bbox_far_dis), dim=1)
        tar_bbox_weight_dist = torch.softmax(tar_bbox_weight_dist / 100, dim=1)

        for _ in range(times):
            # 1. adapt all feature map to the class_conv
            pooled_cls_act = self.class_conv(pooled_feat)

            # 2. calculate relation weights
            # tar_bbox_weight_dist[i]:           [relation_nums, ]
            # pooled_feat_class[tar_bbox_id[i]]: [relation_nums, input_channels, fm_size, fm_size]

            pooled_feat_rel = torch.cat([torch.matmul(tar_bbox_weight_dist[i].view(1, -1),
                                                      pooled_cls_act[tar_bbox_id[i]].view(relation_nums, -1)
                                                      ).view(1, input_channels, fm_size, fm_size)
                                         for i in range(tar_bbox_id.shape[0])], dim=0)

            pooled_feat = pooled_feat + pooled_feat_rel

        return pooled_feat

    def forward(self, proposals, pooled_feat, times=2):
        # proposals [n, bz, 4]
        # pooled_feat [n*bz, channels, fm_size, fm_size]
        # times: broadcast nums
        # here `n` is the batch size of image and `bz`id number of rois
        N = proposals.shape[0]
        bz = self.bz

        pooled_feat = torch.cat([self.relation_single_batch(proposals[n], pooled_feat[bz*n: bz*(n+1)], times)
                                     for n in range(N)])

        return pooled_feat
