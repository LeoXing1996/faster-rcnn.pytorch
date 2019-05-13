import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
from matplotlib.pyplot import imshow

import pdb
import sys
import time
import pprint
import pickle
import argparse

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vtils
from torchvision.utils import make_grid
from torchvision import transforms

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.resnet import resnet
from model.rpn.generate_anchors import generate_anchors
from model.rpn.rpn import _RPN

import pdb

xrange = range  # Python 3

# 1. initialize dataset --> load Pascal VOC dataset
imdbval_name = 'voc_2007_trainval'
imdb, roidb, ratio_list, ratio_index = combined_roidb(imdbval_name, False)

# 2. initialize faster-rcnn --> load faster-rcnn trained with coco

set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
cfg_file = "../cfgs/res101.yml"
cfg_from_file(cfg_file)
cfg_from_list(set_cfgs)
net_classes = np.asarray((['__background__', 'person', 'bicycle',
                           'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                           'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                           'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                           'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                           'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                           'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                           'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']))
fasterRCNN = resnet(net_classes, 101, pretrained=False, class_agnostic=False)
fasterRCNN.create_architecture()
ckpt = torch.load('../data/benchmark/res101/coco/faster_rcnn_1_10_14657.pth')
fasterRCNN.load_state_dict(ckpt['model'])

cfg.POOLING_MODE = 'align'

# 1. get RPN and backbone

backbone = fasterRCNN.RCNN_base
RPN = fasterRCNN.RCNN_rpn


# 2. visual anchors
def visual_anchors(img_path, ctr_coor=None, base_size=16, ratios=[0.5, 1, 2],
                   scales=2 ** np.arange(3, 6), show=None, save_dir='./output/vis_anchor/', save_name=None):
    anchors = generate_anchors(base_size, ratios, scales)
    assert os.path.isfile(img_path)
    img = Image.open(img_path)

    if ctr_coor is None:
        ctr_coor = np.array(img.size)
    else:
        assert 0 < ctr_coor[0] < img.size[0]
        assert 0 < ctr_coor[1] < img.size[1]

    anchors[:, 0] = anchors[:, 0] + ctr_coor[0]
    anchors[:, 1] = anchors[:, 1] + ctr_coor[1]
    anchors[:, 2] = anchors[:, 2] + ctr_coor[0]
    anchors[:, 3] = anchors[:, 3] + ctr_coor[1]

    draw = ImageDraw.Draw(img)
    for anchor in anchors:
        draw.rectangle(anchor)

    if show:
        img.show()

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if save_name is None:
        save_name = img_path.split('/')[-1][-1] + 'anchor.jpg'
    img.savefig(os.path.join(save_dir, save_name))


def visual_anchors_ind():
    # visual anchors through anchors' index
    # which can be adapt to proposals' visualization
    pass


# 3. visual feature map and CAM
# base_feature =
# mean of the images, use to recover images
PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])


def get_im_size(im_info):
    # im_info: torch.Tensor with shape as [3, ]
    return im_info[:2].numpy().astype(np.int).tolist()


def vis_fm(fm, im_info, num=3, save_dir='./output/vis_fm/', save_name=None):
    # fm: feature map, torch.Tensor with shape as [bz, channels, h, w]
    # im_info: torch.Tensor with shape as [3, ]
    # num: num of feature map want to visual
    # save_dir
    # save_name
    fm = fm.permute(1, 0, 2, 3)
    assert num <= fm.shape[0]
    if num < fm.shape[0]:
        id_ = torch.randint(fm.shape[0], (num,))
        fm_to_vis = fm[id_, ::]
    else:
        fm_to_vis = fm

    # resize feature map to original size `scale_factor` default as 16
    fm_to_vis = F.interpolate(fm_to_vis, size=get_im_size(im_info))

    fm_vis = make_grid(fm_to_vis, normalize=True, nrow=5, pad_value=1)

    toPIL = transforms.ToPILImage()
    fm_vis_pil = toPIL(fm_vis)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    fm_vis_pil.save(os.path.join(save_dir, save_name) + '.jpg')
    return fm_vis_pil


def get_heat_map(fm, im_data, im_info, save_dir='./output/vis_CAM/', save_name=None):
    # fm: feature map , torch.Tensor with shape as  [bz, channels, h, w]
    # im_data: original images, torch.Tensor with shape as [channels, h, w] and value range
    #          of the image is -128~128 (a uint8 image subtract PIXEL_MEAN)
    # im_info: torch.Tensor with shape as [3, ]
    fm = fm.permute(1, 0, 2, 3)
    fm = torch.mean(fm, dim=0, keepdim=True)
    fm_ori_scale = F.interpolate(fm, size=get_im_size(im_info))
    fm_ori_scale = make_grid(fm_ori_scale, normalize=True)
    fm_np = fm_ori_scale.permute(1, 2, 0).numpy()[:, :, 0]

    cmap = matplotlib.cm.get_cmap('jet')
    CAM = Image.fromarray((cmap(fm_np) * 255).astype(np.uint8)).convert('RGB')

    ori_img = im_data.permute(1, 2, 0).numpy() + PIXEL_MEANS
    ori_img = Image.fromarray(np.uint8(ori_img))

    CAM_img = Image.blend(CAM, ori_img, alpha=0.5)
    if not os.path.isfile(save_dir):
        os.makedirs(save_dir)
    if save_name is None:
        save_name = 'CAM'
    CAM.save(os.path.join(save_dir, save_name+'.png'))
    CAM_img.save(os.path.join(save_dir, save_name + '_img.png'))
    return CAM, CAM_img


# 4. get raw rois and visual
def get_ctr(anchors):
    # anchors: torch.tensor [num, 4] --> x_min, y_min, x_max, y_max
    width = anchors[:, 2] - anchors[:, 0] + 1
    height = anchors[:, 3] - anchors[:, 1] + 1
    ctr_x = anchors[:, 0] + 0.5 * width
    ctr_y = anchors[:, 1] + 0.5 * height

    return ctr_x, ctr_y


def visual_proposal(im_data, proposals, anchors, point_nums=2, anchors_num=12, vis_anchors=3, idx=None):
    # im_data: original images, torch.Tensor with shape as [channels, h, w] and value range
    #          of the image is -128~128 (a uint8 image subtract PIXEL_MEAN)
    # proposals: regressed box given by RPN, torch.Tensor(cpu), [num, 4]
    # anchors: torch.tensor(cpu) [num, 4]
    # anchors_num: num of anchors generate on a single point
    # vis_anchors: the number of anchors want to vis on a single anchor

    im_data = im_data.permute(1, 2, 0).numpy() + PIXEL_MEANS
    im_data = np.uint8(im_data)
    src_img = Image.fromarray(im_data)

    points = int(proposals.shape[0] / anchors_num)

    if idx is None:
        idx = np.random.randint(0, points, point_nums)
    idx = [np.arange(id_ * anchors_num, (id_ + 1) * anchors_num - 1, int(anchors_num / vis_anchors)) for id_ in idx]
    idx = np.concatenate(idx, axis=-1)

    proposals = proposals[idx].numpy()
    anchors = anchors[idx].numpy()
    ctr_x, ctr_y = get_ctr(anchors)

    draw = ImageDraw.Draw(src_img)
    for bbox, anch in zip(proposals, anchors):
        draw.rectangle(bbox, outline=(255, 105, 180), width=3)
        draw.rectangle(anch, outline=(65, 106, 225), width=3)

    for x, y in zip(ctr_x, ctr_y):
        x_0 = x - 3
        y_0 = y - 3
        x_1 = x + 3
        y_1 = y + 3
        draw.rectangle((x_0, y_0, x_1, y_1), fill=(255, 100, 0))
    return src_img


# 5. get roi after NMS and visual
def visual_proposal_nms(im_data, proposals, num=None, idx=None):
    # do not visual anchors

    # im_data: original images, torch.Tensor with shape as [channels, h, w] and value range
    #          of the image is -128~128 (a uint8 image subtract PIXEL_MEAN)
    # proposals: regressed box given by RPN, torch.Tensor(cpu), [num, 4]
    # num: numbers of the rois want to visual
    # idx: special index roi want to visual

    im_data = im_data.permute(1, 2, 0).numpy() + PIXEL_MEANS
    im_data = np.uint8(im_data)
    src_img = Image.fromarray(im_data)

    #     points = int(proposals.shape[0] / anchors_num)

    #     if idx is None:
    #         idx = np.random.randint(0, points, point_nums)
    #     idx = [np.arange(id_*anchors_num, (id_+1)*anchors_num-1, int(anchors_num/vis_anchors)) for id_ in idx]
    #     idx = np.concatenate(idx, axis=-1)
    if num is None:
        proposals = proposals.numpy()
    else:
        if idx is None:
            proposals = proposals.numpy()[np.random.randint(0, proposals.shape[0], (num, 1))]
        else:
            proposals = proposals.numpy()[idx]
    #     anchors = anchors[idx].numpy()
    #     ctr_x, ctr_y = get_ctr(anchors)

    draw = ImageDraw.Draw(src_img)
    for bbox in proposals:
        draw.rectangle(bbox, outline=(255, 105, 180), width=3)
    #         draw.rectangle(anch, outline=(65, 106, 225), width=3)

    return src_img


# TODO build the visual feature!!
class visualization_offical_dataset(object):
    def __int__(self, imdbval_name='voc_2007_trainval'):
        # dataset setting ---> load PASCAL VOC dataset
        # imdbval_name = 'voc_2007_trainval'
        imdb, roidb, ratio_list, ratio_index = combined_roidb(imdbval_name, False)
        self.dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1,
                                 imdb.num_classes, training=False, normalize=False)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False,
                                                      num_workers=0, pin_memory=True)
        self.data_iter = iter(self.dataloader)
        self.im_data = torch.FloatTensor(1).cuda()
        self.im_info = torch.FloatTensor(1).cuda()
        self.num_boxes = torch.LongTensor(1).cuda()
        self.gt_boxes = torch.FloatTensor(1).cuda()

        # init faster-rcnn --> trained by coco
        set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        cfg_file = "../cfgs/res101.yml"
        cfg_from_file(cfg_file)
        cfg_from_list(set_cfgs)
        net_classes = np.asarray((['__background__', 'person', 'bicycle',
                                   'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                                   'umbrella',
                                   'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                                   'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                                   'bottle',
                                   'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                                   'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                                   'couch',
                                   'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                                   'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']))
        self.fasterRCNN = resnet(net_classes, 101, pretrained=False, class_agnostic=False)
        self.fasterRCNN.create_architecture()
        fasterRCNN.create_architecture()
        ckpt = torch.load('../data/benchmark/res101/coco/faster_rcnn_1_10_14657.pth')
        self.fasterRCNN.load_state_dict(ckpt['model'])
        self.fasterRCNN.cuda()
        self.RCNN_base = self.fasterRCNN.RCNN_base
        self.RPN = self.fasterRCNN.RCNN_rpn
        # self.

    def get_data(self):
        data = next(self.data_iter)
        self.im_data.data.resize_(data[0].size()).copy_(data[0])
        self.im_info.data.resize_(data[1].size()).copy_(data[1])
        self.gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        self.num_boxes.data.resize_(data[3].size()).copy_(data[3])

    def get_base_feature(self):
        return self.RCNN_base(self.im_data.data).detach().cpu()

    def get_RPN_proposal(self):
        # forward
        base_fm = self.get_base_feature()
        rpn_conv1 = F.relu(RPN.RPN_Conv(base_fm), inplace=True)

        rpn_cls_score = self.RPN.RPN_cls_score(rpn_conv1)
        rpn_cls_score_reshape = self.RPN.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.RPN.reshape(rpn_cls_prob_reshape, RPN.nc_score_out)

        rpn_bbox_pred = self.RPN.RPN_bbox_pred(rpn_conv1)

        scores = rpn_cls_prob[:, self.RPN.RPN_proposal._num_anchors:, ::]
        bbox_deltas = rpn_bbox_pred

        pre_nms_topN = cfg['TEST'].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg['TEST'].RPN_POST_NMS_TOP_N
        nms_thresh = cfg['TEST'].RPN_NMS_THRESH
        min_size = cfg['TEST'].RPN_MIN_SIZE

        batch_size = bbox_deltas.size(0)

        feat_height, feat_width = scores.size(2), scores.size(3)
        shift_x = np.arange(0, feat_width) * RPN.RPN_proposal._feat_stride
        shift_y = np.arange(0, feat_height) * RPN.RPN_proposal._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(scores).float()

        # get predict bbox before clip
        A = self.RPN.RPN_proposal._num_anchors
        K = shifts.size(0)

        RPN.RPN_proposal._anchors = RPN.RPN_proposal._anchors.type_as(scores)
        anchors = RPN.RPN_proposal._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:

        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)

        # Same story for the scores:
        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch_size, -1)

        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)
        proposals_clip = clip_boxes(proposals, self.im_info, batch_size)
        proposals_nms = self.RPN.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data, self.im_info, 'TEST'))

        proposals = proposals[0].detach().cpu()
        proposals_clip = proposals_clip[0].detach().cpu()
        proposals_nms = proposals_nms[0][:, 1:].detach().cpu()

        return proposals, proposals_clip, proposals_nms
