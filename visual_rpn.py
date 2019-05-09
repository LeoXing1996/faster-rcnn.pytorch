import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.resnet import resnet
from model.rpn.rpn import _RPN

import pdb

xrange = range  # Python 3

# parser.add_argument('--ls', dest='large_scale',
#                     help='whether use large imag scale',
#                     action='store_true')

DATASET = 'coco'
imdb_name = "coco_2014_train+coco_2014_valminusminival"
imdbval_name = "coco_2014_minival"
set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

cfg_file = "cfgs/res101.yml"
cfg_from_file(cfg_file)
cfg_from_list(set_cfgs)
imdb, roidb, ratio_list, ratio_index = combined_roidb(imdbval_name, False)


cfg.POOLING_MODE = 'align'

im_data = torch.FloatTensor(1)
im_info = torch.FloatTensor(1)
num_boxes = torch.LongTensor(1)
gt_boxes = torch.FloatTensor(1)

im_data = im_data.cuda()
im_info = im_info.cuda()
num_boxes = num_boxes.cuda()
gt_boxes = gt_boxes.cuda()

# 1. initialize a res101 mode as feature extractor
# resnet = resnet101()
# RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
#                           resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3)
# resnet_state_dict = torch.load('data/pretrained_model/resnet101_caffe.pth')
# RCNN_base.load_state_dict({k: v for k, v in resnet_state_dict.items() if k in RCNN_base.state_dict()})
#
# # 2. initialize a RPN
# RPN = _RPN(1024)
# load_name = 'data/branchmark/res101/coco/faster_rcnn_1_10_14657.pth'
# checkpoint = torch.load(load_name)
# RPN_keys = RPN.state_dict().keys()
# RPN_state_dict = {name.split('.', 1)[-1]: parm for name, parm in checkpoint['model'].items() if name.split('.', 1)[-1] in RPN_keys}
# RPN.load_state_dict(RPN_state_dict)
# RCNN_top = nn.Sequential(resnet.layer4)

# 1. initial faster-RCNN and extract RPN and backbone
fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=False)
fasterRCNN.create_architecture()
fasterRCNN.cuda()

backbone = fasterRCNN.RCNN_base
RPN = fasterRCNN.RCNN_rpn

# todo: 1. visual anchors
# todo: 2. get raw rois
# todo: 3. get roi after cliping
# todo: 4. get roi after nms
# todo: 5. get
