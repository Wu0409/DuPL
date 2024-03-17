import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import datetime
from utils import evaluate, imutils

from tqdm import tqdm
from utils import optimizer
from utils.pyutils import AverageMeter
import utils.cam_helper as cam_helper
from utils.pyutils import AverageMeter, cal_eta, format_tabs, setup_logger
from datasets import voc as voc
from datasets import coco as coco

import sys


def get_optimizer(param_groups, args):
    optim = getattr(optimizer, args.optimizer)(
        params=[
            {
                "params": param_groups[0],
                "lr": args.lr,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[1],
                "lr": args.lr,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[2],
                "lr": args.lr * 10,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[3],
                "lr": args.lr * 10,
                "weight_decay": args.wt_decay,
            },
        ],
        lr=args.lr,
        weight_decay=args.wt_decay,
        betas=args.betas,
        warmup_iter=args.warmup_iters,
        max_iter=args.max_iters,
        warmup_ratio=args.warmup_lr,
        power=args.power
    )
    return optim


def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w
    # _hw = (h + max(dilations)) * (w + max(dilations))
    mask = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius + 1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius + 1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask


def get_aff_loss(inputs, targets):
    pos_label = (targets == 1).type(torch.int16)
    pos_count = pos_label.sum() + 1
    neg_label = (targets == 0).type(torch.int16)
    neg_count = neg_label.sum() + 1
    # inputs = torch.sigmoid(input=inputs)

    pos_loss = torch.sum(pos_label * (1 - inputs)) / pos_count
    neg_loss = torch.sum(neg_label * (inputs)) / neg_count

    return 0.5 * pos_loss + 0.5 * neg_loss, pos_count, neg_count


def validate_siamase(model=None, data_loader=None, args=None, return_item=False):
    gts = []
    preds_1, cams_1, cams_aux_1 = [], [], []
    preds_2, cams_2, cams_aux_2 = [], [], []

    model.eval()
    avg_meter = AverageMeter()

    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
            # data fetch
            name, inputs, labels, cls_label = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()
            inputs = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear', align_corners=False)

            # inference
            res = model(inputs, val=True)
            _cls_1, _segs_1, _, _ = res['branch1']
            _cls_2, _segs_2, _, _ = res['branch2']

            # classification results
            cls_pred = (_cls_1 > 0).type(torch.int16)  # branch 1
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score_1": _f1})
            cls_pred = (_cls_2 > 0).type(torch.int16)  # branch 2
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score_2": _f1})

            # CAM results
            _cams_1, _cams_aux_1 = cam_helper.multi_scale_cam2_siamese(
                model, inputs=inputs, scales=args.cam_scales, branch=1
            )
            _cams_2, _cams_aux_2 = cam_helper.multi_scale_cam2_siamese(
                model, inputs=inputs, scales=args.cam_scales, branch=2
            )

            # CAM results -- branch 1
            resized_cam = F.interpolate(_cams_1, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_helper.cam_to_label(
                resized_cam, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre,
                low_thre=args.low_thre, ignore_index=args.ignore_index
            )
            resized_cam_aux = F.interpolate(_cams_aux_1, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label_aux = cam_helper.cam_to_label(
                resized_cam_aux, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre,
                low_thre=args.low_thre, ignore_index=args.ignore_index
            )
            resized_segs = F.interpolate(_segs_1, size=labels.shape[1:], mode='bilinear', align_corners=False)

            preds_1 += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams_1 += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            cams_aux_1 += list(cam_label_aux.cpu().numpy().astype(np.int16))

            # CAM results -- branch 2
            resized_cam = F.interpolate(_cams_2, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_helper.cam_to_label(
                resized_cam, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre,
                low_thre=args.low_thre, ignore_index=args.ignore_index
            )
            resized_cam_aux = F.interpolate(_cams_aux_2, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label_aux = cam_helper.cam_to_label(
                resized_cam_aux, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre,
                low_thre=args.low_thre, ignore_index=args.ignore_index
            )
            resized_segs = F.interpolate(_segs_2, size=labels.shape[1:], mode='bilinear', align_corners=False)

            preds_2 += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams_2 += list(cam_label.cpu().numpy().astype(np.int16))
            cams_aux_2 += list(cam_label_aux.cpu().numpy().astype(np.int16))

    cls_score_1 = avg_meter.pop('cls_score_1')
    cls_score_2 = avg_meter.pop('cls_score_2')

    seg_score_1 = evaluate.scores(gts, preds_1)
    cam_score_1 = evaluate.scores(gts, cams_1)
    cam_aux_score_1 = evaluate.scores(gts, cams_aux_1)

    seg_score_2 = evaluate.scores(gts, preds_2)
    cam_score_2 = evaluate.scores(gts, cams_2)
    cam_aux_score_2 = evaluate.scores(gts, cams_aux_2)

    model.train()

    tab_results, score_item = format_tabs(
        scores=[cam_score_1, cam_aux_score_1, seg_score_1, cam_score_2, cam_aux_score_2, seg_score_2],
        name_list=["CAM_1", "aux_CAM_1", "Seg_1", "CAM_2", "aux_CAM_2", "Seg_2"],
        cat_list=voc.class_list, return_item=True
    )

    if return_item:
        return cls_score_1, cls_score_2, tab_results, score_item

    return cls_score_1, cls_score_2, tab_results


def validate_siamase_coco(model=None, data_loader=None, args=None, return_item=False):
    gts = []
    preds_1, cams_1, cams_aux_1 = [], [], []
    preds_2, cams_2, cams_aux_2 = [], [], []

    model.eval()
    avg_meter = AverageMeter()

    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
            # data fetch
            name, inputs, labels, cls_label = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()
            inputs = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear', align_corners=False)

            # inference
            res = model(inputs, val=True)
            _cls_1, _segs_1, _, _ = res['branch1']
            _cls_2, _segs_2, _, _ = res['branch2']

            # classification results
            cls_pred = (_cls_1 > 0).type(torch.int16)  # branch 1
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score_1": _f1})
            cls_pred = (_cls_2 > 0).type(torch.int16)  # branch 2
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score_2": _f1})

            # CAM results
            _cams_1, _cams_aux_1 = cam_helper.multi_scale_cam2_siamese(
                model, inputs=inputs, scales=args.cam_scales, branch=1
            )
            _cams_2, _cams_aux_2 = cam_helper.multi_scale_cam2_siamese(
                model, inputs=inputs, scales=args.cam_scales, branch=2
            )

            # CAM results -- branch 1
            resized_cam = F.interpolate(_cams_1, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_helper.cam_to_label(
                resized_cam, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre,
                low_thre=args.low_thre, ignore_index=args.ignore_index
            )
            resized_cam_aux = F.interpolate(_cams_aux_1, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label_aux = cam_helper.cam_to_label(
                resized_cam_aux, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre,
                low_thre=args.low_thre, ignore_index=args.ignore_index
            )
            resized_segs = F.interpolate(_segs_1, size=labels.shape[1:], mode='bilinear', align_corners=False)

            preds_1 += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams_1 += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            cams_aux_1 += list(cam_label_aux.cpu().numpy().astype(np.int16))

            # CAM results -- branch 2
            resized_cam = F.interpolate(_cams_2, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_helper.cam_to_label(
                resized_cam, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre,
                low_thre=args.low_thre, ignore_index=args.ignore_index
            )
            resized_cam_aux = F.interpolate(_cams_aux_2, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label_aux = cam_helper.cam_to_label(
                resized_cam_aux, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre,
                low_thre=args.low_thre, ignore_index=args.ignore_index
            )
            resized_segs = F.interpolate(_segs_2, size=labels.shape[1:], mode='bilinear', align_corners=False)

            preds_2 += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams_2 += list(cam_label.cpu().numpy().astype(np.int16))
            cams_aux_2 += list(cam_label_aux.cpu().numpy().astype(np.int16))

    cls_score_1 = avg_meter.pop('cls_score_1')
    cls_score_2 = avg_meter.pop('cls_score_2')

    seg_score_1 = evaluate.scores(gts, preds_1, num_classes=81)
    cam_score_1 = evaluate.scores(gts, cams_1, num_classes=81)
    cam_aux_score_1 = evaluate.scores(gts, cams_aux_1, num_classes=81)

    seg_score_2 = evaluate.scores(gts, preds_2, num_classes=81)
    cam_score_2 = evaluate.scores(gts, cams_2, num_classes=81)
    cam_aux_score_2 = evaluate.scores(gts, cams_aux_2, num_classes=81)

    model.train()

    tab_results, score_item = format_tabs(
        scores=[cam_score_1, cam_aux_score_1, seg_score_1, cam_score_2, cam_aux_score_2, seg_score_2],
        name_list=["CAM_1", "aux_CAM_1", "Seg_1", "CAM_2", "aux_CAM_2", "Seg_2"],
        cat_list=coco.class_list, return_item=True
    )

    if return_item:
        return cls_score_1, cls_score_2, tab_results, score_item

    return cls_score_1, cls_score_2, tab_results


def validate(model=None, data_loader=None, args=None):
    preds, gts, cams, cams_aux = [], [], [], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            inputs = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear', align_corners=False)

            cls, segs, _, _ = model(inputs, val=True)

            cls_pred = (cls > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            _cams, _cams_aux = cam_helper.multi_scale_cam2(model, inputs, args.cam_scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_helper.cam_to_label(resized_cam, cls_label, bkg_thre=args.bkg_thre,
                                                high_thre=args.high_thre,
                                                low_thre=args.low_thre, ignore_index=args.ignore_index)

            resized_cam_aux = F.interpolate(_cams_aux, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label_aux = cam_helper.cam_to_label(resized_cam_aux, cls_label, bkg_thre=args.bkg_thre,
                                                    high_thre=args.high_thre,
                                                    low_thre=args.low_thre, ignore_index=args.ignore_index)

            cls_pred = (cls > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            cams_aux += list(cam_label_aux.cpu().numpy().astype(np.int16))

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds)
    cam_score = evaluate.scores(gts, cams)
    cam_aux_score = evaluate.scores(gts, cams_aux)
    model.train()

    tab_results = format_tabs(
        [cam_score, cam_aux_score, seg_score], name_list=["CAM", "aux_CAM", "Seg_Pred"], cat_list=voc.class_list
    )

    return cls_score, tab_results


def cosine_descent(max_thres, min_thres, step, num_steps):
    if step < 0:
        return max_thres

    if step >= num_steps:
        return min_thres

    interpolation_factor = step / (num_steps - 1)
    weight = max_thres + (min_thres - max_thres) * (1 - np.cos(np.pi * interpolation_factor)) / 2
    return weight
