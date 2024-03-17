import argparse
import os
import sys

sys.path.append(".")

from collections import OrderedDict
import imageio
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import voc
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate, imutils
from utils.dcrf import DenseCRF
from utils.pyutils import format_tabs
from model.model_dupl import siamese_network
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--infer_set", default="val", type=str, help="infer_set")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--model_path", default="your model dir", type=str, help="model_path")

parser.add_argument("--backbone", default='deit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--data_folder", default='your_voc_dir', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=21, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--scales", default=(1.0, 1.5, 1.25), help="multi_scales for seg")


def _validate(model=None, data_loader=None, args=None):
    model.eval()

    with torch.no_grad(), torch.cuda.device(0):
        model.cuda()

        gts, seg_pred_1, seg_pred_2 = [], [], []

        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            _, _, h, w = inputs.shape
            seg_list_1, seg_list_2 = [], []
            for sc in args.scales:
                _h, _w = int(h * sc), int(w * sc)

                _inputs = F.interpolate(inputs, size=[_h, _w], mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                res = model(inputs_cat)
                cls_1, segs_1, fmap_1, cls_aux_1 = res['branch1']  # branch 1
                cls_2, segs_2, fmap_2, cls_aux_2 = res['branch2']  # branch 2

                segs_1 = F.interpolate(segs_1, size=labels.shape[1:], mode='bilinear', align_corners=False)
                segs_2 = F.interpolate(segs_2, size=labels.shape[1:], mode='bilinear', align_corners=False)

                # seg = torch.max(segs[:1,...], segs[1:,...].flip(-1))
                seg_1 = segs_1[:1, ...] + segs_1[1:, ...].flip(-1)
                seg_2 = segs_2[:1, ...] + segs_2[1:, ...].flip(-1)

                seg_list_1.append(seg_1)
                seg_list_2.append(seg_2)

            seg_1 = torch.max(torch.stack(seg_list_1, dim=0), dim=0)[0]
            seg_2 = torch.max(torch.stack(seg_list_2, dim=0), dim=0)[0]

            seg_pred_1 += list(torch.argmax(seg_1, dim=1).cpu().numpy().astype(np.int16))
            seg_pred_2 += list(torch.argmax(seg_2, dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

            np.save(args.logits_dir + "/" + "branch1" + '/' + name[0] + '.npy', {"msc_seg": seg_1.cpu().numpy()})
            np.save(args.logits_dir + "/" + "branch2" + '/' + name[0] + '.npy', {"msc_seg": seg_2.cpu().numpy()})

    seg_score_1 = evaluate.scores(gts, seg_pred_1)
    seg_score_2 = evaluate.scores(gts, seg_pred_2)

    print(format_tabs([seg_score_1, seg_score_2], ["Seg_1", "Seg_2"], cat_list=voc.class_list))

    return seg_score_1, seg_score_2


def crf_proc(branch):
    print("crf post-processing...")

    txt_name = os.path.join(args.list_folder, args.infer_set) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    images_path = os.path.join(args.data_folder, 'JPEGImages', )
    labels_path = os.path.join(args.data_folder, 'SegmentationClassAug')

    post_processor = DenseCRF(
        iter_max=10,  # 10
        pos_xy_std=1,  # 3
        pos_w=1,  # 3
        bi_xy_std=121,  # 121, 140
        bi_rgb_std=5,  # 5, 5
        bi_w=4,  # 4, 5
    )

    def _job(i):

        name = name_list[i]

        logit_name = args.logits_dir + "/" + branch + "/" + name + ".npy"

        logit = np.load(logit_name, allow_pickle=True).item()
        logit = logit['msc_seg']

        image_name = os.path.join(images_path, name + ".jpg")
        image = imageio.imread(image_name).astype(np.float32)
        label_name = os.path.join(labels_path, name + ".png")
        if "test" in args.infer_set:
            label = image[:, :, 0]
        else:
            label = imageio.imread(label_name)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)  # [None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()
        # prob = logit[0]

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)

        # print(pred.shape)
        imageio.imsave(args.segs_dir + "/" + name + ".png", np.squeeze(pred).astype(np.uint8))
        imageio.imsave(args.segs_rgb_dir + "/" + name + ".png", imutils.encode_cmap(np.squeeze(pred)).astype(np.uint8))
        return pred, label

    n_jobs = int(os.cpu_count() * 0.8)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
        [joblib.delayed(_job)(i) for i in range(len(name_list))])

    preds, gts = zip(*results)

    crf_score = evaluate.scores(gts, preds)
    print(format_tabs([crf_score], ["seg_crf"], cat_list=voc.class_list))
    return crf_score


def validate(args=None):
    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder, name_list_dir=args.list_folder, split=args.infer_set, stage='val', aug=False,
        ignore_index=args.ignore_index, num_classes=args.num_classes,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=False, drop_last=False
    )

    model = siamese_network(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=False,
        aux_layer=-3
    )

    trained_state_dict = torch.load(args.model_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v
    model.load_state_dict(state_dict=new_state_dict, strict=True)
    model.eval()

    seg_score_1, seg_score_2 = _validate(model=model, data_loader=val_loader, args=args)
    torch.cuda.empty_cache()

    print(seg_score_1)

    if seg_score_1['miou'] > seg_score_2['miou']:
        crf_score = crf_proc('branch1')
    else:
        crf_score = crf_proc('branch2')

    return True


if __name__ == "__main__":
    args = parser.parse_args()

    base_dir = args.model_path.split("checkpoints")[0]
    args.logits_dir = os.path.join(base_dir, "segs/logits", args.infer_set)
    args.segs_dir = os.path.join(base_dir, "segs/seg_preds", args.infer_set)
    args.segs_rgb_dir = os.path.join(base_dir, "segs/seg_preds_rgb", args.infer_set)

    os.makedirs(args.segs_dir, exist_ok=True)
    os.makedirs(args.segs_rgb_dir, exist_ok=True)
    os.makedirs(args.logits_dir, exist_ok=True)
    os.makedirs(args.logits_dir + '/branch1', exist_ok=True)
    os.makedirs(args.logits_dir + '/branch2', exist_ok=True)

    print(args)
    validate(args=args)
