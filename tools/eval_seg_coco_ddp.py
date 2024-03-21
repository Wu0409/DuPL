import argparse
import os
import sys

sys.path.append(".")

from collections import OrderedDict
import joblib
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from datasets import coco as coco
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate, imutils
from utils.dcrf import DenseCRF
from utils.pyutils import format_tabs
from model.model_dupl import siamese_network
import warnings
from utils.pyutils import AverageMeter, cal_eta, setup_logger, print_progress
import logging

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

local_rank = int(os.environ['LOCAL_RANK'])

#####
# Your weight
####
parser.add_argument("--model_path", default="your_model_dir/checkpoints.pth", type=str, help="model_path")

####
# Dataset
####
parser.add_argument("--img_folder", default='your_coco_dir', type=str, help="dataset folder")
parser.add_argument("--label_folder", default='your_coco_seg_dir', type=str, help="dataset folder")
parser.add_argument("--infer_set", default="val_part", type=str, help="infer_set")
parser.add_argument("--list_folder", default='datasets/coco', type=str, help="train/val/test list file")

parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
parser.add_argument("--num_classes", default=81, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument('--backend', default='nccl')
parser.add_argument("--scales", default=(1.0, 1.25, 1.5), help="multi_scales for seg")


def _validate(pid, model=None, dataset=None, args=None):
    model.eval()
    data_loader = DataLoader(dataset[pid], batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

    with torch.no_grad():
        model.cuda()

        gts, seg_pred_1, seg_pred_2 = [], [], []

        for idx, data in enumerate(data_loader):
            total = len(data_loader)

            if local_rank == 0:
                print_progress(idx, total)

            # if idx >= 500:
            #     break

            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()
            inputs = F.interpolate(inputs, size=[448, 448], mode='bilinear', align_corners=False)
            _, _, h, w = inputs.shape

            seg_list_1, seg_list_2 = [], []

            # 原始尺寸
            _inputs = F.interpolate(inputs, size=[h, w], mode='bilinear', align_corners=False)
            inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

            res = model(inputs_cat)
            cls_1, segs_1, fmap_1, cls_aux_1 = res['branch1']  # branch 1
            cls_2, segs_2, fmap_2, cls_aux_2 = res['branch2']  # branch 2

            # seg = torch.max(segs[:1,...], segs[1:,...].flip(-1))
            seg_1 = segs_1[:1, ...] + segs_1[1:, ...].flip(-1)
            seg_2 = segs_2[:1, ...] + segs_2[1:, ...].flip(-1)

            _, _, h_s, w_s = seg_1.shape

            seg_list_1.append(seg_1)
            seg_list_2.append(seg_2)

            for sc in args.scales:
                if sc != 1.0:
                    _h, _w = int(h * sc), int(w * sc)

                    _inputs = F.interpolate(inputs, size=[_h, _w], mode='bilinear', align_corners=False)
                    inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                    res = model(inputs_cat)
                    cls_1, segs_1, fmap_1, cls_aux_1 = res['branch1']  # branch 1
                    cls_2, segs_2, fmap_2, cls_aux_2 = res['branch2']  # branch 2

                    segs_1 = F.interpolate(segs_1, size=(h_s, w_s), mode='bilinear', align_corners=False)
                    segs_2 = F.interpolate(segs_2, size=(h_s, w_s), mode='bilinear', align_corners=False)

                    # seg = torch.max(segs[:1,...], segs[1:,...].flip(-1))
                    seg_1 = segs_1[:1, ...] + segs_1[1:, ...].flip(-1)
                    seg_2 = segs_2[:1, ...] + segs_2[1:, ...].flip(-1)

                    seg_list_1.append(seg_1)
                    seg_list_2.append(seg_2)

            seg_1 = torch.sum(torch.stack(seg_list_1, dim=0), dim=0)
            seg_2 = torch.sum(torch.stack(seg_list_2, dim=0), dim=0)

            resized_segs_1 = F.interpolate(seg_1, size=labels.shape[1:], mode='bilinear', align_corners=False)
            resized_segs_2 = F.interpolate(seg_2, size=labels.shape[1:], mode='bilinear', align_corners=False)

            seg_pred_1 += list(torch.argmax(resized_segs_1, dim=1).cpu().numpy().astype(np.int16))
            seg_pred_2 += list(torch.argmax(resized_segs_2, dim=1).cpu().numpy().astype(np.int16))

            gts += list(labels.cpu().numpy().astype(np.int16))

            np.save(args.logits_dir + "/" + "branch1" + '/' + name[0] + '.npy', {"msc_seg": seg_1.cpu().numpy()})
            np.save(args.logits_dir + "/" + "branch2" + '/' + name[0] + '.npy', {"msc_seg": seg_2.cpu().numpy()})

    seg_score_1 = evaluate.scores(gts, seg_pred_1, num_classes=81)
    seg_score_2 = evaluate.scores(gts, seg_pred_2, num_classes=81)

    logging.info(format_tabs([seg_score_1, seg_score_2], ["Seg_1", "Seg_2"], cat_list=coco.class_list))

    return seg_score_1, seg_score_2


def crf_proc(branch):
    print("crf post-processing...")

    txt_name = os.path.join(args.list_folder, args.infer_set) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    if "val" in args.infer_set:
        images_path = os.path.join(args.img_folder, "val2014")
        labels_path = os.path.join(args.label_folder, "val2014")
    elif "train" in args.infer_set:
        images_path = os.path.join(args.img_folder, "train2014")
        labels_path = os.path.join(args.label_folder, "train2014")

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
        image = coco.robust_read_image(image_name).astype(np.float32)
        label_name = os.path.join(labels_path, name + ".png")
        if "test" in args.infer_set:
            label = image[:, :, 0]
        else:
            label = imageio.imread(label_name)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)  # [None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)

        imageio.imsave(args.segs_dir + "/" + name + ".png", imutils.encode_cmap(np.squeeze(pred)).astype(np.uint8))
        return pred, label

    n_jobs = int(os.cpu_count() * 0.4)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
        [joblib.delayed(_job)(i) for i in range(len(name_list))])

    preds, gts = zip(*results)

    crf_score = evaluate.scores(gts, preds, num_classes=81)
    logging.info(crf_score)
    return crf_score


def validate(args=None):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.backend, )
    val_dataset = coco.CocoSegDataset(
        img_dir=args.img_folder,
        label_dir=args.label_folder,
        name_list_dir=args.list_folder,
        split=args.infer_set,
        stage='val',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
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
    # new_state_dict.pop("conv.weight")
    # new_state_dict.pop("aux_conv.weight")
    model.to(torch.device(local_rank))
    model.load_state_dict(state_dict=new_state_dict, strict=True)
    model.eval()

    model = DistributedDataParallel(model, device_ids=[local_rank])

    n_gpus = dist.get_world_size()
    split_dataset = [torch.utils.data.Subset(val_dataset, np.arange(i, len(val_dataset), n_gpus)) for i in
                     range(n_gpus)]

    # split_dataset = [torch.utils.data.Subset(val_dataset, np.arange(i, len(val_dataset), n_gpus)) for i in range (n_gpus)]

    seg_score_1, seg_score_2 = _validate(pid=local_rank, model=model, dataset=split_dataset, args=args, )
    torch.cuda.empty_cache()
    torch.distributed.barrier()

    if local_rank == 0:
        if seg_score_1['miou'] > seg_score_2['miou']:
            crf_score = crf_proc('branch1')
        else:
            crf_score = crf_proc('branch2')

    return True


if __name__ == "__main__":

    args = parser.parse_args()

    base_dir = args.model_path.split("checkpoints")[0]
    args.logits_dir = os.path.join(base_dir, "logits", args.infer_set)
    args.segs_dir = os.path.join(base_dir, "segs", args.infer_set)

    os.makedirs(args.segs_dir, exist_ok=True)
    os.makedirs(args.logits_dir, exist_ok=True)
    os.makedirs(args.logits_dir + '/branch1', exist_ok=True)
    os.makedirs(args.logits_dir + '/branch2', exist_ok=True)

    setup_logger(filename=os.path.join(base_dir, 'result_{}.log'.format(local_rank)))

    if local_rank == 0:
        print(args)
    validate(args=args)
