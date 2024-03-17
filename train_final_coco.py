import argparse
import datetime
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
import warnings
from tensorboardX import SummaryWriter
from sklearn.mixture import GaussianMixture

from datasets import coco as coco
from model.losses import get_masked_ptc_loss, get_seg_loss, get_seg_loss_conflict_v2
from model.model_dupl import siamese_network
from model.PAR import PAR
from utils import evaluate, imutils, cam_helper, train_helper, pyutils
from utils.pyutils import AverageMeter, cal_eta, setup_logger

sys.path.append(".")
torch.hub.set_dir("./pretrained")
local_rank = int(os.environ['LOCAL_RANK'])
warnings.filterwarnings("ignore")

### Arguments ###
# ------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--comment", default='_train_coco', type=str, help="comment")

parser.add_argument("--num_workers", default=16, type=int, help="num_workers")
parser.add_argument('--backend', default='nccl')
parser.add_argument("--seed", default=0, type=int, help="fix random seed")

# DIR
parser.add_argument("--work_dir", default="work_dir_coco_wseg", type=str, help="work_dir_voc_wseg")
parser.add_argument("--img_folder", default='/your_voc_dir', type=str, help="dataset folder")
parser.add_argument("--label_folder", default='/home/wyc/Dataset/MSCOCO/SegmentationClass', type=str,
                    help="dataset folder")
parser.add_argument("--list_folder", default='datasets/coco', type=str, help="train/val/test list file")

# DATASET
parser.add_argument("--train_set", default="train", type=str, help="training split")
parser.add_argument("--val_set", default="val_part", type=str, help="validation split")
parser.add_argument("--scales", default=(0.5, 2), help="random rescale in training")

# MODEL
parser.add_argument("--backbone", default='deit_base_patch16_224', type=str, help="backbone")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")

# TRAINING
parser.add_argument("--samples_per_gpu", default=1, type=int, help="samples_per_gpu")
parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--warmup_iters", default=1500, type=int, help="warmup_iters")
parser.add_argument("--lr", default=6e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="poweer factor for poly scheduler")
parser.add_argument("--save_ckpt", default=True, type=bool, help="save_ckpt")

# TASK
parser.add_argument("--num_classes", default=81, type=int, help="number of classes")
parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")

parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--max_iters", default=80000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=200, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=4000, type=int, help="validation iters")
parser.add_argument("--cam_iters", default=8000, type=int)

parser.add_argument("--high_thre", default=0.65, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.45, type=float, help="bkg_score")
parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5), help="multi_scales for cam")

# GMM
parser.add_argument("--gmm_iters", default=32000, type=int)
parser.add_argument("--gmm_valid_thre", default=1.0, type=float)
parser.add_argument("--gamma", default=0.95, type=float)


### SEED ###
# ------------------------------------------------------------------------------------------------------
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


### Train ###
# ------------------------------------------------------------------------------------------------------
def train(args=None):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.backend, timeout=datetime.timedelta(seconds=5400))
    device = torch.device(local_rank)

    if local_rank == 0:
        tblogger = SummaryWriter(comment=args.comment)  # debug

    logging.info("Total gpus: %d, samples per gpu: %d..." % (dist.get_world_size(), args.samples_per_gpu))

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    # Dataset
    # ------------------------------------------------------------------------------------------------------
    train_dataset = coco.CocoClsDataset(
        img_dir=args.img_folder, label_dir=args.label_folder, name_list_dir=args.list_folder, split=args.train_set,
        stage='train', aug=True, rescale_range=args.scales, crop_size=args.crop_size, img_fliplr=True,
        ignore_index=args.ignore_index, num_classes=args.num_classes,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=args.samples_per_gpu, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, sampler=train_sampler, prefetch_factor=4
    )
    train_sampler.set_epoch(np.random.randint(args.max_iters))
    train_loader_iter = iter(train_loader)

    val_dataset = coco.CocoSegDataset(
        img_dir=args.img_folder, label_dir=args.label_folder, name_list_dir=args.list_folder, split=args.val_set,
        stage='val', aug=False, ignore_index=args.ignore_index, num_classes=args.num_classes,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=False
    )

    # Model, modules and optimizer
    # ------------------------------------------------------------------------------------------------------
    model = siamese_network(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        aux_layer=9  # for coco
    )

    param_groups = model.get_param_groups()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    optim = train_helper.get_optimizer(param_groups, args)
    logging.info('\nOptimizer: \n%s' % optim)

    par = PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).cuda()

    high_thres_start = torch.ones(80, device=device) * args.high_thre
    high_thres_target = torch.ones(80).to(device) * 0.55

    ce_criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index, reduction='none').to(device)

    # Training
    # ------------------------------------------------------------------------------------------------------
    avg_meter = AverageMeter()

    for n_iter in range(args.max_iters):

        if local_rank == 0:
            pyutils.print_progress(n_iter, args.max_iters)

        try:
            img_name, inputs, cls_label, img_box, _ = next(train_loader_iter)
        except:
            train_sampler.set_epoch(np.random.randint(args.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_label, img_box, _ = next(train_loader_iter)

        inputs = inputs.to(device, non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_label = cls_label.to(device, non_blocking=True)

        # strong perturb
        inputs_aug = imutils.augment_data_strong(inputs_denorm.clone(), n=5, m=10)
        inputs_denorm_aug = imutils.denormalize_img2(inputs_aug.clone())

        if n_iter < args.cam_iters:
            # Inference
            # ------------------------------------------------------------------------------------------------------
            cams_1, cams_aux_1 = cam_helper.multi_scale_cam2_siamese(
                model, inputs=inputs, scales=args.cam_scales, branch=1
            )
            cams_2, cams_aux_2 = cam_helper.multi_scale_cam2_siamese(
                model, inputs=inputs, scales=args.cam_scales, branch=2
            )

            res = model(inputs)
            cls_1, segs_1, fmap_1, cls_aux_1 = res['branch1']  # branch 1
            cls_2, segs_2, fmap_2, cls_aux_2 = res['branch2']  # branch 2

            # Classification Loss
            # ------------------------------------------------------------------------------------------------------
            cls_loss_fmap_1 = F.multilabel_soft_margin_loss(cls_1, cls_label)
            cls_loss_aux_1 = F.multilabel_soft_margin_loss(cls_aux_1, cls_label)

            cls_loss_fmap_2 = F.multilabel_soft_margin_loss(cls_2, cls_label)
            cls_loss_aux_2 = F.multilabel_soft_margin_loss(cls_aux_2, cls_label)

            cls_loss = cls_loss_fmap_1 + cls_loss_aux_1 + cls_loss_fmap_2 + cls_loss_aux_2

            # PTC Loss
            # ------------------------------------------------------------------------------------------------------
            ptc_loss = torch.ones(1).cuda()

            # Segmentation Loss
            # ------------------------------------------------------------------------------------------------------
            seg_loss = torch.ones(1).cuda()

            # Discrepancy loss
            # ------------------------------------------------------------------------------------------------------
            fmap_1_flat = fmap_1.view(fmap_1.shape[0], fmap_1.shape[1], -1)
            fmap_2_flat = fmap_2.view(fmap_2.shape[0], fmap_2.shape[1], -1)

            cos_simi = nn.CosineSimilarity(dim=-1, eps=1e-6)
            sim_loss_1 = 1 + cos_simi(fmap_1_flat.detach(), fmap_2_flat).mean()
            sim_loss_2 = 1 + cos_simi(fmap_2_flat.detach(), fmap_1_flat).mean()

            sim_loss = sim_loss_1 + sim_loss_2

            # Regularization Loss
            # ------------------------------------------------------------------------------------------------------
            reg_loss = torch.zeros(1).cuda()

        else:
            # generate high threshold mask
            # ------------------------------------------------------------------------------------------------------
            high_thres = train_helper.cosine_descent(
                high_thres_start, high_thres_target, n_iter - 12000, args.max_iters - 12000
            )
            b, _, h, w = inputs.shape
            high_thres_mask_list = []
            high_thres_list = []
            for i in range(args.samples_per_gpu):
                high_thres_i = torch.max(high_thres[torch.nonzero(cls_label[i]).squeeze(-1)])
                high_thres_list.append(high_thres_i)
                high_thres_mask_list.append(torch.ones((h, w), device='cuda') * high_thres_i)

            high_thres = torch.stack(high_thres_list, dim=0)
            high_thres_mask = torch.stack(high_thres_mask_list, dim=0).unsqueeze(1)

            # Inference
            # ------------------------------------------------------------------------------------------------------
            cams_1, cams_aux_1 = cam_helper.multi_scale_cam2_siamese(
                model, inputs=inputs, scales=args.cam_scales, branch=1
            )
            cams_2, cams_aux_2 = cam_helper.multi_scale_cam2_siamese(
                model, inputs=inputs, scales=args.cam_scales, branch=2
            )

            if n_iter < args.gmm_iters:
                res = model(inputs)
                cls_1, segs_1, fmap_1, cls_aux_1 = res['branch1']  # branch 1
                cls_2, segs_2, fmap_2, cls_aux_2 = res['branch2']  # branch 2
            else:
                inputs_cat = torch.cat([inputs, inputs_aug], dim=0)
                res = model(inputs_cat, need_sp=True)
                cls_1, segs_1, fmap_1, cls_aux_1 = res['branch1']  # branch 1
                cls_2, segs_2, fmap_2, cls_aux_2 = res['branch2']  # branch 2
                segs_1_aug, segs_2_aug = res['branch1_aug'], res['branch2_aug']

            # Classification Loss
            # ------------------------------------------------------------------------------------------------------
            cls_loss_fmap_1 = F.multilabel_soft_margin_loss(cls_1, cls_label)
            cls_loss_aux_1 = F.multilabel_soft_margin_loss(cls_aux_1, cls_label)

            cls_loss_fmap_2 = F.multilabel_soft_margin_loss(cls_2, cls_label)
            cls_loss_aux_2 = F.multilabel_soft_margin_loss(cls_aux_2, cls_label)

            cls_loss = cls_loss_fmap_1 + cls_loss_aux_1 + cls_loss_fmap_2 + cls_loss_aux_2

            # PTC Loss
            # ------------------------------------------------------------------------------------------------------
            resized_cams_aux_1 = F.interpolate(cams_aux_1, size=fmap_1.shape[2:], mode="bilinear", align_corners=False)
            _, pseudo_label_aux_1 = cam_helper.cam_to_label_dynamic_cls(
                resized_cams_aux_1.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True,
                bkg_thre=args.bkg_thre, high_thre=high_thres, low_thre=args.low_thre,
                ignore_index=args.ignore_index
            )

            resized_cams_aux_2 = F.interpolate(cams_aux_2, size=fmap_2.shape[2:], mode="bilinear", align_corners=False)
            _, pseudo_label_aux_2 = cam_helper.cam_to_label_dynamic_cls(
                resized_cams_aux_2.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True,
                bkg_thre=args.bkg_thre, high_thre=high_thres, low_thre=args.low_thre,
                ignore_index=args.ignore_index
            )

            aff_mask_1 = cam_helper.label_to_aff_mask(pseudo_label_aux_1)
            aff_mask_2 = cam_helper.label_to_aff_mask(pseudo_label_aux_2)
            ptc_loss_1 = get_masked_ptc_loss(fmap_1, aff_mask_1)
            ptc_loss_2 = get_masked_ptc_loss(fmap_2, aff_mask_2)

            ptc_loss = ptc_loss_1 + ptc_loss_2

            # Segmentation Loss
            # ------------------------------------------------------------------------------------------------------
            b, c, h, w = cams_1.shape
            cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])

            if n_iter <= 12000:
                refined_pseudo_label_1 = cam_helper.refine_cams_with_bkg_v2(
                    par, inputs_denorm, cams=cams_aux_1.detach() * cls_label_rep, cls_labels=cls_label,
                    high_thre=args.high_thre, low_thre=args.low_thre,
                    ignore_index=args.ignore_index, img_box=img_box
                )
                refined_pseudo_label_2 = cam_helper.refine_cams_with_bkg_v2(
                    par, inputs_denorm, cams=cams_aux_2.detach() * cls_label_rep, cls_labels=cls_label,
                    high_thre=args.high_thre, low_thre=args.low_thre,
                    ignore_index=args.ignore_index, img_box=img_box
                )
            else:
                refined_pseudo_label_1 = cam_helper.refine_cams_with_dynamic_thres(
                    par, inputs_denorm, cams=cams_1.detach() * cls_label_rep, cls_labels=cls_label,
                    high_thre_map=high_thres_mask, low_thre=args.low_thre,
                    ignore_index=args.ignore_index, img_box=img_box
                )
                refined_pseudo_label_2 = cam_helper.refine_cams_with_dynamic_thres(
                    par, inputs_denorm, cams=cams_2.detach() * cls_label_rep, cls_labels=cls_label,
                    high_thre_map=high_thres_mask, low_thre=args.low_thre,
                    ignore_index=args.ignore_index, img_box=img_box
                )

            segs_1 = F.interpolate(segs_1, size=refined_pseudo_label_1.shape[1:], mode='bilinear', align_corners=False)
            segs_2 = F.interpolate(segs_2, size=refined_pseudo_label_2.shape[1:], mode='bilinear', align_corners=False)

            # Warm-up
            if n_iter < args.gmm_iters:
                # Direct backward seg loss
                seg_loss_1 = get_seg_loss(segs_1, refined_pseudo_label_2.type(torch.long))
                seg_loss_2 = get_seg_loss(segs_2, refined_pseudo_label_1.type(torch.long))

                seg_loss = seg_loss_1 + seg_loss_2

                reg_loss = seg_loss_1 * 0 + seg_loss_2 * 0

            else:
                # Use GMM to filter noise labels
                seg_loss_1 = ce_criterion(segs_1, refined_pseudo_label_1.type(torch.long)).detach()
                seg_loss_2 = ce_criterion(segs_2, refined_pseudo_label_2.type(torch.long)).detach()

                roi_mask_1 = (refined_pseudo_label_1 != 0).bool() & (refined_pseudo_label_1 != 255).bool()
                roi_mask_2 = (refined_pseudo_label_2 != 0).bool() & (refined_pseudo_label_2 != 255).bool()

                for i in range(args.samples_per_gpu):
                    seg_loss_1_m = seg_loss_1[i][roi_mask_1[i]]
                    seg_loss_2_m = seg_loss_2[i][roi_mask_2[i]]

                    # GMM 1
                    if (seg_loss_1_m > 0.1).sum().item() > 1000:
                        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=0)
                        gmm.fit(seg_loss_1_m[seg_loss_1_m > 0.1].unsqueeze(-1).cpu().detach().numpy())
                        means = gmm.means_

                        if abs(means[0, 0] - means[1, 0]) > args.gmm_valid_thre:  # two normal distributions
                            noise_idx = gmm.means_.argmax()
                            prob = gmm.predict_proba(seg_loss_1[i].view(-1).unsqueeze(-1).cpu().detach().numpy())
                            noise_mask = torch.tensor(prob[:, noise_idx] > args.gamma, device=device).reshape(h, w)
                            noise_mask = noise_mask & (refined_pseudo_label_1[i] != 0).bool()
                            refined_pseudo_label_1[i][noise_mask] = 255

                    # GMM 2
                    if (seg_loss_2_m > 0.1).sum().item() > 1000:
                        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=0)
                        gmm.fit(seg_loss_2_m[seg_loss_2_m > 0.1].unsqueeze(-1).cpu().detach().numpy())
                        means = gmm.means_

                        if abs(means[0, 0] - means[1, 0]) > args.gmm_valid_thre:  # two normal distributions
                            noise_idx = gmm.means_.argmax()
                            prob = gmm.predict_proba(seg_loss_2[i].view(-1).unsqueeze(-1).cpu().detach().numpy())
                            noise_mask = torch.tensor(prob[:, noise_idx] > args.gamma, device=device).reshape(h, w)
                            noise_mask = noise_mask & (refined_pseudo_label_2[i] != 0).bool()
                            refined_pseudo_label_2[i][noise_mask] = 255

                seg_loss_1 = get_seg_loss(
                    segs_1, refined_pseudo_label_2.type(torch.long), ignore_index=args.ignore_index
                )
                seg_loss_2 = get_seg_loss(
                    segs_2, refined_pseudo_label_1.type(torch.long), ignore_index=args.ignore_index
                )
                seg_loss = seg_loss_1 + seg_loss_2

                # After warnup -> consistent regularization
                # Consistency Regularization loss
                # ------------------------------------------------------------------------------------------------------
                segs_1_aug = torch.flip(segs_1_aug, dims=[3])
                segs_2_aug = torch.flip(segs_2_aug, dims=[3])
                segs_1_aug = F.interpolate(
                    segs_1_aug, size=inputs_denorm.shape[2:], mode='bilinear', align_corners=False
                )
                segs_2_aug = F.interpolate(
                    segs_2_aug, size=inputs_denorm.shape[2:], mode='bilinear', align_corners=False
                )

                pseudo_seg_1 = segs_1.detach().data.max(1)[1]
                pseudo_seg_2 = segs_2.detach().data.max(1)[1]

                confidence_map_1 = torch.softmax(segs_1.detach(), dim=1).max(1)[0]
                confidence_map_2 = torch.softmax(segs_2.detach(), dim=1).max(1)[0]

                uncertain_mask_1 = (refined_pseudo_label_2 == args.ignore_index).bool() & (confidence_map_1 > 0.9)
                uncertain_mask_2 = (refined_pseudo_label_1 == args.ignore_index).bool() & (confidence_map_2 > 0.9)

                pseudo_seg_1[~uncertain_mask_1] = args.ignore_index
                pseudo_seg_2[~uncertain_mask_2] = args.ignore_index

                reg_loss_1, reg_loss_2 = seg_loss_1 * 0.0, seg_loss_2 * 0.0

                if uncertain_mask_1.sum() > 0:
                    reg_loss_1 = (ce_criterion(segs_1_aug, pseudo_seg_1)).sum() / uncertain_mask_1.sum()

                if uncertain_mask_2.sum() > 0:
                    reg_loss_2 = (ce_criterion(segs_2_aug, pseudo_seg_2)).sum() / uncertain_mask_2.sum()

                reg_loss = reg_loss_1 + reg_loss_2

            # Discrepancy loss
            # ------------------------------------------------------------------------------------------------------
            fmap_1_flat = fmap_1.view(fmap_1.shape[0], fmap_1.shape[1], -1)
            fmap_2_flat = fmap_2.view(fmap_2.shape[0], fmap_2.shape[1], -1)

            cos_simi = nn.CosineSimilarity(dim=-1, eps=1e-6)
            sim_loss_1 = 1 + cos_simi(fmap_1_flat.detach(), fmap_2_flat).mean()
            sim_loss_2 = 1 + cos_simi(fmap_2_flat.detach(), fmap_1_flat).mean()

            sim_loss = sim_loss_1 + sim_loss_2

        # Total Loss
        # ------------------------------------------------------------------------------------------------------
        if n_iter <= 8000:
            loss = 1.0 * cls_loss + 0.0 * ptc_loss + 0.0 * seg_loss + 0.0 * sim_loss
        elif n_iter <= 12000:
            loss = 1.0 * cls_loss + 0.0 * ptc_loss + 0.2 * seg_loss + 0.05 * sim_loss
        elif n_iter <= args.gmm_iters:
            loss = 1.0 * cls_loss + 0.2 * ptc_loss + 0.2 * seg_loss + 0.05 * sim_loss + 0.05 * reg_loss
        else:
            loss = 1.0 * cls_loss + 0.2 * ptc_loss + 0.2 * seg_loss + 0.05 * sim_loss + 0.05 * reg_loss

        cls_pred = (cls_1 > 0).type(torch.int16)
        cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])

        avg_meter.add({
            'cls_score': cls_score.item(),
            'cls_loss': cls_loss.item(),
            'ptc_loss': ptc_loss.item(),
            'seg_loss': seg_loss.item(),
            'sim_loss': sim_loss.item(),
            'reg_loss': reg_loss.item(),
        })

        optim.zero_grad()
        loss.backward()
        optim.step()

        # Logger
        # ------------------------------------------------------------------------------------------------------
        if (n_iter + 1) % args.log_iters == 0:
            delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
            cur_lr = optim.param_groups[0]['lr']

            if local_rank == 0:
                cls_loss_sc, ptc_loss_sc, seg_loss_sc, sim_loss_sc, reg_loss_sc = (
                    avg_meter.pop('cls_loss'), avg_meter.pop('ptc_loss'), avg_meter.pop('seg_loss'),
                    avg_meter.pop('sim_loss'), avg_meter.pop('reg_loss')
                )

                logging.info(
                    "\nIter: %d; Elasped: %s; ETA: %s; LR: %.3e; "
                    "cls_loss: %.4f | ptc_loss: %.4f | seg_loss: %.4f | sim_loss: %.4f"
                    % (n_iter + 1, delta, eta, cur_lr, cls_loss_sc, ptc_loss_sc, seg_loss_sc, sim_loss_sc)
                )

                # tensorboard
                tblogger.add_scalar('Loss/ptc_loss', ptc_loss_sc, n_iter + 1)
                tblogger.add_scalar('Loss/seg_loss', seg_loss_sc, n_iter + 1)
                tblogger.add_scalar('Loss/sim_loss', sim_loss_sc, n_iter + 1)
                tblogger.add_scalar('Loss/reg_loss', reg_loss_sc, n_iter + 1)

                b, c, h, w = cams_1.shape
                cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
                valid_cam_1 = cls_label_rep * cams_1
                valid_cam_2 = cls_label_rep * cams_2

                grid_imgs, grid_cam_1 = imutils.tensorboard_image(imgs=inputs.clone(), cam=valid_cam_1)
                grid_imgs, grid_cam_2 = imutils.tensorboard_image(imgs=inputs.clone(), cam=valid_cam_2)

                tblogger.add_image("CAM/inputs", grid_imgs, global_step=n_iter + 1)
                tblogger.add_image("CAM/cams_1", grid_cam_1, global_step=n_iter + 1)
                tblogger.add_image("CAM/cams_2", grid_cam_2, global_step=n_iter + 1)

        # Evaluation
        # ------------------------------------------------------------------------------------------------------
        if (n_iter + 1) % args.eval_iters == 0:
            # ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
            ckpt_name = os.path.join(args.ckpt_dir, "checkpoint.pth")
            if local_rank == 0:
                logging.info('Validating...')

                if args.save_ckpt:
                    torch.save(model.state_dict(), ckpt_name)

                val_cls_score_1, val_cls_score_2, tab_results, score_item = train_helper.validate_siamase_coco(
                    model=model, data_loader=val_loader, args=args, return_item=True
                )

                cam_score_1, cam_aux_score_1, seg_score_1, cam_score_2, cam_aux_score_2, seg_score_2 = score_item

                logging.info("val cls score of branch #1: %.6f" % (val_cls_score_1))
                logging.info("val cls score of branch #2: %.6f" % (val_cls_score_2))
                logging.info("\n" + tab_results)

                tblogger.add_scalar('CAM/cam_1', cam_score_1, n_iter + 1)
                tblogger.add_scalar('CAM/cam_2', cam_score_2, n_iter + 1)
                tblogger.add_scalar('CAM_AUX/cam_aux_1', cam_aux_score_1, n_iter + 1)
                tblogger.add_scalar('CAM_AUX/cam_aux_2', cam_aux_score_2, n_iter + 1)
                tblogger.add_scalar('SEG/seg_1', seg_score_1, n_iter + 1)
                tblogger.add_scalar('SEG/seg_2', seg_score_2, n_iter + 1)

    return True


if __name__ == "__main__":
    args = parser.parse_args()

    timestamp = "{0:%Y-%m-%d-%H-%M-%S-%f}".format(datetime.datetime.now()) + args.comment
    args.work_dir = os.path.join(args.work_dir, timestamp)
    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    args.pred_dir = os.path.join(args.work_dir, "predictions")

    if local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)

        setup_logger(filename=os.path.join(args.work_dir, 'train.log'))
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s" % (torch.cuda.get_device_name(0)))
        logging.info('\nargs: %s' % args)

    ## fix random seed
    setup_seed(args.seed)
    train(args=args)
